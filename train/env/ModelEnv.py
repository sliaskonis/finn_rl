import copy
import math
import torch
import numpy as np
import torch.nn as nn
from enum import IntEnum
from copy import deepcopy

import brevitas.nn as qnn
import brevitas.export as bo
from brevitas.graph.utils import get_module
from brevitas.graph.quantize import preprocess_for_quantize

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType

import gymnasium as gym
from gymnasium import spaces

from ..quantizer import Quantizer
from ..finetune import Finetuner
from ..exporter.Exporter import (
	tidy_up,
	name_nodes,
	preprocessing,
	postprocessing,
	make_input_channels_last,
	qonnx_to_finn,
	create_dataflow_partition,
	specialize_layers,
	set_folding,
	streamline_resnet,
	convert_to_hw_resnet,
	streamline_lenet,
	convert_to_hw_lenet,
	streamline_mobilenet,
	convert_to_hw_mobilenet,
)

from ..utils import measure_model

# Global variable
count_reward = 0

platform_path = './platforms'
platform_files = {
    'U250': f'{platform_path}/u250.json',
    'ZCU102': f'{platform_path}/zcu102.json'
}

streamline_functions = {
	'LeNet5' : streamline_lenet,
	'resnet18' : streamline_resnet,
	'resnet34' : streamline_resnet,
	'resnet50' : streamline_resnet,
	'resnet100' : streamline_resnet,
	'resnet152' : streamline_resnet,
	'MobileNet' : streamline_mobilenet
}

convert_to_hw_functions = {
	'LeNet5' : convert_to_hw_lenet,
	'resnet18' : convert_to_hw_resnet,
	'resnet34' : convert_to_hw_resnet,
	'resnet50' : convert_to_hw_resnet,
	'resnet100' : convert_to_hw_resnet,
	'resnet152' : convert_to_hw_resnet,
	'MobileNet' : convert_to_hw_mobilenet
}

class LayerTypes(IntEnum):
	LINEAR = 0
	MHA = 1
	CONV1D = 2
	CONV2D = 3
	CONVTRANSPOSE1D = 4
	CONVTRANSPOSE2D = 5

class ActTypes(IntEnum):
	RELU = 0
	RELU6 = 1
	SIGMOID = 2

class ModelEnv(gym.Env):
	def __init__(self, args, model_config, testing = False):
		self.args = args
		self.testing = testing

		self.observation_space = spaces.Box(low = 0.0, high = 1.0, shape=(6, ), dtype = np.float32)
		self.action_space = spaces.Box(low = -1.0, high = 1.0, shape = (1, ), dtype = np.float32)

		self.quantizable_acts = [
			nn.ReLU,
			nn.ReLU6,
			nn.Sigmoid,
			qnn.QuantReLU,
			qnn.QuantSigmoid
		]

		self.quantizable_layers = [
			nn.Linear,
			nn.MultiheadAttention,
			nn.Conv1d,
			nn.Conv2d,
			nn.ConvTranspose1d,
			nn.ConvTranspose2d,
			qnn.QuantLinear,
			qnn.QuantMultiheadAttention,
			qnn.QuantConv1d,
			qnn.QuantConv2d,
			qnn.QuantConvTranspose1d,
			qnn.QuantConvTranspose2d
		]

		self.finetuner = Finetuner(args, model_config)
		self.model = copy.deepcopy(self.finetuner.model)
		self.model_config = model_config
		self.orig_model = copy.deepcopy(self.model)
		self.strategy = [] # quantization strategy
		self.cur_ind = 0

		self.min_bit = args.min_bit
		self.max_bit = args.max_bit
		self.last_action = self.max_bit
		# init reward
		self.best_reward = -math.inf
		self.qonnx_to_pytorch = {}
		
		self.build_state_embedding() # build the states for each layer
		self.index_to_quantize = self.quantizable_idx[self.cur_ind]

		self.quantizer = Quantizer(
			args.weight_bit_width,
			args.act_bit_width,
			args.residual_bit_width
		)
	
		self.orig_acc = self.finetuner.orig_acc
		self.max_fps = 0.0
		self.cur_fps = 0.0
		self.max_acc = 0.0
		self.cur_acc = 0.0
		self.avg_util = 0.0

		print('Original Accuracy: {:.3f}%'.format(self.orig_acc * 100))

		# find maximum fps (minimum bitwidth in all layers and maximum folding
		if not self.testing:
			self.maximum_fps()

	def build_state_embedding(self):
		self.model = preprocess_for_quantize(self.model)

		measure_model(self.model, self.model_config['center_crop_shape'], 
					self.model_config['center_crop_shape'], self.finetuner.in_channels)
	
		self.quantizable_idx = []
		self.bound_list = []
		self.num_quant_acts = 0
		self.quantizable_nodes = []
		layer_embedding = []

		# Activations first
		for i, node in enumerate(self.model.graph.nodes):
			this_state = []
			if node.op == 'call_module':
				module = get_module(self.model, node.target)
				if type(module) in self.quantizable_acts:
					self.quantizable_idx.append(i)

					# for activations, increase by 1 the minimum bit (activations should be at least 2 bits, so that 0 is represented (padding for conv))
					if self.min_bit == 1:
						self.bound_list.append((self.min_bit + 1, self.max_bit))
					else:
						self.bound_list.append((self.min_bit, self.max_bit))

					this_state.append([i])
					this_state.append([1])
					
					if type(module) == nn.ReLU or type(module) == qnn.QuantReLU:
						this_state.append([ActTypes.RELU])
					elif type(module) == nn.ReLU6 or type(module) == qnn.QuantReLU:
						this_state.append([ActTypes.RELU6])
					elif type(module) == nn.Sigmoid or type(module) == qnn.QuantSigmoid:
						this_state.append([ActTypes.SIGMOID])
			   
					this_state.append([module.flops])
					this_state.append([module.params])
					this_state.append([1.0])
					layer_embedding.append(np.hstack(this_state))
		
		# number of activation layers
		self.num_quant_acts = len(self.quantizable_idx)

		# Compute layers
		for i, node in enumerate(self.model.graph.nodes):
			this_state = []
			if node.op == 'call_module':
				module = get_module(self.model, node.target)
				if type(module) in self.quantizable_layers:
					self.quantizable_idx.append(i)
					self.bound_list.append((self.min_bit, self.max_bit))
					this_state.append([i])
					this_state.append([0])

					if type(module) == nn.Linear or type(module) == qnn.QuantLinear:
						this_state.append([LayerTypes.LINEAR])
					elif type(module) == nn.MultiheadAttention or type(module) == qnn.QuantMultiheadAttention:
						this_state.append([LayerTypes.MHA])
					elif type(module) == nn.Conv1d or type(module) == qnn.QuantConv1d:
						this_state.append([LayerTypes.CONV1D])
					elif type(module) == nn.Conv2d or type(module) == qnn.QuantConv2d:
						this_state.append([LayerTypes.CONV2D])
					elif type(module) == nn.ConvTranpose1d or type(module) == qnn.QuantConvTranspose1d:
						this_state.append([LayerTypes.CONVTRANSPOSE1D])
					elif type(module) == nn.ConvTranspose2d or type(module) == qnn.QuantConvTranspose2d:
						this_state.append([LayerTypes.CONVTRANSPOSE2D])

					this_state.append([module.flops])
					this_state.append([module.params])
					this_state.append([1.0])
					layer_embedding.append(np.hstack(this_state))

					if type(module) == nn.Linear or type(module) == qnn.QuantLinear or type(module) == nn.Conv2d or type(module) == qnn.QuantConv2d:
						# count how many linear layer or convolution layers preceed it, because those will generate MVAU or VVAU
						num_prec = 0
						for j, n in enumerate(self.model.graph.nodes):
							if n.op == 'call_module':
								m = get_module(self.model, n.target)
								if m == module:
									break
								if type(m) in [nn.Linear, nn.Conv2d, qnn.QuantLinear, qnn.QuantConv2d]:
									num_prec+=1
								
						self.qonnx_to_pytorch[f'MVAU_hls_{num_prec}'] = len(self.quantizable_idx) - 1
						self.qonnx_to_pytorch[f'MVAU_rtl_{num_prec}'] = len(self.quantizable_idx) - 1
						self.qonnx_to_pytorch[f'VVAU_hls_{num_prec}'] = len(self.quantizable_idx) - 1
						self.qonnx_to_pytorch[f'VVAU_rtl_{num_prec}'] = len(self.quantizable_idx) - 1
					
					if type(module) == nn.Conv2d or type(module) == qnn.QuantConv2d:
						num_prec = 0
						for j, n in enumerate(self.model.graph.nodes):
							if n.op == 'call_module':
								m = get_module(self.model, n.target)
								if m == module:
									break
								if type(m) in [nn.Conv2d, qnn.QuantConv2d, nn.MaxPool2d]:
									num_prec+=1
								
						self.qonnx_to_pytorch[f'ConvolutionInputGenerator_hls_{num_prec}'] = len(self.quantizable_idx) - 1
						self.qonnx_to_pytorch[f'ConvolutionInputGenerator_rtl_{num_prec}'] = len(self.quantizable_idx) - 1

		layer_embedding = np.array(layer_embedding, dtype=np.float32)
		# normalize to (0, 1)
		for i in range(layer_embedding.shape[1]):
			fmin = min(layer_embedding[:, i])
			fmax = max(layer_embedding[:, i])

			if fmax - fmin > 0:
				layer_embedding[:, i] = (layer_embedding[:, i] - fmin) / (fmax - fmin)

		self.layer_embedding = layer_embedding

	def reset(self, seed = None, option = None):
		self.model = copy.deepcopy(self.orig_model).to(self.finetuner.device)
		self.build_state_embedding()

		self.finetuner.model = copy.deepcopy(self.model)
		self.finetuner.model.to(self.finetuner.device)
		self.finetuner.init_finetuning_optim()
		self.finetuner.init_loss()
		
		self.cur_ind = 0
		self.index_to_quantize = self.quantizable_idx[self.cur_ind]
		self.strategy = []

		obs = self.layer_embedding[0].copy()
		# reinitialize quantizer to be sure
		self.quantizer = Quantizer(
			self.args.weight_bit_width,
			self.args.act_bit_width,
			self.args.residual_bit_width
		)
		return obs, {}

	def step(self, action):
		action = self.get_action(action)
		self.last_action = action
		self.strategy.append(self.last_action)

		if self.is_final_layer():
			print("Strategy: " + str(self.strategy))
			self.cur_fps, self.avg_util, self.max_util, self.available_resources, self.resources_total = self.final_action_wall()
			self.model = self.quantizer.quantize_model(self.model,
											self.strategy,
											self.quantizable_idx,
											self.num_quant_acts)
			# calibrate model 
			self.finetuner.model = deepcopy(self.model) 
			self.finetuner.model.to(self.finetuner.device)
			self.finetuner.calibrate()

			# finetune model
			self.finetuner.init_finetuning_optim()
			self.finetuner.init_loss()
			self.finetuner.finetune()

			# validate model
			self.cur_acc = self.finetuner.validate()
			self.model = deepcopy(self.finetuner.model)
			
			# reward(self, acc, avg_util, max_util, available_resources, resources_total)
			reward = self.reward(self.cur_acc, self.avg_util, self.max_util, self.available_resources, self.resources_total)

			if reward > self.best_reward:
				self.best_reward = reward
			
			obs = self.layer_embedding[self.cur_ind, :].copy()
			done = True
			info = {'accuracy' : self.cur_acc, 'fps' : self.cur_fps, 'avg_util' : self.avg_util, 'strategy' : self.strategy}
			return obs, reward, done, False, info
		
		reward = 0 

		self.cur_ind += 1
		self.index_to_quantize = self.quantizable_idx[self.cur_ind]
		
		self.layer_embedding[self.cur_ind][-1] = float(self.last_action) / float(self.max_bit)
		
		done = False
		obs = self.layer_embedding[self.cur_ind, :].copy()
		info = {'accuracy' : 0.0, 'fps' : 0.0, 'avg_util' : 0.0, 'strategy' : self.strategy}
		return obs, reward, done, False, info

	def step_(self, action):
		self.last_action = action
		self.strategy.append(self.last_action)

		if self.is_final_layer():
			print("Strategy: " + str(self.strategy))
			self.cur_fps, self.avg_util, self.max_util, self.available_resources, self.resources_total = self.final_action_wall()
			self.model = self.quantizer.quantize_model(self.model,
											self.strategy,
											self.quantizable_idx,
											self.num_quant_acts)
			# calibrate model 
			self.finetuner.model = deepcopy(self.model) 
			self.finetuner.model.to(self.finetuner.device)
			self.finetuner.calibrate()

			# finetune model
			self.finetuner.init_finetuning_optim()
			self.finetuner.init_loss()
			self.finetuner.finetune()

			# validate model
			self.cur_acc = self.finetuner.validate()
			self.model = deepcopy(self.finetuner.model)
			
			reward = self.reward(self.cur_acc, self.strategy)

			if reward > self.best_reward:
				self.best_reward = reward
			
			done = True
			info = {'accuracy' : self.cur_acc, 'fps' : self.cur_fps, 'avg_util' : self.avg_util, 'strategy' : self.strategy}
			return done, info 
		
		reward = 0 

		self.cur_ind += 1
		self.index_to_quantize = self.quantizable_idx[self.cur_ind]

		done = False
		info = {'accuracy' : 0.0, 'fps' : 0.0, 'avg_util' : 0.0, 'strategy' : self.strategy}
		return done, info

	
	def reward(self, acc, avg_util, max_util, available_resources, resources_total, print_info=True):
        # reward should be within [-1, 1]

		# count_reward += 1

		# if count_reward == 1:


		# The weight of the accuracy in the reward function should be 25%
		acc_weighted = (acc * 0.02 - 1.0) # Normalize accuracy to [-1, 1]
		acc_weight = 0.25 # 25% of the reward

		bram_util = resources_total['BRAM_18K'] / available_resources['BRAM_18K']
		lut_util = resources_total['LUT'] / available_resources['LUT']
		dsp_util = resources_total['DSP'] / available_resources['DSP']

		# Check which of the components have the bigger usage
		if bram_util >= lut_util and bram_util >= dsp_util:
			# BRAM is the most utilized resource
			if lut_util >= dsp_util:
				# LUT is the second most utilized resource
				weight_BRAM = 0.35
				weight_LUT = 0.25
				weight_DSP = 0.15
			else:
				# DSP is the second most utilized resource
				weight_BRAM = 0.35
				weight_LUT = 0.15
				weight_DSP = 0.25
		elif lut_util >= bram_util and lut_util >= dsp_util:
			# LUT is the most utilized resource
			if bram_util >= dsp_util:
				# BRAM is the second most utilized resource
				weight_LUT = 0.35
				weight_BRAM = 0.25
				weight_DSP = 0.15
			else:
				# DSP is the second most utilized resource
				weight_LUT = 0.35
				weight_BRAM = 0.15
				weight_DSP = 0.25
		else:
			# DSP is the most utilized resource
			if bram_util >= lut_util:
				# BRAM is the second most utilized resource
				weight_DSP = 0.35
				weight_BRAM = 0.25
				weight_LUT = 0.15
			else:
				# LUT is the second most utilized resource
				weight_DSP = 0.35
				weight_BRAM = 0.15
				weight_LUT = 0.25
		
		# Calculate the reward components
		if resources_total['BRAM_18K'] != 0:
			resources_BRAM = (2 * (-resources_total['BRAM_18K'] + available_resources['BRAM_18K'])) / available_resources['BRAM_18K'] - 1
		else:
			resources_BRAM = 1.0
		if resources_total['LUT'] != 0:
			resources_LUT = (2 * (-resources_total['LUT'] + available_resources['LUT'])) / available_resources['LUT'] - 1
		else:
			resources_LUT = 1.0
		if resources_total['DSP'] != 0:
			resources_DSP = (2 * (-resources_total['DSP'] + available_resources['DSP'])) / available_resources['DSP'] - 1
		else:
			resources_DSP = 1.0
		
		# if weight_BRAM == 0:
		# 	weight_LUT += 0.125
		# 	weight_DSP += 0.125
		# elif weight_DSP == 0:
		# 	weight_LUT += 0.125
		# 	weight_BRAM += 0.125
		# elif weight_BRAM == 0 and weight_DSP == 0:
		# 	weight_LUT += 0.25

		reward = acc_weight * acc_weighted + weight_BRAM * resources_BRAM + weight_LUT * resources_LUT + weight_DSP * resources_DSP

		# Print out the reward components
		if(print_info):
			print("\033[91m================USED RESOURCES================\033[0m")
			print(f"\033[91mBRAM USED: {resources_total['BRAM_18K']}\033[0m")
			print(f"\033[91mLUT USED: {resources_total['LUT']}\033[0m")
			print(f"\033[91mDSP USED: {resources_total['DSP']}\033[0m")

			print("\033[92m================AVAILABLE RESOURCES================\033[0m")
			print(f"\033[92mBRAM AVAILABLE: {available_resources['BRAM_18K']}\033[0m")
			print(f"\033[92mLUT AVAILABLE: {available_resources['LUT']}\033[0m")
			print(f"\033[92mDSP AVAILABLE: {available_resources['DSP']}\033[0m")

			print("\033[94m================REWARD COMPONENTS================\033[0m")
			print(f"\033[94mBRAM: {weight_BRAM * resources_BRAM}\033[0m")
			print(f"\033[94mLUT: {weight_LUT * resources_LUT}\033[0m")
			print(f"\033[94mDSP: {weight_DSP * resources_DSP}\033[0m")
			print(f"\033[94mAcc: {acc_weight * acc_weighted}\033[0m")
			print(f"\033[94mReward: {acc_weight * acc_weighted + weight_BRAM * resources_BRAM + weight_LUT * resources_LUT + weight_DSP * resources_DSP}\033[0m")

		return reward

	def get_action(self, action):
		action = float(action[0])
		lbound, rbound = self.bound_list[self.cur_ind]
		action = (action + 1) * (rbound - lbound) / 2.0 + lbound - 0.5
		action = np.ceil(action).astype(int)
		return action

	def is_final_layer(self):
		return self.cur_ind == len(self.quantizable_idx) - 1
	
	def final_action_wall(self):
		while True:
			model_for_measure = copy.deepcopy(self.model)
			model_for_measure = self.quantizer.quantize_model(model_for_measure,
															self.strategy,
															self.quantizable_idx,
															self.num_quant_acts)
			model_for_measure.eval()
			
			# Export model to qonnx first
			img_shape = self.model_config['center_crop_shape']
			device, dtype = next(model_for_measure.parameters()).device, next(model_for_measure.parameters()).dtype
			ref_input = torch.randn(1, self.finetuner.in_channels, img_shape, img_shape, device = device, dtype = dtype)
			bo.export_qonnx(model_for_measure, ref_input, export_path = 'model.onnx', keep_initializers_as_inputs = True, opset_version = 11)

			# Choose streamlining and hw function depending on model
			streamline_function = streamline_functions[self.args.model_name]
			convert_to_hw_function = convert_to_hw_functions[self.args.model_name]

			model = ModelWrapper('model.onnx')
			model = preprocessing(model)
			model = postprocessing(model)
			model = make_input_channels_last(model)
			model = tidy_up(model)
			model = qonnx_to_finn(model)
			model = streamline_function(model)
			model = name_nodes(model)
			model = convert_to_hw_function(model)
			model = create_dataflow_partition(model)
			model = specialize_layers(model, self.args.fpga_part)
			model, cycles, avg_util, bottleneck_layer, max_util, available_resources, resources_total = set_folding(model, self.args.output_dir, self.args.board_file, self.args.freq, self.args.target_fps, self.args.slr)

			fps = self.args.freq * 10**6 / cycles
			print(f'Achieved fps: {fps}')
			if fps >= self.args.target_fps:
				print(f'Achieved desired fps')
				break
			else:
				freq = cycles * self.args.target_fps / 10**6

				if freq <= self.args.max_freq:
					print(f'Managed to achieve {self.args.target_fps} fps using freq = {freq} MHz. Consider changing target frequency')
				
				print(f'Target fps not achieved (achieved fps: {fps})')
				
				reduced = False
				idx = np.lexsort((np.arange(len(self.strategy)), self.strategy))[::-1][0]
				bit = self.strategy[idx]
				if bit > self.min_bit and bit > self.bound_list[idx][0]:
					self.strategy[idx] -= 1
					reduced = True
					print("Strategy: " + str(self.strategy))
				
				if not reduced:
					# not another opportunity to minimize bit width
					break

		return fps, avg_util, max_util, available_resources, resources_total
	
	def maximum_fps(self):
		strategy = [self.bound_list[i][0] for i in range(len(self.quantizable_idx))]
		model_for_measure = copy.deepcopy(self.model)
		model_for_measure = self.quantizer.quantize_model(model_for_measure,
														strategy,
														self.quantizable_idx,
														self.num_quant_acts)
		model_for_measure.eval()

		# export model to qonnx
		img_shape = self.model_config['center_crop_shape']
		device, dtype = next(model_for_measure.parameters()).device, next(model_for_measure.parameters()).dtype
		ref_input = torch.randn(1, self.finetuner.in_channels, img_shape, img_shape, device = device, dtype = dtype)
		model_for_measure.eval()
		bo.export_qonnx(model_for_measure, ref_input, export_path = 'model.onnx', keep_initializers_as_inputs = True, verbose = False, opset_version = 11)
	
		# Choose streamlining and hw function
		streamline_function = streamline_functions[self.args.model_name]
		convert_to_hw_function = convert_to_hw_functions[self.args.model_name]
		
		model = ModelWrapper('model.onnx')
		model = preprocessing(model)
		model = postprocessing(model)
		model = make_input_channels_last(model)
		model = tidy_up(model)
		model = qonnx_to_finn(model)
		model = streamline_function(model)
		model = name_nodes(model)
		model = convert_to_hw_function(model)
		model = create_dataflow_partition(model)
		model = specialize_layers(model, self.args.fpga_part)
		model, cycles, _, _, _, _, _ = set_folding(model, self.args.output_dir, self.args.board_file, self.args.freq, self.args.target_fps, self.args.slr)
		
		if cycles < 0:
			print('Initial model infeasible')
			exit()

		fps = self.args.freq * 10**6 / cycles

		print(f'Maximum achievable fps at {self.args.freq} MHz: {fps} fps')
		
		if fps < self.args.target_fps:
			print(f'Target fps not achievable, changing target fps from {self.args.target_fps} to {fps}')
			self.args.target_fps = fps