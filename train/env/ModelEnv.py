import copy
import math
import platform
import json
from venv import create
import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from enum import IntEnum

from brevitas.graph.utils import get_module
from brevitas.graph.quantize import preprocess_for_quantize
import brevitas.nn as qnn

from ..quantizer import Quantizer
from ..finetune import Finetuner
from ..exporter.Exporter import (
    set_fifo_depths,
    tidy_up,
    preprocessing,
    postprocessing,
    make_input_channels_last,
    qonnx_to_finn,
    create_dataflow_partition,
    specialize_layers,
    set_folding,
    apply_folding_config,
    minimize_bit_width,
    resource_estimates,
    streamline_resnet,
    convert_to_hw_resnet,
    name_nodes,
    streamline_lenet,
    convert_to_hw_lenet
)
from ..utils import measure_model

from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env

import brevitas.onnx as bo
from qonnx.core.modelwrapper import ModelWrapper
from copy import deepcopy

platform_path = './platforms'
platform_files = {}
platform_files['U250'] = f'{platform_path}/u250.json'

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

class ModelEnv:
    def __init__(self, args, model_config):
        self.args = args

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
        
        self.build_state_embedding() # build the states for each layer
        self.index_to_quantize = self.quantizable_idx[self.cur_ind]

        self.quantizer = Quantizer(
            self.model,
            args.weight_bit_width,
            args.act_bit_width,
            args.bias_bit_width
        )
    
        self.orig_acc = self.finetuner.orig_acc
        self.max_fps = 0.0
        self.max_acc = 0.0

        print('=> Original Accuracy: {:.3f}%'.format(self.orig_acc * 100))

    def build_state_embedding(self):
        self.model = preprocess_for_quantize(self.model)

        measure_model(self.model, self.model_config['center_crop_shape'], 
                    self.model_config['center_crop_shape'], self.finetuner.in_channels)
    
        self.quantizable_idx = []
        self.bound_list = []
        self.num_quant_acts = 0
        layer_embedding = []

        # Activations first
        for i, node in enumerate(self.model.graph.nodes):
            this_state = []
            if node.op == 'call_module':
                module = get_module(self.model, node.target)
                if type(module) in self.quantizable_acts:
                    self.quantizable_idx.append(i)

                    # for activations, increase by 1 the minimum bit
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

        self.model.to(self.finetuner.device)

        self.finetuner.model = self.model
        self.finetuner.model.to(self.finetuner.device)
        self.finetuner.init_finetuning_optim()
        self.finetuner.init_loss()
        
        self.cur_ind = 0
        self.index_to_quantize = self.quantizable_idx[self.cur_ind]
        self.strategy = []

        obs = self.layer_embedding[0].copy()
        return obs

    def step(self, action):
        action = self.get_action(action)
        self.last_action = action
        self.strategy.append(self.last_action)

        if self.is_final_layer():
            print(self.strategy)
            
            if self.args.target == 'latency':
                fps, avg_util, max_util, penalty = self.final_action_wall()
                self.model = self.quantizer.quantize_model(self.model,
                                                self.strategy,
                                                self.quantizable_idx,
                                                self.num_quant_acts)
                # calibrate model 
                self.finetuner.model = self.model 
                self.finetuner.model.to(self.finetuner.device)
                self.finetuner.calibrate()

                # finetuner model
                self.finetuner.init_finetuning_optim()
                self.finetuner.init_loss()
                self.finetuner.finetune()

                # validate model
                acc = self.finetuner.validate()
                
            elif self.args.target == 'accuracy':
                penalty = 0.0
                
                fps, avg_util, max_util, penalty = self.final_action_wall()
                self.prev_model = deepcopy(self.model)
                self.model = self.quantizer.quantize_model(self.model,
                                                self.strategy,
                                                self.quantizable_idx,
                                                self.num_quant_acts)
                
                self.finetuner.model = self.model 
                self.finetuner.model.to(self.finetuner.device)
                self.finetuner.calibrate()

                # finetuner model
                self.finetuner.init_finetuning_optim()
                self.finetuner.init_loss()
                self.finetuner.finetune()

                # validate model
                acc = self.finetuner.validate()

                if acc > self.max_acc:
                    self.max_acc = acc

                if acc < self.args.target_acc:
                    penalty = (acc - self.args.target_acc) * 100000.0

                '''
                idx, bit = min(enumerate(self.strategy), key = lambda x : x[1])
                if bit < self.max_bit:
                    self.strategy[idx] += 1
                    print(self.strategy)
                    penalty -= acc
                    self.model = deepcopy(self.prev_model)
                else:
                    # all bits are quantized
                    self.args.target_acc = self.max_acc
                    print(f'Failed to achieve target accuracy, new target accuracy is set to {self.max_acc} %')
                '''

            reward = self.reward(acc, fps, penalty, target = self.args.target)

            if reward > self.best_reward:
                self.best_reward = reward
            
            obs = self.layer_embedding[self.cur_ind, :].copy()
            done = True
            info = {'accuracy' : acc, 'fps' : fps, 'avg_util' : avg_util}
            return obs, reward, done, info 
        
        reward = 0 

        self.cur_ind += 1
        self.index_to_quantize = self.quantizable_idx[self.cur_ind]
        
        self.layer_embedding[self.cur_ind][-1] = float(self.last_action) / float(self.max_bit)
        
        done = False
        obs = self.layer_embedding[self.cur_ind, :].copy()
        info = {'accuracy' : 0.0, 'fps' : 0.0, 'avg_util' : 0.0}
        return obs, reward, done, info
    
    def reward(self, acc, fps, penalty, target = 'latency'):
        print(f'penalty = {penalty}')
        if target == 'latency':
            return (acc - self.orig_acc) * 0.1 + penalty
        else:
            return fps + penalty
        
    def get_action(self, action):
        action = float(action)
        min_bit, max_bit = self.bound_list[self.cur_ind]
        lbound, rbound = min_bit - 0.5, max_bit + 0.5
        action = (rbound - lbound) * action + lbound
        action = int(np.round(action, 0))
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
            # export model to qonnx
            img_shape = self.model_config['center_crop_shape']
            device, dtype = next(model_for_measure.parameters()).device, next(model_for_measure.parameters()).dtype
            ref_input = torch.randn(1, self.finetuner.in_channels, img_shape, img_shape, device = device, dtype = dtype)
            bo.export_qonnx(model_for_measure, ref_input, export_path = 'model.onnx', keep_initializers_as_inputs = False, opset_version = 11, verbose = False)
        
            # Transformations
            model = ModelWrapper('model.onnx')
            model = preprocessing(model)
            model = postprocessing(model)
            model = make_input_channels_last(model)
            model = tidy_up(model)
            model = qonnx_to_finn(model)
            model = streamline_lenet(model)
            model = convert_to_hw_lenet(model)
            model = create_dataflow_partition(model)
            model = specialize_layers(model, self.args.fpga_part)
            model, cycles, avg_util, max_util = set_folding(model, self.args.board)

            penalty = 0.0

            fps = self.args.freq * 10**6 / cycles
            if fps > self.max_fps:
                self.max_fps = fps

            if self.args.target == 'latency':
                if fps >= self.args.target_fps:
                    break
                else:
                    freq = cycles * self.args.target_fps / 10**6

                    if freq <= self.args.max_freq:
                        print(f'Managed to achieve {self.args.target_fps} fps using freq = {freq} MHz. Consider changing target frequency')
                    
                    print(f'Target fps not achieved (achieved fps: {fps})')
                    penalty -= fps
                    # reduce bitwidth in the hopes that maximum fps is achieved
                    idx, bit = max(enumerate(self.strategy), key = lambda x : x[1])
                    print(self.strategy)
                    print(self.bound_list)
                    if bit > self.min_bit and bit > self.bound_list[idx][0]:
                        self.strategy[idx] -= 1
                    else:
                        # all bits are quantized
                        self.args.target_fps = self.max_fps
                        print(f'Failed to achieve target fps, new target fps is set to {self.max_fps} fps')
                        break
            else:
                break
        return fps, avg_util, max_util, penalty
    '''
    def final_action_wall(self):
        model_for_measure = copy.deepcopy(self.model)
        model_for_measure =  self.quantizer.quantize_model(model_for_measure,
                                                        self.strategy,
                                                        self.quantizable_idx,
                                                        self.num_quant_acts)
        # export model to qonnx
        img_shape = self.model_config['center_crop_shape']
        device, dtype = next(model_for_measure.parameters()).device, next(model_for_measure.parameters()).dtype
        ref_input = torch.randn(1, self.finetuner.in_channels, img_shape, img_shape, device = device, dtype = dtype)
        bo.export_qonnx(model_for_measure, ref_input, export_path = 'model.onnx', keep_initializers_as_inputs = False, opset_version = 11, verbose = False)
    
        # Transformations
        model = ModelWrapper('model.onnx')
        model = preprocessing(model)
        model = postprocessing(model)
        model = make_input_channels_last(model)
        model = tidy_up(model)
        model = qonnx_to_finn(model)
        model.save('qonnx_finn.onnx')
        model = streamline_lenet(model)
        model.save('streamlined.onnx')
        model = convert_to_hw_lenet(model)
        model = create_dataflow_partition(model)
        model = specialize_layers(model, self.args.fpga_part)
        model, cycles, avg_util, max_util = set_folding(model, self.args.board)

        penalty = 0.0

        fps = self.args.freq * 10**6 / cycles
        if fps > self.max_fps:
            self.max_fps = fps

        if self.args.target == 'latency':
            if fps < self.args.target_fps:
                freq = cycles * self.args.target_fps / 10**6

                if freq <= self.args.max_freq:
                    print(f'Managed to achieve {self.args.target_fps} fps using freq = {freq} MHz. Consider changing target frequency')
                
                print(f'Target fps not achieved (achieved fps: {fps})')
                penalty = (fps - self.args.target_fps) * 1000.0
                # reduce bitwidth in the hopes that maximum fps is achieved
                idx, bit = max(enumerate(self.strategy), key = lambda x : x[1])
                if bit > self.min_bit and bit > self.bound_list[idx]:
                    self.strategy[idx] -= 1
                else:
                    # all bits are quantized
                    self.args.target_fps = self.max_fps
                    print(f'Failed to achieve target fps, new target fps is set to {self.max_fps} fps')
                    break
        return fps, avg_util, max_util, penalty
        '''