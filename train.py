from copy import deepcopy
import os
import torch
import random
import argparse
import numpy as np

from train.env import ModelEnv
from train.env.ModelEnv import platform_files
from train.env.utils import get_model_config

from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback, StopTrainingOnNoModelImprovement, EvalCallback, BaseCallback

from finn.util.basic import part_map

rl_algorithms = {
    'A2C': A2C,
    'DDPG': DDPG,
    'PPO': PPO,
    'SAC': SAC,
    'TD3': TD3
}

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Ensure environment exists
        if hasattr(self.training_env, 'envs'):
            env = self.training_env.envs[0]  # Assuming a single environment

            # Log cur_acc and cur_fps
            if hasattr(env, 'cur_acc'):
                self.logger.record("env/cur_acc", env.unwrapped.cur_acc)
            if hasattr(env, 'cur_fps'):
                self.logger.record("env/cur_fps", env.unwrapped.cur_fps)
            if hasattr(env, 'avg_util'):
                self.logger.record("env/avg_util", env.unwrapped.avg_util)

        return True

# Parse arguments
parser = argparse.ArgumentParser(description = 'Train RL Agent')

# Model Parameters
parser.add_argument('--model-name', default='resnet18', help = 'Target model name')
parser.add_argument('--model-path', required = True, default = None, help = 'Path to pretrained model')

# Dataset Parameters
parser.add_argument('--datadir', default = './data', help='Directory where datasets are stored (default: ./data)')
parser.add_argument('--dataset', default = 'CIFAR10', choices = ['MNIST', 'CIFAR10'], help = 'Name of dataset (default: CIFAR10)')
parser.add_argument('--batch-size-finetuning', default = 64, type = int, help = 'Batch size for finetuning (default: 64)')
parser.add_argument('--batch-size-testing', default = 64, type = int, help = 'Batch size for testing (default: 64)')
parser.add_argument('--num-workers', default = 8, type = int, help = 'Num workers (default: 8)')
parser.add_argument('--calib-subset', default = 0.1, type = float, help = 'Percentage of training dataset for calibration (default: 0.1)')
parser.add_argument('--finetuning-subset', default = 0.5, type = float, help = 'Percentage of dataset to use for finetuning (default: 0.5)')

# Trainer Parameters
parser.add_argument('--finetuning-epochs', default = 2, type = int, help = 'Finetuning epochs (default: 2)')
parser.add_argument('--print-every', default = 100, type = int, help = 'How frequent to print progress (default: 100)')

# Optimizer Parameters
parser.add_argument('--optimizer', default = 'Adam', choices = ['Adam', 'SGD'], help = 'Optimizer (default: Adam)')
parser.add_argument('--finetuning-lr', default = 1e-5, type = float, help = 'Training finetuning learning rate (default: 1e-5)')
parser.add_argument('--weight-decay', default = 0, type = float, help = 'Weight decay for optimizer (default: 0)')

# Loss Parameters
parser.add_argument('--loss', default = 'CrossEntropy', choices = ['CrossEntropy'], help = 'Loss Function for training (default: CrossEntropy)')

# Device Parameters
parser.add_argument('--device', default = 'GPU', help = 'Device for training (default: GPU)')

# Quantization Parameters
parser.add_argument('--residual-bit-width', default = 4, type = int, help = 'Bit width for residual connections (default: 4)')
parser.add_argument('--act-bit-width', default=4, type=int, help = 'Bit width for activations (default: 4)')
parser.add_argument('--weight-bit-width', default=4, type=int, help = 'Bit width for weights (default: 4)')
parser.add_argument('--min-bit', type=int, default=1, help = 'Minimum bit width (default: 1)')
parser.add_argument('--max-bit', type=int, default=8, help = 'Maximum bit width (default: 8)')

# Agent Parameters
parser.add_argument('--agent', default = 'TD3', choices = ['A2C', 'DDPG', 'PPO', 'SAC', 'TD3'], help = 'Choose algorithm to train agent (default: TD3)')
parser.add_argument('--noise', default = 0.1, type = float, help = 'Std for added noise in agent (default: 0.1)')
parser.add_argument('--num-episodes', default = 500, type = int, help = 'Number of episodes to train the agent for (default: 500)')
parser.add_argument('--log-every', default = 10, type = int, help = 'How many episodes to wait to log agent (default: 10)')
parser.add_argument('--save-every', default = 10, type = int, help = 'How many episodes to wait to save agent checkpoint (default: 10)')
parser.add_argument('--seed', default = 234, type = int, help = 'Seed to reproduce (default: 234)')

# Design Parameters
parser.add_argument('--board', default = "U250", choices = ['U250', 'ZCU102'], help = "Name of target board (default: U250)")
parser.add_argument('--board-file', default = 'platforms/u250.json', help = "Name of file with resources (default: platforms/u250.json)")
parser.add_argument('--slr', type = int, default = -1, help = 'SLR to map the accelerator (default: -1)')
parser.add_argument('--shell-flow-type', default = "vitis_alveo", choices = ["vivado_zynq", "vitis_alveo"], help = "Target shell type (default: vitis_alveo)")
parser.add_argument('--freq', type = float, default = 300.0, help = 'Frequency in MHz (default: 300)')
parser.add_argument('--max-freq', type = float, default = 300.0, help = 'Maximum device frequency in MHz (default: 300)')
parser.add_argument('--target-fps', default = 6000, type = float, help = 'Target fps (default: 6000)')

# Logger parameters
parser.add_argument('--tensorboard-log', default='./tensorboard_logs', help='Directory for TensorBoard logs')

def main():
    args = parser.parse_args()

    # set seed to reproduce
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == 'GPU' and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    args.fpga_part = part_map[args.board]
    args.output_dir = args.model_name
    args.board_file = platform_files[args.board]

    eval_env = ModelEnv(args, get_model_config(args.dataset), testing = True)

    env = Monitor(
        ModelEnv(args, get_model_config(args.dataset), testing = False),
        filename = 'monitor.csv',
        info_keywords=('accuracy', 'fps', 'avg_util', 'strategy'),
    )

    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=args.noise * np.ones(n_actions))

    agent = rl_algorithms[args.agent](
        "MlpPolicy", 
        env, 
        action_noise = action_noise, 
        verbose = 1, 
        seed = args.seed,
        tensorboard_log = args.tensorboard_log
        )
    
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals = 3, min_evals = 5, verbose = 1)
    eval_callback = EvalCallback(eval_env, eval_freq = len(env.quantizable_idx) * 30, callback_after_eval = stop_train_callback, verbose = 1, n_eval_episodes = 1)
    checkpoint_callback = CheckpointCallback(save_freq = args.save_every * len(env.quantizable_idx), save_path = 'agents', name_prefix = f'agent_{args.model_name}') 
    tensorboard_callback = TensorboardCallback()
    agent.learn(total_timesteps = len(env.quantizable_idx) * args.num_episodes, 
                callback = [eval_callback, checkpoint_callback, tensorboard_callback],
                log_interval = 1,
                tb_log_name = f"RL_{args.agent}_{args.model_name}"
                )
    agent.save(f'agents/agent_{args.model_name }')
    
if __name__ == "__main__":
    main()
