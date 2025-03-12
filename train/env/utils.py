from ..env import ModelEnv

def get_model_config(dataset):
    config = dict()

    if dataset == "MNIST":
        input_shape = 28
        resize_shape = 28
    elif dataset == "CIFAR10":
        input_shape = 32
        resize_shape = 32

    config.update({'resize_shape': resize_shape, 'center_crop_shape': input_shape})
    return config

def make_env(args):
    env = ModelEnv(args, get_model_config(args.model_name, args.dataset))
    return env

def make_vec_envs(args, num_agents):
    envs = [make_env(args) for i in range(num_agents)]
    
