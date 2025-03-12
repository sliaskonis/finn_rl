import argparse
from torchvision import transforms

def get_transforms(dataset_config, apply_transformations):
    transform_list = []

    if apply_transformations:
        for t in dataset_config.get("transformations", []):
            transform_name = t["name"]
            params = t.get("params", {})

            if hasattr(transforms, transform_name):
                transform = getattr(transforms, transform_name)(**params)
            transform_list.append(transform)

    transform_list.append(transforms.ToTensor())  # Always include ToTensor
    return transforms.Compose(transform_list)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')