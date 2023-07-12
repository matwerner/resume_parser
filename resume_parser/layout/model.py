import os
import sys
import torch
import torch.nn as nn

from argparse import Namespace
from typing import Type, Tuple, Dict, Any
from torchvision import models


def set_parameter_requires_grad(model: nn.Module, is_required: bool):
    for param in model.parameters():
        param.requires_grad = is_required


def get_model_image_size(model_name: str) -> int:
    if model_name.startswith("resnet"):
        return 224
    elif model_name.startswith("efficientnet"):
        if model_name == "efficientnet_b0":
            return 224
        elif model_name == "efficientnet_b1":
            return 240
        elif model_name == "efficientnet_b2":
            return 260
        elif model_name == "efficientnet_b3":
            return 300
        else:
            print("Not implemented yet, exiting...")
            sys.exit()
    else:
        print("Not implemented yet, exiting...")
        sys.exit()


def initialize_model(model_name: str,
                     num_classes: int,
                     pretrained: str,
                    ) -> Type[nn.Module]:
    # Initialize these variables which will be set in this if statement.
    # Each of these variables is model specific.
    if model_name not in ["resnet18", "resnet34", "resnet50",
                          "efficientnet_b0", "efficientnet_b1",
                          "efficientnet_b2", "efficientnet_b3"]:
        print("Invalid model name, exiting...")
        sys.exit()
    if pretrained and pretrained not in ["default", "imagenet"]:
        print("Invalid weights name, exiting...")
        sys.exit()

    remove_head = num_classes < 0

    model = None
    if model_name.startswith("resnet"):
        if model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
        elif model_name == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            model = models.resnet34(weights=weights)
        elif model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
        else:
            print("Not implemented yet, exiting...")
            sys.exit()

        set_parameter_requires_grad(model, False)
        if remove_head:
            model.fc = nn.Identity()
        else:
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)

    else:
        if model_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
        elif model_name == "efficientnet_b1":
            weights = models.EfficientNet_B1_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b1(weights=weights)
        elif model_name == "efficientnet_b2":
            weights = models.EfficientNet_B2_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b2(weights=weights)
        elif model_name == "efficientnet_b3":
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b3(weights=weights)
        else:
            print("Not implemented yet, exiting...")
            sys.exit()

        set_parameter_requires_grad(model, False)
        if remove_head:
            model.classifier[1] = nn.Identity()
        else:
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_ftrs, num_classes)

    return model


def save_model(model: Type[torch.nn.Module], args: Namespace) -> None:
    """Save a trained model and the associated configuration to output dir."""
    torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'model.pth'))


def load_model(args: Namespace) -> Tuple[Type[nn.Module], Namespace]:
    training_args = torch.load(os.path.join(args.output_dir, 'training_args.bin'))
    checkpoint = torch.load(os.path.join(args.output_dir, 'model.pth'))
    model = initialize_model(training_args.model, training_args.num_labels, 
                             training_args.pretrained)
    model.load_state_dict(checkpoint)
    set_parameter_requires_grad(model, False)
    return model, training_args


def train_last_blocks(model: nn.Module,
                      model_name: str,
                      num_blocks_unfreeze=-1,
                      freeze_all_bn_layers=True
                     ) -> None:
    if not model_name.startswith("efficientnet"):
        sys.exit()
    # Unfreeze all layers
    set_parameter_requires_grad(model, True)

    features = model.get_submodule("features")
    num_blocks = len(features)
    # Train all
    if num_blocks_unfreeze < 0:
        num_blocks_unfreeze = num_blocks
    for i, (_, child) in enumerate(features.named_children()):
        # 0 and 8: Conv2dNormActivation
        # 1 to 7: Sequential
        is_requires_grad = i >= num_blocks - num_blocks_unfreeze
        for param in child.parameters():
            param.requires_grad = is_requires_grad

    # Freeze Batch Norm layers
    if freeze_all_bn_layers:
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()
