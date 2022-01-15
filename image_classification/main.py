import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import os
from argparse import ArgumentParser

# Manage command line arguments
parser = ArgumentParser()

parser.add_argument("--device", default="cuda", type=str,
                    help="Device to be utilized (cuda recommended).")
parser.add_argument("--cuda_devices", default="0, 1", type=str,
                    help="String of cuda device indexes to be used. Indexes must be separated by a comma.")
parser.add_argument("--data_parallel", default=False, action="store_true",
                    help="Binary flag. If set data parallel is utilized.")
parser.add_argument("--epochs", default=500, type=int,
                    help="Number of epochs to perform while training.")
parser.add_argument("--lr", default=1e-03, type=float,
                    help="Learning rate to be employed.")
parser.add_argument("--weight_decay", default=1e-02, type=float,
                    help="Weight decay to be employed.")
parser.add_argument("--batch_size", default=256, type=int,
                    help="Number of epochs to perform while training.")
parser.add_argument("--load_network", default=None, type=str,
                    help="If set given network (state dict) is loaded.")
parser.add_argument("--dataset", default="cifar10", type=str, choices=["cifar10, places365"],
                    help="Dataset to be used (CIFAR10 or Places365). "
                         "Places365 dataset (easy directory structure) must be downloaded in advance.")
parser.add_argument("--dataset_path", default="", type=str,
                    help="Dataset path, only needed for Places365 dataset.")

# Get arguments
args = parser.parse_args()

# Set cuda visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

from swin_transformer_v2 import swin_transformer_v2_t
from metrics import Accuracy
from model_wrapper import ModelWrapper
from logger import Logger
from utils import ClassificationModelWrapper


def main(args) -> None:
    if args.dataset == "cifar10":
        print("CIFAR10 dataset utilized")
        # Init transformations
        transform_train = transforms.Compose([
            transforms.RandomCrop(size=32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomRotation(degrees=2),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
        ])
        # Init datasets
        training_dataset = torchvision.datasets.CIFAR10(root="./CIFAR10", train=True, download=True,
                                                        transform=transform_train)
        training_dataset = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=min(20, args.batch_size), pin_memory=True)
        test_dataset = torchvision.datasets.CIFAR10(root="./CIFAR10", train=False, download=True,
                                                    transform=transform_test)
        test_dataset = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=min(20, args.batch_size), pin_memory=True)
    else:
        print("Places365 dataset utilized")
        # Init transformations
        transform_train = transforms.Compose([
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(size=256),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        # Init datasets
        training_dataset = torchvision.datasets.ImageFolder(root=os.path.join(args.dataset_path, "train"),
                                                            transform=transform_train)
        training_dataset = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True,
                                      num_workers=min(20, args.batch_size), pin_memory=True)
        test_dataset = torchvision.datasets.CIFAR10(root=os.path.join(args.dataset_path, "val"),
                                                    transform=transform_test)
        test_dataset = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                  num_workers=min(20, args.batch_size), pin_memory=True)
    # Init model
    model = ClassificationModelWrapper(
        model=swin_transformer_v2_t(input_resolution=(32, 32) if args.dataset == "cifar10" else (256, 256),
                                    window_size=8, dropout=0.1, dropout_path=0.2),
        number_of_classes=10 if args.dataset == "cifar10" else 365)
    # Print number of parameters
    print("# parameters", sum([p.numel() for p in model.parameters()]))
    # Model to device
    model.to(args.device)
    # Init data parallel
    if args.data_parallel:
        model = nn.DataParallel(model)
    # Init optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Init learning rate schedule
    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[50, 100, 150], gamma=0.1,
                                                       verbose=True)
    # Init loss function
    loss_function = nn.CrossEntropyLoss()
    # Init model wrapper
    model_wrapper = ModelWrapper(model=model,
                                 optimizer=optimizer,
                                 loss_function=loss_function,
                                 training_dataset=training_dataset,
                                 test_dataset=test_dataset,
                                 lr_schedule=lr_schedule,
                                 validation_metric=Accuracy(),
                                 logger=Logger(experiment_path_extension=str(model.__class__.__name__)),
                                 device=args.device)
    # Perform training
    model_wrapper.train(epochs=args.epochs)


if __name__ == '__main__':
    main(args=args)
