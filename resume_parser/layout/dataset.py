import json
import os
import numpy as np
import pytorch_lightning as pl
import torch

from argparse import Namespace
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm.auto import tqdm
from typing import Optional, Callable

# Reproducibility
pl.seed_everything(0, workers=True)


class ResumeLayoutDataModule(pl.LightningDataModule):
    """DataModule used for semantic segmentation in geometric generalization
    project.
    """

    train: ImageFolder
    valid: ImageFolder
    test: ImageFolder
    train_sampler: WeightedRandomSampler

    def __init__(self, config_filepath: str,
                       train_batch_size: int,
                       test_batch_size: int,
                       train_transform = None,
                       test_transform = None,
                       use_uniform_sampler: bool =False,
                       is_anonymized: bool =False,
                ) -> None:
        super().__init__()
        self.config = json.load(open(config_filepath, encoding='utf-8'))
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.use_uniform_sampler = use_uniform_sampler
        self.is_anonymized = is_anonymized


    def setup(self, stage: Optional[str] = None):
        root_dirpath = self.config['root_dirpath']

        train_filenames = self.config['train']
        valid_filenames = self.config['valid']
        test_filenames = self.config['test']

        def is_valid_file_func(setname: str) -> Callable[[str], bool]:
            def is_valid_file(filepath: str) -> bool:
                image_filename = os.path.splitext(filepath)[0]
                image_basename = os.path.basename(image_filename)
                filename = image_basename.split('_')[0]
                if setname == 'test':
                    return filename in test_filenames
                if setname == 'valid':
                    return filename in valid_filenames
                return filename in train_filenames
            return is_valid_file

        if self.is_anonymized:
            dataset_dirpath = os.path.join(root_dirpath, 'anonymized')
        else:
            dataset_dirpath = os.path.join(root_dirpath, 'default')

        self.train = ImageFolder(dataset_dirpath, transform=self.train_transform,
                                 is_valid_file=is_valid_file_func('train'))

        self.train_sampler = None
        if self.use_uniform_sampler:
            train_targets = self.train.targets
            label_counts = np.bincount(train_targets)
            weights = 1. / np.array(label_counts, dtype=np.float)
            samples_weights = torch.tensor(weights[train_targets])

            self.train_sampler = WeightedRandomSampler(
                weights=samples_weights,
                num_samples=len(samples_weights),
                replacement=True
            )

        if stage == 'fit' or stage is None:
            self.valid = ImageFolder(dataset_dirpath, transform=self.test_transform,
                                     is_valid_file=is_valid_file_func('valid'))
        elif stage == 'test':
            self.test = ImageFolder(dataset_dirpath, transform=self.test_transform,
                                    is_valid_file=is_valid_file_func('test'))
        else:
            raise ValueError(f'Invalid stage: {stage}')


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.train_batch_size,
            num_workers=4, sampler=self.train_sampler)


    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid, batch_size=self.test_batch_size,
            num_workers=4)


    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=self.test_batch_size,
            num_workers=4)



class FeatureExtractorDataset(Dataset):


    def __init__(self,
                 dataset: Dataset,
                 model: torch.nn.Module,
                 device: torch.device = torch.device('cpu'),
                 batch_size: int = 256,
                ) -> None:
        self.dataset = dataset
        self.model = model
        self.device = device
        self.batch_size = batch_size

        self.feature_vectors = None
        self.targets = None
        self._extract_features_v2()


    def _extract_features_v2(self) -> None:
        self.model.eval()

        x_batches, y_batches = [], []
        with torch.no_grad():
            data_loader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=4)
            for image, targets in tqdm(data_loader):
                image = image.to(self.device)

                # forward pass
                outputs = self.model(image).cpu().detach()
                x_batches.append(outputs)
                y_batches.append(targets)
        self.feature_vectors = torch.cat(x_batches, 0)
        self.targets = torch.cat(y_batches, 0)

    def _extract_features(self) -> None:
        feature_vectors = []
        targets = []

        self.model.eval()

        with torch.no_grad():
            for input_data, target in tqdm(self.dataset):
                input_data = input_data.unsqueeze(0)
                input_data = input_data.to(self.device)
                features = self.model(input_data)

                feature_vectors.append(features.squeeze(0).cpu().detach())
                targets.append(target)

        self.feature_vectors = torch.stack(feature_vectors)
        self.targets = torch.tensor(targets)


    def __len__(self) -> int:
        return len(self.dataset)


    def __getitem__(self, index: int):
        return self.feature_vectors[index], self.targets[index]


class FeatureExtractorDataModule(pl.LightningDataModule):
    """DataModule used for semantic segmentation in geometric generalization
    project.
    """

    train: FeatureExtractorDataset
    valid: FeatureExtractorDataset
    test: FeatureExtractorDataset

    def __init__(self,
                 data_module: pl.LightningDataModule,
                 model: torch.nn.Module,
                 device: torch.device,
                 train_batch_size: int,
                 test_batch_size: int,
                 use_uniform_sampler: bool =False,
                ) -> None:
        super().__init__()
        self.data_module = data_module
        self.model = model
        self.device = device
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.use_uniform_sampler = use_uniform_sampler
        self.classes = self.data_module.train.classes


    def setup(self, stage: Optional[str] = None):
        self.train = FeatureExtractorDataset(self.data_module.train, self.model, self.device)
        if stage == 'fit' or stage is None:
            self.valid = FeatureExtractorDataset(self.data_module.valid, self.model, self.device)
        elif stage == 'test':
            self.test = FeatureExtractorDataset(self.data_module.test, self.model, self.device)
        else:
            raise ValueError(f'Invalid stage: {stage}')

        self.train_sampler = None
        if self.use_uniform_sampler:
            train_targets = self.train.targets
            label_counts = np.bincount(train_targets)
            weights = 1. / np.array(label_counts, dtype=np.float)
            samples_weights = torch.tensor(weights[train_targets])

            self.train_sampler = WeightedRandomSampler(
                weights=samples_weights,
                num_samples=len(samples_weights),
                replacement=True
            )


    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.train_batch_size, num_workers=4,
                          sampler=self.train_sampler)


    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.valid, batch_size=self.test_batch_size, num_workers=4)


    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test, batch_size=self.test_batch_size, num_workers=4)


def compute_image_mean_std_from_train(args: Namespace):
    output_shape = (args.image_size, args.image_size)

    stats_transform = transforms.Compose([
        transforms.Resize(output_shape),
        transforms.ToTensor()
    ])

    stats_dataset = ResumeLayoutDataModule(
        config_filepath=args.dataset_file,
        train_batch_size=args.test_batch_size,
        test_batch_size=args.test_batch_size,
        train_transform=stats_transform,
        test_transform=stats_transform,
        use_uniform_sampler=False,
        is_anonymized=args.discretized_image
    )
    stats_dataset.setup('fit')

    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in stats_dataset.train_dataloader():
        b, c, h, w = images.shape
        num_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + num_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + num_pixels)
        cnt += num_pixels

    mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)
    return mean.tolist(), std.tolist()


def load_data_module(args: Namespace, stage: Optional[str] = None) -> ResumeLayoutDataModule:
    output_shape = (args.image_size, args.image_size)

    train_transform = transforms.Compose([
        transforms.Resize(output_shape),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            fill=0
        ),
        transforms.RandomInvert(0.2),
        transforms.RandomHorizontalFlip(0.2),
        transforms.RandomVerticalFlip(0.2),
        transforms.ToTensor(),
        transforms.Normalize(args.image_mean, args.image_std),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(output_shape),
        transforms.ToTensor(),
        transforms.Normalize(args.image_mean, args.image_std)
    ])

    dataset = ResumeLayoutDataModule(
        config_filepath=args.dataset_file,
        train_batch_size=args.train_batch_size,
        test_batch_size=args.test_batch_size,
        train_transform=train_transform,
        test_transform=test_transform,
        use_uniform_sampler=True,
        is_anonymized=args.discretized_image
    )
    dataset.setup(stage)
    return dataset
