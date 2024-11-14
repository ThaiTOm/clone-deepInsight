import queue as Queue
import threading
from typing import Iterable

import numpy as np
import torch
from functools import partial
from torch import distributed
from torch.utils.data import DataLoader, Dataset
from utils.utils_distributed_sampler import DistributedSampler
from utils.utils_distributed_sampler import get_dist_info, worker_init_fn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

def iccFileFixed(image_path):
    img = Image.open(image_path)
    img.info.pop('icc_profile', None)
    img.save(image_path)

import random
import cv2
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data_file, real_world_transform, web_image_transform, validation_transform=None):
        self.images_dir = []
        self.labels = []
        self.real_world_transform = real_world_transform
        self.web_image_transform = web_image_transform
        self.validation_transform = validation_transform
        self.default_image = "kaggle/input/myfolder/content/all/train/class1681/qr125_jpg.rf.68be0b0e1167e631aa7b6f830f1de1ca33.jpg"
        self.default_label = 1

        # Read the file and process image paths and labels
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                image_dir, label = line.strip().split('\t')
                image_dir = image_dir.replace("/kaggle", "kaggle")  # Normalize paths
                self.images_dir.append(image_dir)
                self.labels.append(int(label))

        # Shuffle the dataset
        combined = list(zip(self.images_dir, self.labels))
        random.shuffle(combined)
        self.images_dir, self.labels = zip(*combined)

        # Convert back to lists if needed
        self.images_dir = list(self.images_dir)
        self.labels = list(self.labels)

        print(f"Loaded {len(self.images_dir)} images with {len(set(self.labels))} unique labels.")

    def __len__(self):
        return len(self.images_dir)

    def __getitem__(self, idx):
        # Try to retrieve the image path and label
        try:
            img_path = self.images_dir[idx]
            label = self.labels[idx]
        except IndexError:
            print(f"Error: Index {idx} is out of bounds. Using default image and label.")
            img_path = self.default_image  # Default image if index is out of range
            label = self.default_label  # Default label if index is out of range

        # Attempt to load the image and handle errors gracefully
        try:
            # Try to read the image using OpenCV
            image = cv2.imread(img_path)

            # If image is None, it indicates the file is invalid or not an image
            if image is None:
                raise OSError(f"Failed to load image at {img_path}.")

            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        except (OSError, cv2.error) as e:
            # If any error occurs, use the default image
            print(f"Error loading image {img_path}: {e}. Using default image.")
            image = cv2.imread(self.default_image)
            if image is None:
                raise ValueError(f"Default image not found or invalid: {self.default_image}.")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            label = self.default_label  # Assign default label

        # Apply transformations if any
        try:
            if not self.validation_transform:
                if "sucbat" not in img_path and "train" not in img_path:
                    image = self.web_image_transform(image=image)["image"]
                else:
                    image = self.real_world_transform(image=image)["image"]
            else:
                image = self.validation_transform(image=image)["image"]

        except Exception as e:
            print(f"Error during transformation: {e}. Returning the original image.")
            # Return the image as is if transformation fails
            # Optionally, you could apply a default transformation or handle the error in another way

        return image, label


def get_dataloader(
        root_dir,
        batch_size,
        dali=False,
        dali_aug=False,
        seed=2048,
        num_workers=2,
) -> Iterable:

    transform = A.Compose([
        A.Resize(height =224, width=224,interpolation=3),
        # A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=(0, 0, 0)),
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5),  # Sun flare at the top of the image
            angle_lower=None,
            angle_upper=None,
            num_flare_circles_lower=None,
            num_flare_circles_upper=None,
            src_radius=50,
            src_color=(255, 255, 255),  # Flare color (white)
            angle_range=(0, 1),  # Flare angle range
            num_flare_circles_range=(2, 3),  # Number of flare circles
            p=1,  # Probability of applying the effect
        ),
        A.ImageCompression(
            compression_type="jpeg",  # JPEG compression
            quality_range=(30, 40),  # Quality range for compression
            p=.5,  # Probability of applying the effect
        ),
        A.Sharpen(
            alpha=(0.1, 0.2),  # Sharpening factor
            lightness=(0.5, 0.6),  # Lightness of the sharpen effect
            p=.5),
            # Probability of applying the effect),
        A.Defocus(
            radius=(1, 2),  # Defocus radius
            alias_blur=(0.8, 0.9),  # Alias blur amount
            p=.5,  # Probability of applying the effect
        ),
        A.Affine(
            rotate=(-15, 15),  # Rotate the image
            shear=(-15, 15),  # Shear the image
            interpolation=cv2.INTER_NEAREST,
            fit_output=False,
            rotate_method="largest_box",
            mode=cv2.BORDER_REPLICATE,
            cval=0,
            p=0.8,  # Probability of applying the effect
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    WebTransform = A.Compose([
        A.Resize(height =224, width=224,interpolation=3),
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5),  # Sun flare at the top of the image
            angle_lower=None,
            angle_upper=None,
            num_flare_circles_lower=None,
            num_flare_circles_upper=None,
            src_radius=50,
            src_color=(255, 255, 255),  # Flare color (white)
            angle_range=(0, 1),  # Flare angle range
            num_flare_circles_range=(2, 3),  # Number of flare circles
            p=0.5,  # Probability of applying the effect
        ),
        A.Affine(
            rotate=(-15, 15),  # Rotate the image
            shear=(-15, 15),  # Shear the image
            interpolation=cv2.INTER_NEAREST,
            fit_output=False,
            rotate_method="largest_box",
            mode=cv2.BORDER_REPLICATE,
            cval=0,
            p=0.5,  # Probability of applying the effect
        ),
        A.GaussianBlur(blur_limit=(5, 10), sigma_limit=5, p=1),
        A.ImageCompression(compression_type="jpeg", quality_range=(30, 80), p=1, ),
        A.Sharpen(
            alpha=(0.1, 0.2),  # Sharpening factor
            lightness=(0.5, 0.6),  # Lightness of the sharpen effect
            p=.5),
            # Probability of applying the effect),
        A.Defocus(
            radius=(1, 2),  # Defocus radius
            alias_blur=(0.8, 0.9),  # Alias blur amount
            p=.5,  # Probability of applying the effect
        ),
        A.Affine(
            rotate=(-15, 15),  # Rotate the image
            shear=(-15, 15),  # Shear the image
            interpolation=cv2.INTER_NEAREST,
            fit_output=False,
            rotate_method="largest_box",
            mode=cv2.BORDER_REPLICATE,
            cval=0,
            p=0.8,  # Probability of applying the effect
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    trungbay_transform = A.Compose([
        A.Resize(height =224, width=224,interpolation=3),
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5),  # Sun flare at the top of the image
            angle_lower=None,
            angle_upper=None,
            num_flare_circles_lower=None,
            num_flare_circles_upper=None,
            src_radius=50,
            src_color=(255, 255, 255),  # Flare color (white)
            angle_range=(0, 1),  # Flare angle range
            num_flare_circles_range=(2, 3),  # Number of flare circles
            p=0.5,  # Probability of applying the effect
        ),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    train_set = CustomDataset(data_file='kaggle/working/data.txt', real_world_transform=trungbay_transform,
                              web_image_transform=WebTransform)

    rank, world_size = get_dist_info()

    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=False, num_workers=num_workers, pin_memory=True,
                                               drop_last=True, sampler = train_sampler)

    return train_loader

def get_valloader(
        root_dir,
        batch_size,
        dali=False,
        dali_aug=False,
        seed=2048,
        num_workers=2,
) -> Iterable:

    transform = A.Compose([
        A.LongestMaxSize(max_size=224, interpolation=3),
        A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=(0, 0, 0)),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    val_set = CustomDataset(data_file='kaggle/working/validation.txt', real_world_transform=transform,
                            web_image_transform=transform, validation_transform=transform)

    rank, world_size = get_dist_info()

    val_sampler = DistributedSampler(
        val_set, num_replicas=world_size, rank=rank, shuffle=True, seed=seed)

    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                               shuffle=False, num_workers=num_workers, pin_memory=True,
                                               drop_last=True, sampler = val_sampler)

    return val_loader


class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):

    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank, non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch

class SyntheticDataset(Dataset):
    def __init__(self):
        super(SyntheticDataset, self).__init__()
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.int32)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).squeeze(0).float()
        img = ((img / 255) - 0.5) / 0.5
        self.img = img
        self.label = 1

    def __getitem__(self, index):
        return self.img, self.label

    def __len__(self):
        return 1000000

if __name__ == '__main__':
    from tqdm import tqdm
    train_loader = get_dataloader(
        "root_dir",
        10,
        dali=False,
        dali_aug=False,
        seed=2048,
        num_workers=2,
    )
    # print("run")
    # for _ in tqdm(train_loader):
    #     pass
    # import os
    # print(os.listdir("kaggle/input/amazonProductDataset/images/71vyGFnOeGL"))