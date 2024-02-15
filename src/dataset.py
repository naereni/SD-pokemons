import os
import random

import numpy as np
import torch
from datasets import load_dataset
from transformers import CLIPTokenizer
from torchvision import transforms

from config import config


train_transforms = transforms.Compose(
    [
        transforms.Resize(
            config.resolution, interpolation=transforms.InterpolationMode.BILINEAR
        ),
        (
            transforms.CenterCrop(config.resolution)
            if config.center_crop
            else transforms.RandomCrop(config.resolution)
        ),
        (
            transforms.RandomHorizontalFlip()
            if config.random_flip
            else transforms.Lambda(lambda x: x)
        ),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)


def tokenize_captions(examples, is_train=True):
    tokenizer = CLIPTokenizer.from_pretrained(
        config.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=config.revision,
    )
    captions = []
    for caption in examples[config.caption_column]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column `{config.caption_column}` should contain either strings or lists of strings."
            )
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids


def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[config.image_column]]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    examples["input_ids"] = tokenize_captions(examples)
    return examples


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}


def get_train_dataloader(accelerator):
    if config.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            config.dataset_name,
            config.dataset_config_name,
            cache_dir=config.cache_dir,
            data_dir=config.train_data_dir,
        )
    else:
        data_files = {}
        if config.train_data_dir is not None:
            data_files["train"] = os.path.join(config.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=config.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

    with accelerator.main_process_first():
        if config.max_train_samples is not None:
            dataset["train"] = (
                dataset["train"]
                .shuffle(seed=config.seed)
                .select(range(config.max_train_samples))
            )
        # Set the training transforms
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=config.train_batch_size,
        num_workers=config.dataloader_num_workers,
    )
