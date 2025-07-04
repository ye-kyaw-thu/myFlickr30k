"""
Image Captioning Demo for Intern3.
Written by Ye Kyaw Thu, LU Lab., Myanmar
Last Updated: 30 June 2025
"""

import os
import argparse
import json
import pickle
import warnings
from collections import Counter, OrderedDict
from dataclasses import dataclass
from functools import reduce
from itertools import count, islice
from pathlib import Path
from re import sub
from string import ascii_lowercase
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from PIL import Image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
try:
    # For newer versions of PyTorch Lightning
    from torchmetrics import Metric
except ImportError:
    # For older versions (deprecated)
    from pytorch_lightning.metrics import Metric
from tqdm import tqdm
import heapq

from torchmetrics.text.rouge import ROUGEScore
from torchmetrics.text.chrf import CHRFScore
from torchmetrics.text.bleu import BLEUScore
#from torchmetrics.text.cider import CIDErScore
try:
    from pycocoevalcap.cider.cider import Cider
    HAVE_CIDER = True
except ImportError:
    HAVE_CIDER = False
    print("Warning: pycocoevalcap not installed - CIDEr scores will not be computed")
    print("Install with: pip install pycocoevalcap")

# Constants
BOS = "[cls]"
EOS = "[sep]"
UNK = "[unk]"
PAD = "[pad]"
smoothie = SmoothingFunction().method4

# Suppress warnings
warnings.filterwarnings("ignore")

class WordTokenizer:
    """Implements a simple, pure-python, word-based tokenizer"""
    def __init__(self) -> None:
        self._incrementer = count()
        self.tokens = dict()
        self._padding = False
        self._truncation = False
        self.is_trained = False
        self.max_length = None
        self.padding_config = {"pad_id": 0}  # Initialize with default pad_id
        self.special_tokens = []
        self.max_vocab_size = None
        self.min_frequency = None
        self.ids = {}

    def check_trained(self):
        if not self.is_trained:
            raise ValueError("Train tokenizer before using it")

    def train(self, filename: Union[str, Path], vocab_size: int, min_frequency: int, 
              special_tokens: List[int], initial_alphabet: str = None, limit_alphabet: int = None):
        if self.is_trained:
            raise ValueError("Don't retrain; load config or use a new instance instead")
        self.max_vocab_size = vocab_size
        self.min_frequency = min_frequency

        if len(special_tokens) != 4:
            raise ValueError("Special tokens must have exactly 4 values: PAD, BOS, EOS, UNK")

        self.special_tokens = special_tokens
        for t in self.special_tokens:
            self.tokens[t] = next(self._incrementer)

        with open(filename, "r", encoding='utf-8') as f:
            data = f.readlines()
        data = map(lambda s: s.split(), data)
        counter = Counter()
        list(map(lambda lst: counter.update(lst), data))
        vocab = counter.most_common(self.max_vocab_size - len(self.tokens))
        vocab = filter(lambda tup: tup[1] >= self.min_frequency, vocab)
        vocab = filter(lambda tup: tup[0] not in self.tokens, vocab)
        for word, _ in vocab:
            self.tokens[word] = next(self._incrementer)
        self.ids = {v: k for k, v in self.tokens.items()}
        self.is_trained = True

    def enable_truncation(self, max_length):
        """Enable truncation of sequences to max_length"""
        self._truncation = True
        self.max_length = max_length

    def enable_padding(self, pad_id=0):
        """Enable padding of sequences"""
        if not self._truncation:
            raise AttributeError("Enable truncation before enabling padding")
        self._padding = True
        self.padding_config["pad_id"] = pad_id  # Update the pad_id in config

    @property
    def padding(self):
        """Return padding configuration when accessed as property"""
        return self.padding_config if self._padding else False


    def get_vocab_size(self):
        """Return the size of the vocabulary"""
        return len(self.tokens)

    @property
    def vocab_size(self):
        """Return the size of the vocabulary (property version)"""
        return len(self.tokens)

    @property
    def truncation(self):
        return self._truncation

    def token_to_id(self, token):
        self.check_trained()
        return self.tokens.get(token)

    def decode_batch(self, batch: List[List[int]], skip_special_tokens=True):
        return list(self._decode_batch(batch, skip_special_tokens))

    def _decode_batch(self, batch: List[List[int]], skip_special_tokens):
        self.check_trained()
        for element in batch:
            element = map(lambda i: self.ids.get(i, ""), element)
            if skip_special_tokens:
                element = filter(lambda w: w not in self.special_tokens, element)
            yield " ".join(element)

    def encode_batch(self, batch: List[str]):
        return list(self._encode_batch(batch))

    def _encode_batch(self, batch: List[str]):
        self.check_trained()
        for element in batch:
            split = element.strip().split(" ")
            if self._truncation and len(split) > self.max_length:
                split = split[:self.max_length]
            if self._padding and len(split) < self.max_length:
                split = split + [self.padding_config["pad_id"]] * (self.max_length - len(split))
            yield list(map(lambda w: self.tokens.get(w, 3), split))  # Default to UNK token (3) if word not found

    def save_model(self, directory, filename):
        self.check_trained()
        with open(os.path.join(directory, filename), "wb") as f:
            pickle.dump(self.__dict__, f, 3)

    def load_model(self, directory, filename):
        with open(os.path.join(directory, filename), "rb") as f:
            tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)

    @property
    def config(self):
        return self.__dict__

    def load_config(self, config):
        self.__dict__.update(config)

def __init__(
    self,
    flickr_txt=None,
    flickr_dir=None,
    batch_size=64,
    val_size=1024,
    test_size=1024,
    remove_prefixes=False,  # Changed from True to False for non-English
    transform="augment",
    target_transform="shuffle",
    val_transform="normalize",
    val_target_transform="tokenize",
    vocab_size=5000,
    min_word_occurrences=1,
    max_caption_length=25,
    dev_set=None,
    num_workers=4,
    pin_memory=True,
):
    super().__init__()
    self.flickr_txt = flickr_txt
    self.flickr_dir = flickr_dir
    self.batch_size = batch_size
    self.val_size = val_size
    self.test_size = test_size

    # Remove English-specific prefixes
    self.remove_prefixes = []  # Empty list for non-English


    def token_to_id(self, token):
        self.check_trained()
        return self.tokens.get(token)

    @property
    def vocab_size(self):
        return len(self.tokens)

    def get_vocab_size(self):
        return self.vocab_size

    def decode_batch(self, batch: List[List[int]], skip_special_tokens=True):
        return list(self._decode_batch(batch, skip_special_tokens))

    def _decode_batch(self, batch: List[List[int]], skip_special_tokens):
        self.check_trained()
        for element in batch:
            element = map(lambda i: self.ids.get(i, ""), element)
            if skip_special_tokens:
                element = filter(lambda w: w not in self.special_tokens, element)
            yield " ".join(element)

    def encode_batch(self, batch: List[str]):
        return list(self._encode_batch(batch))

    def _encode_batch(self, batch: List[str]):
        self.check_trained()
        for element in batch:
            split = element.strip().split(" ")
            if self.truncation and len(split) > self.max_length:
                split = split[: self.max_length]
            if self.padding and len(split) < self.max_length:
                split = split + [self.special_tokens[0]] * (self.max_length - len(split))
            yield list(map(lambda w: self.tokens.get(w, 3), split))

    def save_model(self, directory, filename):
        self.check_trained()
        with open(os.path.join(directory, filename), "wb") as f:
            pickle.dump(self.__dict__, f, 3)

    def load_model(self, directory, filename):
        with open(os.path.join(directory, filename), "rb") as f:
            tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)

    def enable_padding(self):
        if not self.truncation:
            raise AttributeError("Enable truncation before enabling padding")
        self.padding = {"pad_id": 0}

    def enable_truncation(self, max_length):
        self.truncation = True
        self.max_length = max_length

    @property
    def config(self):
        return self.__dict__

    def load_config(self, config):
        self.__dict__.update(config)

class NormalizeInverse(transforms.Normalize):
    """Invert an image normalization. Default values are for Flickr30k dataset."""
    def __init__(self, mean=(0.4435, 0.4201, 0.3837), std=(0.2814, 0.2734, 0.2820)):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())

def load_flickr_txt(txt_path, img_dir):
    """Load Flickr30k data from text file format"""
    captions = {}
    found_images = 0
    
    with open(txt_path, 'r', encoding='utf-8') as f:
        # Skip header line
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            # Split on tab to separate image filename from caption
            parts = line.split('\t')
            if len(parts) != 2:
                continue  # Skip malformed lines
                
            img_file, caption = parts
            img_file = img_file.strip()
            caption = caption.strip()
            
            # Create full image path
            img_path = os.path.join(img_dir, 'Images', img_file)
            
            # Only add if image exists
            if os.path.exists(img_path):
                if img_path not in captions:
                    captions[img_path] = {}
                # Assign sequential caption numbers (0-4)
                caption_num = len(captions[img_path])
                captions[img_path][caption_num] = caption
                found_images += 1
            else:
                print(f"Warning: Image not found at {img_path} - skipping")
    
    if not captions:
        raise ValueError(f"No valid image-caption pairs found. Check paths:\n"
                        f" - Image directory: {os.path.join(img_dir, 'Images')}\n"
                        f" - Caption file: {txt_path}")
    
    print(f"Successfully loaded {found_images} images with captions")
    
    # Convert to DataFrame
    max_captions = max(len(v) for v in captions.values())
    data = []
    
    for img_path, caption_dict in captions.items():
        row = {'path': img_path}
        for i in range(max_captions):
            row[i] = caption_dict.get(i, '')
        data.append(row)
        
    return pd.DataFrame(data)

def remove_prefixes(captions_df: pd.DataFrame, prefixes: List[str]):
    """Strip a list of prefixes from captions"""
    for prefix in prefixes:
        captions_df.iloc[:, 1:] = captions_df.iloc[:, 1:].applymap(
            lambda s: sub("^\s*" + prefix, "", s)
        )
    return captions_df

def add_special_tokens(df, pad=PAD, start=BOS, end=EOS, unk=UNK):
    """Add start and end tokens to strings in dataframe"""
    for col in df.iloc[:, 1:].columns:
        if not df.loc[0, col].startswith(start):
            df[col] = start + " " + df[col] + " " + end
    return df, [pad, start, end, unk]

def tokens_to_ids(tokenizer, tokens):
    """Returns dict of 'token: id' for tokens in tokenizer"""
    if hasattr(tokenizer, "token_to_id"):
        return {t: tokenizer.token_to_id(t) for t in tokens}
    else:
        return {t: tokenizer.convert_tokens_to_ids(t) for t in tokens}

def vocab_size(tokenizer):
    """Returns vocab size from tokenizer"""
    if hasattr(tokenizer, "get_vocab_size"):
        return tokenizer.get_vocab_size()
    elif hasattr(tokenizer, "vocab_size"):
        return tokenizer.vocab_size
    else:
        return len(tokenizer.tokens)

def ids_to_captions(ids_tensor, tokenizer, skip_special_tokens=False):
    """Return captions from tensor of ids using tokenizer"""
    if isinstance(ids_tensor, list):
        ids_tensor = torch.tensor(ids_tensor)
    if ids_tensor.dim() == 1:
        ids_tensor = ids_tensor.reshape(1, -1)
    ids_tensor = ids_tensor.cpu()
    
    if isinstance(tokenizer, WordTokenizer):
        strings = tokenizer.decode_batch(ids_tensor.tolist(), skip_special_tokens=skip_special_tokens)
    else:
        strings = tokenizer.batch_decode(ids_tensor, skip_special_tokens=skip_special_tokens)
    
    if skip_special_tokens:
        strings = [s.lstrip(BOS).partition(EOS)[0] for s in strings]
    return strings

class TokenizeTransform:
    __slots__ = ["tokenizer"]
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, captions):
        """A list of strings to a tensor"""
        if isinstance(self.tokenizer, WordTokenizer):
            return torch.tensor([t for t in self.tokenizer.encode_batch(captions)])

class ShuffleCaptions:
    """Shuffle a (n_captions, seq_len) tensor of captions"""
    __slots__ = []
    def __call__(self, tensor):
        idxs = torch.randperm(tensor.shape[0])
        return tensor[idxs, :]

class CaptioningDataset(Dataset):
    """Pytorch dataset of Flickr images and captions"""
    def __init__(self, df, split, transform, target_transform, val_size, test_size, random_state=42):
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.val_size = val_size
        self.test_size = test_size
        self.random_state = random_state
        self.subset(df)

    def subset(self, df):
        if self.split not in {"train", "test", "val"}:
            raise ValueError
        train, test = train_test_split(df, test_size=self.test_size, random_state=self.random_state)
        if self.split == "test":
            self.split_df = test
            return
        train, val = train_test_split(train, test_size=self.val_size, random_state=self.random_state)
        if self.split == "train":
            self.split_df = train
        elif self.split == "val":
            self.split_df = val

    def __len__(self):
        return self.split_df.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        img_path = self.split_df.iloc[idx, 0]
        
        try:
            image = Image.open(img_path).convert("RGB").copy()
        except FileNotFoundError:
            # Return a blank image if file not found
            print(f"Warning: Image not found at {img_path}, using blank image")
            image = Image.new('RGB', (224, 224), color=(0, 0, 0))

        captions = self.split_df.iloc[idx, 1:].to_list()

        if self.transform:
            with torch.no_grad():
                image = self.transform(image)

        if self.target_transform:
            with torch.no_grad():
                captions = self.target_transform(captions)

        return {
            "image": image,
            "captions": captions,
            "path": img_path  # Add this line to include the path
        }

class DatasetBuilder:
    """Utility class to reduce boilerplate in data module"""
    def __init__(self, captions_df, transform, target_transform, val_transform, 
                 val_target_transform, tokenizer, val_size, test_size):
        self.captions_df = captions_df
        self.transform = transform
        self.target_transform = target_transform
        self.val_transform = val_transform
        self.val_target_transform = val_target_transform
        self.tokenizer = tokenizer
        self.val_size = val_size
        self.test_size = test_size

    def new(self, split):
        if split in ("val", "test"):
            return CaptioningDataset(
                self.captions_df, split, self.val_transform, 
                self.val_target_transform, self.val_size, self.test_size
            )
        else:
            return CaptioningDataset(
                self.captions_df, split, self.transform, 
                self.target_transform, self.val_size, self.test_size
            )

class CombinedDataModule(pl.LightningDataModule):
    def __init__(
        self,
        flickr_txt=None,
        flickr_dir=None,
        batch_size=64,
        val_size=1024,
        test_size=1024,
        remove_prefixes=True,
        transform="augment",
        target_transform="shuffle",
        val_transform="normalize",
        val_target_transform="tokenize",
        vocab_size=5000,
        min_word_occurrences=1,
        max_caption_length=25,
        dev_set=None,
        num_workers=4,
        pin_memory=True,
    ):
        super().__init__()
        self.flickr_txt = flickr_txt
        self.flickr_dir = flickr_dir
        self.batch_size = batch_size
        self.val_size = val_size
        self.test_size = test_size

        self.remove_prefixes = (
            [
                "there is",
                "there are",
                "this is",
                "these are",
                "a photo of",
                "a picture of",
                "an image of",
            ]
            if remove_prefixes is True
            else remove_prefixes
        )

        self.transform = transform
        self.target_transform = target_transform
        self.val_transform = val_transform
        self.val_target_transform = val_target_transform

        self.vocab_size = vocab_size
        self.min_word_occurrences = min_word_occurrences
        self.max_caption_length = max_caption_length + 1  # including start token
        self.dev_set = dev_set
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.is_setup = False

    def setup(self, stage=None):
        if self.is_setup:
            self.make_loader(stage)
            return None
            
        try:
            captions_df = load_flickr_txt(self.flickr_txt, self.flickr_dir)
            
            if self.remove_prefixes:
                captions_df = remove_prefixes(captions_df, self.remove_prefixes)

            if self.dev_set:
                captions_df = captions_df.iloc[:self.dev_set]
                
            self.captions_df, self.special_tokens = add_special_tokens(captions_df)
            
            # Train tokenizer
            self.tokenizer = WordTokenizer()
            strings = self.captions_df.iloc[:, 1:].stack(-1).reset_index(drop=True)
            strings.to_csv("temp_captions.txt", header=False, index=False)
            self.tokenizer.train(
                "temp_captions.txt",
                vocab_size=self.vocab_size,
                min_frequency=self.min_word_occurrences,
                special_tokens=self.special_tokens,
            )
            self.tokenizer.enable_truncation(self.max_caption_length)
            self.tokenizer.enable_padding()
            os.remove("temp_captions.txt")

            # Image transforms
            if self.transform == "augment":
                random_xforms = [
                    transforms.RandomAffine(degrees=30, scale=(0.9, 1.1), shear=10),
                    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2),
                ]
                img_xforms = [
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply(nn.ModuleList(random_xforms), p=0.5),
                ]
                self.transform = transforms.Compose([
                    transforms.Compose(img_xforms),
                    transforms.ToTensor(),
                    transforms.RandomErasing(p=0.3),
                    transforms.Normalize((0.4435, 0.4201, 0.3837), (0.2814, 0.2734, 0.2820)),
                ])
            elif self.transform == "normalize":
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4435, 0.4201, 0.3837), (0.2814, 0.2734, 0.2820)),
                ])

            if self.val_transform == "normalize":
                self.val_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4435, 0.4201, 0.3837), (0.2814, 0.2734, 0.2820)),
                ])

            if self.target_transform == "shuffle":
                self.target_transform = transforms.Compose([
                    TokenizeTransform(self.tokenizer), 
                    ShuffleCaptions()
                ])
            elif self.target_transform == "tokenize":
                self.target_transform = transforms.Compose([TokenizeTransform(self.tokenizer)])

            if self.val_target_transform == "tokenize":
                self.val_target_transform = TokenizeTransform(self.tokenizer)

            self.dbuild = DatasetBuilder(
                self.captions_df,
                self.transform,
                self.target_transform,
                self.val_transform,
                self.val_target_transform,
                self.tokenizer,
                self.val_size,
                self.test_size,
            )
            self.make_loader(stage)
            self.is_setup = True

        except Exception as e:
            print(f"Error during setup: {str(e)}")
            raise

        # Train tokenizer
        self.tokenizer = WordTokenizer()
        strings = self.captions_df.iloc[:, 1:].stack(-1).reset_index(drop=True)
        strings.to_csv("temp_captions.txt", header=False, index=False)
        self.tokenizer.train(
            "temp_captions.txt",
            vocab_size=self.vocab_size,
            min_frequency=self.min_word_occurrences,
            special_tokens=self.special_tokens,
        )
        self.tokenizer.enable_truncation(self.max_caption_length)
        self.tokenizer.enable_padding()
        os.remove("temp_captions.txt")

        # Image transforms
        if self.transform == "augment":
            random_xforms = [
                transforms.RandomAffine(degrees=30, scale=(0.9, 1.1), shear=10),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2),
            ]
            img_xforms = [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply(nn.ModuleList(random_xforms), p=0.5),
            ]
            self.transform = transforms.Compose([
                transforms.Compose(img_xforms),
                transforms.ToTensor(),
                transforms.RandomErasing(p=0.3),
                transforms.Normalize((0.4435, 0.4201, 0.3837), (0.2814, 0.2734, 0.2820)),
            ])
        elif self.transform == "normalize":
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.4435, 0.4201, 0.3837), (0.2814, 0.2734, 0.2820)),
            ])

        if self.val_transform == "normalize":
            self.val_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.4435, 0.4201, 0.3837), (0.2814, 0.2734, 0.2820)),
            ])

        if self.target_transform == "shuffle":
            self.target_transform = transforms.Compose([
                TokenizeTransform(self.tokenizer), 
                ShuffleCaptions()
            ])
        elif self.target_transform == "tokenize":
            self.target_transform = transforms.Compose([TokenizeTransform(self.tokenizer)])

        if self.val_target_transform == "tokenize":
            self.val_target_transform = TokenizeTransform(self.tokenizer)

        self.dbuild = DatasetBuilder(
            self.captions_df,
            self.transform,
            self.target_transform,
            self.val_transform,
            self.val_target_transform,
            self.tokenizer,
            self.val_size,
            self.test_size,
        )
        self.make_loader(stage)
        self.is_setup = True

    def make_loader(self, stage):
        if stage == "fit" or stage is None:
            self.train = self.dbuild.new("train")
            self.val = self.dbuild.new("val")

        if stage == "test" or stage is None:
            self.test = self.dbuild.new("test")

    def train_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(
            self.train,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(
            self.val,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        return DataLoader(
            self.test,
            batch_size=batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

class ImageFeatureExtractor(nn.Module):
    def __init__(self, encoder="resnext50_32x4d", freeze_weights=True, remove_last="infer", 
                 pooling="infer", convolution_in=False, projection_in="infer", projection_out=1024):
        super().__init__()
        
        # Map user-friendly names to actual model names
        model_map = {
            "resnet50": "resnet50",
            "resnet101": "resnet101",
            "resnet152": "resnet152",
            "mobilenetv2": "mobilenet_v2",
            "vgg16": "vgg16",
            "resnext50": "resnext50_32x4d",
            "resnext101": "resnext101_32x8d"
        }
        
        if isinstance(encoder, str):
            if encoder in model_map:
                model_name = model_map[encoder]
                self.encoder = torchvision.models.__dict__[model_name](pretrained=True)
            else:
                raise ValueError(f"Encoder {encoder} not supported. Available options: {list(model_map.keys())}")
        else:
            self.encoder = encoder
            
        if freeze_weights:
            for param in self.encoder.parameters():
                param.requires_grad_(False)
            if hasattr(self.encoder, "eval"):
                self.encoder.eval()
                
        # Handle different architectures
        if "vgg" in encoder:
            # For VGG, we'll use features before the classifier
            if remove_last == "infer":
                remove_last = 1  # Remove the classifier
            if remove_last:
                self.encoder = nn.Sequential(*list(self.encoder.children())[: (-1 * remove_last)])
            # VGG output features dimension
            projection_in = 512 if projection_in == "infer" else projection_in
        else:
            # For ResNet/ResNeXt/MobileNet
            if remove_last == "infer":
                remove_last = 2 if "resnet" in encoder or "resnext" in encoder else 1
            if remove_last:
                self.encoder = nn.Sequential(*list(self.encoder.children())[: (-1 * remove_last)])
            # Set proper projection dimensions based on encoder type
            if projection_in == "infer":
                if "resnet" in encoder or "resnext" in encoder:
                    projection_in = 2048
                elif "mobilenet" in encoder:
                    projection_in = 1280
                else:
                    projection_in = None

        if pooling == "infer":
            pooling = True
        if pooling:
            self.pooling = nn.AdaptiveAvgPool2d(1)
        else:
            self.pooling = None
            
        # Ensure projection_out matches the hidden_size of the RNN
        self.projection_in = projection_in
        self.projection_out = projection_out
            
        if projection_in:
            self.projector = nn.Sequential(
                OrderedDict([
                    ("linear", nn.Linear(projection_in, projection_out)),
                    ("relu", nn.ReLU()),
                ])
            )
        else:
            self.projector = None
            
        if convolution_in == "infer":
            convolution_in = projection_in
            self.convolution = nn.Conv2d(convolution_in, projection_out, 1)
        else:
            self.convolution = None

    # ... rest of the class remains the same ...
        self.projection_in = projection_in
        self.projection_out = projection_out

    def init_weights(self, method="kaiming"):
        if method not in {"kaiming", "xavier"}:
            raise ValueError(f"Initialization method {method} not supported")

        if method == "kaiming":
            if self.convolution:
                nn.init.kaiming_normal_(self.convolution.weight)
            if self.projector:
                nn.init.kaiming_normal_(self.projector.linear.weight)
        elif method == "xavier":
            if self.convolution:
                nn.init.xavier_normal_(self.convolution.weight)
            elif self.projector:
                nn.init.xavier_normal_(self.projector.linear.weight)

        if self.convolution:
            nn.init.zeros_(self.convolution.bias)
        if self.projector:
            nn.init.zeros_(self.projector.linear.bias)

    def forward(self, image):
        raise NotImplementedError("Don't the extractor directly; use the submodels instead.")

class WordEmbedder(nn.Module):
    """Word embedding module with padding handling"""
    def __init__(self, wordvec_dim, tokenizer):
        super().__init__()
        self.wordvec_dim = wordvec_dim
        self.vocab_size = vocab_size(tokenizer)
        self._pad = tokens_to_ids(tokenizer, [PAD])[PAD]
        self.embedder = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.wordvec_dim,
            padding_idx=self._pad,
        )

    def init_weights(self, method="kaiming"):
        if method not in {"kaiming", "xavier"}:
            raise ValueError(f"Initialization method {method} not supported")

        if method == "kaiming":
            nn.init.kaiming_normal_(self.embedder.weight)
        elif method == "xavier":
            nn.init.xavier_normal_(self.embedder.weight)

        with torch.no_grad():
            self.embedder.weight[self._pad].fill_(0.0)

    def forward(self, x):
        return self.embedder(x)

class RNN(nn.Module):
    """RNN decoder module"""
    def __init__(self, input_size, hidden_size, num_rnns, num_layers, nonlinearity, dropout, bidirectional):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_rnns = num_rnns
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.nonlinearity = 'tanh' if nonlinearity is None else nonlinearity

        for i in range(num_rnns):
            setattr(
                self, f"rnn{i}",
                nn.RNN(
                    self.input_size, self.hidden_size,
                    num_layers=self.num_layers,
                    nonlinearity=self.nonlinearity,
                    batch_first=True,
                    dropout=dropout,
                    bidirectional=bidirectional,
                )
            )

    def init_weights(self, method="kaiming"):
        if method not in {"kaiming", "xavier"}:
            raise ValueError(f"Initialization method {method} not supported")

        if method == "kaiming":
            weight_fcn = lambda w: nn.init.kaiming_normal_(w)
        elif method == "xavier":
            weight_fcn = lambda w: nn.init.xavier_normal_(w)

        bias_fcn = lambda b: nn.init.zeros_(b)

        for i in range(self.num_rnns):
            params = list(sum(zip(*getattr(self, f"rnn{i}")._all_weights), ()))
            weights = params[: (len(params) // 2)]
            list(map(weight_fcn, [getattr(self, f"rnn{i}")._parameters[w] for w in weights]))
            list(map(bias_fcn, [getattr(self, f"rnn{i}")._parameters[b] for b in params[(len(params) // 2):]]))

    def forward(self, wds, h0):
        rnn_outs = {}
        for i in range(self.num_rnns):
            rnn_outs[f"rnn{i}"] = getattr(self, f"rnn{i}")(wds[:, i, :, :], h0)
        return rnn_outs

class LSTM(nn.Module):
    """LSTM decoder module"""
    def __init__(self, input_size, hidden_size, num_rnns, num_layers, nonlinearity, dropout, bidirectional):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_rnns = num_rnns
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        if nonlinearity == "relu":
            warnings.warn("LSTM uses tanh and sigmoid internally; relu argument ignored")

        for i in range(num_rnns):
            setattr(
                self, f"rnn{i}",
                nn.LSTM(
                    self.input_size, self.hidden_size,
                    num_layers=self.num_layers,
                    batch_first=True,
                    dropout=dropout,
                    bidirectional=bidirectional,
                )
            )

    def init_weights(self, method="kaiming"):
        if method not in {"kaiming", "xavier"}:
            raise ValueError(f"Initialization method {method} not supported")

        if method == "kaiming":
            weight_fcn = lambda w: nn.init.kaiming_normal_(w)
        elif method == "xavier":
            weight_fcn = lambda w: nn.init.xavier_normal_(w)

        bias_fcn = lambda b: nn.init.zeros_(b)

        for i in range(self.num_rnns):
            params = list(sum(zip(*getattr(self, f"rnn{i}")._all_weights), ()))
            weights = params[: (len(params) // 2)]
            list(map(weight_fcn, [getattr(self, f"rnn{i}")._parameters[w] for w in weights]))
            list(map(bias_fcn, [getattr(self, f"rnn{i}")._parameters[b] for b in params[(len(params) // 2):]]))

    def forward(self, wds, hn):
        rnn_outs = {}
        for i in range(self.num_rnns):
            rnn_outs[f"rnn{i}"] = getattr(self, f"rnn{i}")(wds[:, i, :, :], hn)
        return rnn_outs

class ParallelFCScorer(nn.Module):
    """Parallel fully connected scorer module"""
    def __init__(self, num_scorers, hidden_size, vocab_size):
        super().__init__()
        self.num_scorers = num_scorers
        for i in range(num_scorers):
            setattr(self, f"fc{i}", nn.Linear(hidden_size, vocab_size))

    def init_weights(self, method="kaiming"):
        if method not in {"kaiming", "xavier"}:
            raise ValueError(f"Initialization method {method} not supported")

        if method == "kaiming":
            weight_fcn = lambda fc: nn.init.kaiming_normal_(fc.weight)
        elif method == "xavier":
            weight_fcn = lambda fc: nn.init.xavier_normal_(fc.weight)

        bias_fcn = lambda fc: nn.init.zeros_(fc.bias)

        for i in range(self.num_scorers):
            weight_fcn(getattr(self, f"fc{i}"))
            bias_fcn(getattr(self, f"fc{i}"))

    def forward(self, x):
        fc_outs = {}
        for i in range(self.num_scorers):
            fc_outs[f"fc{i}"] = getattr(self, f"fc{i}")(x[f"rnn{i}"][0])
        return fc_outs

def temporal_softmax_loss(x, y, ignore_index):
    """Temporal softmax loss function"""
    loss = F.cross_entropy(x.transpose(-1, 1), y, ignore_index=ignore_index, reduction="mean")
    return loss

def multi_caption_temporal_softmax_loss(x, y, ignore_index):
    """Multi-caption temporal softmax loss"""
    loss = torch.zeros(1, device=x["fc0"].device)
    for i in range(y.shape[1]):
        scores = x[f"fc{i}"].transpose(-1, 1)
        loss += F.cross_entropy(scores, y[:, i, :], ignore_index=ignore_index, reduction="mean")
    return loss

def smoothing_temporal_softmax_loss(x, y, ignore_index, epsilon=0.1):
    """Smoothing temporal softmax loss"""
    num_classes = x.shape[1]
    log_preds = F.log_softmax(x, dim=1)
    loss = -log_preds.sum(dim=1).mean() / num_classes
    nll = F.nll_loss(log_preds, y, ignore_index=ignore_index, reduction="mean")
    return (epsilon * loss) + (1 - epsilon) * nll

def multi_caption_smoothing_temporal_softmax_loss(x, y, ignore_index, epsilon=0.1):
    """Multi-caption smoothing temporal softmax loss"""
    loss = torch.zeros(1, device=x["fc0"].device)
    for i in range(y.shape[1]):
        scores = x[f"fc{i}"].transpose(-1, 1)
        loss += smoothing_temporal_softmax_loss(scores, y[:, i, :], ignore_index=ignore_index, epsilon=epsilon)
    return loss

@dataclass
class Candidate:
    words_so_far: list
    yn: torch.Tensor  # (wordvec_dim,)
    hn: Union[torch.Tensor, None]  # (hidden_dim,)
    cn: Union[torch.Tensor, None]  # (hidden_dim,)
    states: Union[tuple, None]
    attn_weights: Union[torch.Tensor, None] = None

@dataclass
class PQCandidate:
    priority: float = 0.0
    candidate: Candidate = Any

    def __lt__(self, other):
        return self.priority < other.priority

def batch_beam_search(rnn_captioner, yns, hns, cns, states, features, max_length, 
                      which_rnn=0, alpha=0.7, beam_width=10, need_weights=False):
    """Beam search for batch processing"""
    batch_size = yns.shape[0]
    captions = torch.full((batch_size, max_length + 1), rnn_captioner._pad, device=yns.device, dtype=torch.long)
    captions[:, 0] = rnn_captioner._start

    if need_weights:
        pixels = features.shape[-1]
        batch_attn_weights = torch.empty((batch_size, max_length, pixels, pixels), device=features.device)

    for i in range(batch_size):
        yn = yns[i:i+1, :, :]
        hn = None if hns is None else hns[:, i:i+1, :]
        cn = None if cns is None else cns[:, i:i+1, :]
        state = None if states is None else (states[0][:, i:i+1, :], states[1][:, i:i+1, :])
        feature = None if features is None else features[i:i+1, :, :, :]
        attn_weights = None if not need_weights else torch.empty(max_length, features.shape[-1], features.shape[-1])
        
        best = single_beam_search(
            [PQCandidate(candidate=Candidate([], yn, hn, cn, state, attn_weights))],
            rnn_captioner=rnn_captioner,
            features=feature,
            num_words_left=max_length,
            which_rnn=which_rnn,
            alpha=alpha,
            beam_width=beam_width,
            need_weights=need_weights
        )
        captions[i, 1:] = torch.tensor(best.candidate.words_so_far, dtype=captions.dtype, device=captions.device)
        if need_weights:
            batch_attn_weights[i, :, :] = best.candidate.attn_weights
            
    if need_weights:
        return captions, batch_attn_weights
    return captions

def single_beam_search(candidates, rnn_captioner, features, num_words_left, 
                       which_rnn=0, alpha=0.7, beam_width=10, need_weights=False):
    """Single beam search implementation"""
    if num_words_left == 0:
        return next(candidates)
    to_consider = []
    for candidate in candidates:
        to_consider = heapq.merge(
            to_consider,
            get_new_candidates(
                candidate,
                rnn_captioner,
                features,
                which_rnn,
                beam_width,
                alpha,
                need_weights,
                num_words_left,
            ),
        )
    candidates = islice(to_consider, beam_width)
    return single_beam_search(
        candidates=candidates,
        rnn_captioner=rnn_captioner,
        features=features,
        num_words_left=(num_words_left - 1),
        which_rnn=which_rnn,
        alpha=alpha,
        beam_width=beam_width,
        need_weights=need_weights,
    )

def get_new_candidates(pqcandidate, rnn_captioner, features=None, which_rnn=0, 
                      num_new=1, alpha=0.7, need_weights=False, num_words_left=0):
    """Generate new candidates for beam search"""
    candidate = pqcandidate.candidate
    old_priority = pqcandidate.priority
    
    if rnn_captioner.rnn_type in ("rnn", "gru"):
        output, hn = getattr(rnn_captioner.decoder, f"rnn{which_rnn}")(candidate.yn, candidate.hn.contiguous())
        cn, states = None, None
    elif rnn_captioner.rnn_type == "lstm":
        output, states = getattr(rnn_captioner.decoder, f"rnn{which_rnn}")(
            candidate.yn, (candidate.states[0].contiguous(), candidate.states[1].contiguous()))
        hn, cn = None, None
    
    scores = getattr(rnn_captioner.fc_scorer, f"fc{which_rnn}")(output)
    topk_scores, idxs = scores.topk(k=num_new, dim=2)  # (1, 1, num_new)
    topk_scores = F.log_softmax(topk_scores, dim=2).squeeze()  # (num_new,)
    yns = rnn_captioner.word_embedder(idxs).squeeze()  # (num_new, wordvec_dim)
    idxs = idxs.squeeze()  # (num_new,)

    if num_new == 1:
        topk_scores = topk_scores.unsqueeze(0)
        idxs = idxs.unsqueeze(0)
        yns = yns.unsqueeze(0)

    ret = []
    norm_factor = 1.0 / (len(candidate.words_so_far) + 1e-7) * alpha
    for i in range(idxs.shape[0]):
        if need_weights:
            attn_weights = candidate.attn_weights.clone()
        else:
            attn_weights = None
            
        priority = (old_priority + (-1 * topk_scores[i])) / norm_factor
        heapq.heappush(
            ret,
            PQCandidate(
                priority=priority,
                candidate=Candidate(
                    words_so_far=(candidate.words_so_far + [idxs[i]]),
                    yn=yns[i].view(1, 1, -1),
                    hn=hn,
                    cn=cn,
                    states=states,
                    attn_weights=attn_weights,
                ),
            ),
        )
    return ret

class CorpusBleu(Metric):
    """BLEU score metric for evaluation"""
    def __init__(self, tokenizer, weights=(0.25, 0.25, 0.25, 0.25), dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("total_score", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")
        self.tokenizer = tokenizer
        self.weights = weights

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.dim() == 2
        assert target.dim() == 3
        assert preds.shape[0] == target.shape[0]

        preds = [s.strip().split(" ") for s in ids_to_captions(preds, self.tokenizer, True)]
        target = [[s.strip().split(" ") for s in ids_to_captions(lst, self.tokenizer, True)] for lst in target]
        
        new_preds, new_target = [], []
        num_too_short = 0
        for i, pred in enumerate(preds):
            if len(pred) < 2:
                num_too_short += 1
            else:
                new_preds.append(pred)
                new_target.append(target[i])
                
        score = corpus_bleu(new_target, new_preds, weights=self.weights, smoothing_function=smoothie)
        self.total_score += (score * (len(preds) - num_too_short)) / len(preds)
        self.n += 1

    def compute(self):
        return self.total_score / self.n.float()

class 	CaptioningRNN(pl.LightningModule):
    """Main image captioning model"""
    def __init__(self, datamodule, config=None):
        super().__init__()
        self.current_session_epochs = 0

        if config is None:
            config = self.default_config()

        self.output_dir = config.get("output_dir", "output")
        self.batch_size = config["batch_size"]
        self.datamodule = datamodule
        self.tokenizer = self.datamodule.tokenizer
        self.max_length = config["max_length"]

        self.train_loss = []
        self.val_loss = []
        self.test_loss = []
        self.train_losses = []  # per-epoch training losses
        self.val_losses = []    # per-epoch validation losses
        #self.current_epoch = 0

        self.val_bleu = CorpusBleu(self.tokenizer)
        self.test_bleu = CorpusBleu(self.tokenizer)
        self.rouge = ROUGEScore()
        self.chrf = CHRFScore()
        #self.cider = CIDErScore()
        # Initialize CIDEr if available

        # Add these for metric tracking
        self.train_metrics = []  # Store training metrics per epoch
        self.val_metrics = []    # Store validation metrics per epoch
        self.test_metrics = None # Store test metrics

        # Initialize metric trackers
        self.train_bleu = CorpusBleu(self.tokenizer)
        self.val_bleu = CorpusBleu(self.tokenizer)
        self.test_bleu = CorpusBleu(self.tokenizer)

        # CHRF metric
        self.train_chrf = CHRFScore()
        self.val_chrf = CHRFScore()

        if HAVE_CIDER:
            self.cider = Cider()
        else:
            self.cider = None

        if config["rnn_type"] not in ("rnn", "lstm"):
            raise ValueError(f"RNN type {config['rnn_type']} not implemented")

        self.rnn_type = config["rnn_type"]
        self.image_extractor = ImageFeatureExtractor(
            encoder=config["image_encoder"], 
            projection_out=config["hidden_size"],
            #projection_in=2048 if "resnext" in config["image_encoder"] else None,
            freeze_weights=True
        )
        if config["encoder_init"]:
            self.image_extractor.init_weights(config["encoder_init"])

        self.word_embedder = WordEmbedder(config["wordvec_dim"], self.tokenizer)
        if config["wd_embedder_init"]:
            self.word_embedder.init_weights(config["wd_embedder_init"])

        self.vocab_size = self.word_embedder.vocab_size
        self.wordvec_dim = self.word_embedder.wordvec_dim
        #self._pad = self.tokenizer.padding["pad_id"]
        self._pad = self.tokenizer.padding_config["pad_id"]  # Directly access the config
        self._start = tokens_to_ids(self.tokenizer, [BOS])[BOS]
        self._end = tokens_to_ids(self.tokenizer, [EOS])[EOS]
        self.ignore_index = self._pad

        self.num_rnn_layers = config["num_rnn_layers"]
        self.num_rnn_directions = 2 if config["rnn_bidirectional"] else 1
        self.rnn_dropout = config["rnn_dropout"] if config["rnn_dropout"] else 0
        self.learning_rate = config["learning_rate"]

        # Initialize decoder
        if config["rnn_type"] == "rnn":
            self.decoder = RNN(
                input_size=self.wordvec_dim,
                hidden_size=config["hidden_size"],
                num_rnns=config["num_rnns"],
                num_layers=config["num_rnn_layers"],
                nonlinearity=config["rnn_nonlinearity"],
                dropout=self.rnn_dropout,
                bidirectional=config["rnn_bidirectional"],
            )
        elif config["rnn_type"] == "lstm":
            self.decoder = LSTM(
                input_size=self.wordvec_dim,
                hidden_size=config["hidden_size"],
                num_rnns=config["num_rnns"],
                num_layers=config["num_rnn_layers"],
                nonlinearity=config["rnn_nonlinearity"],
                dropout=self.rnn_dropout,
                bidirectional=config["rnn_bidirectional"],
            )
            
        if config["rnn_init"]:
            self.decoder.init_weights(config["rnn_init"])

        self.fc_scorer = ParallelFCScorer(
            config["num_rnns"], config["hidden_size"], self.vocab_size
        )
        if config["fc_init"]:
            self.fc_scorer.init_weights(config["fc_init"])

        self.inference_beam_alpha = config["inference_beam_alpha"]
        self.inference_beam_width = config["inference_beam_width"]
        self.label_smoothing_epsilon = config["label_smoothing_epsilon"]

        self.optimizer = config["optimizer"]
        self.scheduler = config["scheduler"]
        self.momentum = config["momentum"]

        self.save_hyperparameters(config)

    def train_dataloader(self):
        return self.datamodule.train_dataloader(self.batch_size)

    def val_dataloader(self):
        return self.datamodule.val_dataloader(self.batch_size)

    def test_dataloader(self):
        return self.datamodule.test_dataloader(self.batch_size)

    def forward(self, batch, n_captions=1):
        """Inference forward pass"""
        if n_captions > self.decoder.num_rnns:
            raise ValueError("Cannot generate more captions than trained rnns")
            
        x = batch["image"]
        batch_size = x.shape[0]
        x = self.image_extractor.encoder(x)
        # Debug print dimensions
        print(f"Encoder output features before pooling: {x.shape}")

        if self.image_extractor.pooling:
            x = self.image_extractor.pooling(x)

        print(f"Encoder output features after pooling: {x.shape}")
        print(f"Projection input dimension: {self.image_extractor.projection_in}")
        print(f"Projection output dimension: {self.image_extractor.projection_out}")

        x = F.normalize(x, p=2, dim=1)
        if self.image_extractor.projector:
            x = x.view(batch_size, -1)
            x = self.image_extractor.projector(x)
            
        captions = torch.empty((batch_size, n_captions, self.max_length + 1), device=x.device, dtype=torch.long)
        
        y = torch.tensor([self._start] * batch_size, device=x.device).view(batch_size, -1)
        y = self.word_embedder(y)

        cn, states = None, None

        # Build predictions network-by-network
        for i in range(captions.shape[1]):
            yn = y
            if self.rnn_type == "rnn":
                hn = x.unsqueeze(0).repeat(self.num_rnn_layers * self.num_rnn_directions, 1, 1)
            elif self.rnn_type == "lstm":
                hn = x.unsqueeze(0).repeat(self.num_rnn_layers * self.num_rnn_directions, 1, 1)
                states = (hn, torch.zeros_like(hn))
                
            captions[:, i, :] = batch_beam_search(
                rnn_captioner=self,
                yns=yn,
                hns=hn,
                cns=cn,
                states=states,
                features=None,
                max_length=self.max_length,
                which_rnn=i,
                alpha=self.inference_beam_alpha,
                beam_width=self.inference_beam_width,
            )
        return captions

    def forward_step(self, batch, batch_idx):
        """Training forward pass"""
        x, y = batch["image"], batch["captions"]

        # Extract image features
        x = self.image_extractor.encoder(x)
        
        # Handle different feature shapes
        if x.dim() == 4:  # For CNN features (B, C, H, W)
            if self.image_extractor.pooling:
                x = self.image_extractor.pooling(x)
            x = x.flatten(start_dim=1)
        
        # Project to hidden_size if needed
        if self.image_extractor.projector:
            x = self.image_extractor.projector(x)
        x = F.normalize(x, p=2, dim=1)

        # Prepare initial hidden state
        x = x.unsqueeze(0)  # Add sequence dimension
        x = x.repeat(self.num_rnn_layers * self.num_rnn_directions, 1, 1)

        # Offset captions for teacher forcing
        y_in, y_out = y[:, :, :-1], y[:, :, 1:]

        # Get input caption features
        y_in = self.word_embedder(y_in)

        # Run through decoder
        if self.rnn_type == "rnn":
            rnn_outs = self.decoder(y_in, x)
        elif self.rnn_type == "lstm":
            c0 = torch.zeros_like(x)
            rnn_outs = self.decoder(y_in, (x, c0))
            
        scores = self.fc_scorer(rnn_outs)
        y_out = y_out[:, :self.decoder.num_rnns, :]

        loss = multi_caption_smoothing_temporal_softmax_loss(
            scores,
            y_out,
            ignore_index=self.ignore_index,
            epsilon=self.label_smoothing_epsilon,
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.forward_step(batch, batch_idx)
        self.test_loss.append(loss.item())
        self.log("test_loss", loss)
        
        with torch.no_grad():
            preds = self.forward(batch, n_captions=self.decoder.num_rnns)
            
            if not hasattr(self, 'test_predictions'):
                self.test_predictions = []
                
            self.test_predictions.append({
                "image_paths": batch.get("path", ["unknown"] * len(batch["image"])),
                "predictions": preds.cpu().numpy(),
                "references": batch["captions"].cpu().numpy()
            })
        
        return None

    def training_step(self, batch, batch_idx):
        loss = self.forward_step(batch, batch_idx)
        self.train_loss.append(loss.item())
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        
        # Calculate metrics every 100 steps to save time
        if batch_idx % 100 == 0:
            with torch.no_grad():
                preds = self.forward(batch, n_captions=1)  # Only generate 1 caption
                self.train_bleu.update(preds[:, 0, :], batch["captions"])
                
                pred_texts = ids_to_captions(preds[:, 0, :], self.tokenizer, True)
                target_texts = [ids_to_captions(t, self.tokenizer, True) for t in batch["captions"]]
                
                # Calculate CHRF for each prediction-reference pair
                for pred, refs in zip(pred_texts, target_texts):
                    self.train_chrf.update(pred, ' '.join(refs))
        
        return loss

    def on_train_epoch_end(self):
        if not self.train_loss:
            return
            
        avg_train_loss = torch.mean(torch.tensor(self.train_loss))
        self.train_losses.append(avg_train_loss.item())
        
        # Get metrics
        train_metrics = {
            'loss': avg_train_loss.item(),
            'bleu': self.train_bleu.compute().item(),
            'chrf': self.train_chrf.compute().item()
        }
        self.train_metrics.append(train_metrics)
        
        # Reset metrics
        self.train_bleu.reset()
        self.train_chrf.reset()
        self.train_loss = []
        
        # Log metrics
        self.log_dict({
            "train_loss_epoch": avg_train_loss,
            "train_bleu": train_metrics['bleu'],
            "train_chrf": train_metrics['chrf']
        }, prog_bar=False)  # Don't clutter progress bar

    def validation_step(self, batch, batch_idx):
        loss = self.forward_step(batch, batch_idx)
        self.val_loss.append(loss.item())
        
        # Calculate validation metrics
        with torch.no_grad():
            preds = self.forward(batch, n_captions=1)
            self.val_bleu.update(preds[:, 0, :], batch["captions"])
            
            pred_texts = ids_to_captions(preds[:, 0, :], self.tokenizer, True)
            target_texts = [ids_to_captions(t, self.tokenizer, True) for t in batch["captions"]]
            
            for pred, refs in zip(pred_texts, target_texts):
                self.val_chrf.update(pred, ' '.join(refs))
        
        return loss

    def on_validation_epoch_end(self):
        if not self.val_loss:
            return
            
        avg_val_loss = torch.mean(torch.tensor(self.val_loss))
        self.val_losses.append(avg_val_loss.item())
        
        val_metrics = {
            'loss': avg_val_loss.item(),
            'bleu': self.val_bleu.compute().item(),
            'chrf': self.val_chrf.compute().item()
        }
        self.val_metrics.append(val_metrics)
        
        # Log metrics
        self.log_dict({
            "val_loss": avg_val_loss,
            "val_bleu": val_metrics['bleu'],
            "val_chrf": val_metrics['chrf']
        }, prog_bar=True)
        
        # Reset metrics
        self.val_bleu.reset()
        self.val_chrf.reset()
        self.val_loss = []

    def _plot_losses(self):
        """Simplified plotting with just loss and BLEU"""
        if len(self.train_losses) == 0 or len(self.val_losses) == 0:
            return
            
        # Plot losses
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        epochs = range(len(self.train_losses))
        plt.plot(epochs, self.train_losses, label="Train Loss")
        plt.plot(epochs, self.val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        
        # Plot BLEU
        plt.subplot(1, 2, 2)
        if len(self.train_metrics) == len(self.train_losses):
            plt.plot(epochs, [m['bleu'] for m in self.train_metrics], label="Train BLEU")
            plt.plot(epochs, [m['bleu'] for m in self.val_metrics], label="Val BLEU")
            plt.xlabel("Epoch")
            plt.ylabel("BLEU Score")
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        os.makedirs(self.output_dir, exist_ok=True)
        plt.savefig(os.path.join(self.output_dir, "training_metrics.png"))
        plt.close()

    def on_test_epoch_end(self):
        """Generate detailed test outputs with all metrics"""
        if not hasattr(self, 'test_predictions'):
            return
            
        all_results = []
        all_preds = []
        all_refs = []
        
        # Initialize metric objects
        bleu1 = CorpusBleu(self.tokenizer, weights=(1, 0, 0, 0))
        bleu2 = CorpusBleu(self.tokenizer, weights=(0.5, 0.5, 0, 0))
        bleu4 = CorpusBleu(self.tokenizer)
        rouge = ROUGEScore()
        chrf = CHRFScore()
        if HAVE_CIDER:
            cider = Cider()
        
        # Process each batch
        for batch in self.test_predictions:
            preds = torch.from_numpy(batch["predictions"])
            refs = torch.from_numpy(batch["references"])
            paths = batch["image_paths"]
            
            # Process each image in batch
            for i in range(len(paths)):
                img_preds = preds[i]  # All predicted captions
                img_refs = refs[i]    # All reference captions
                
                # Decode texts
                pred_texts = ids_to_captions(img_preds, self.tokenizer, True)
                ref_texts = [ids_to_captions(r.unsqueeze(0), self.tokenizer, True)[0] for r in img_refs]
                
                # Calculate metrics for first prediction
                bleu1.update(img_preds[0:1], img_refs.unsqueeze(0))
                bleu2.update(img_preds[0:1], img_refs.unsqueeze(0))
                bleu4.update(img_preds[0:1], img_refs.unsqueeze(0))
                rouge.update(pred_texts[0], ' '.join(ref_texts))
                chrf.update(pred_texts[0], ' '.join(ref_texts))
                
                # Semantic similarity
                sim = semantic_similarity(pred_texts[0], ref_texts)
                
                # Store results
                img_id = os.path.basename(paths[i]) if paths[i] != "unknown" else f"unknown_{i}"
                all_results.append({
                    "image_id": img_id,
                    "predicted": pred_texts,
                    "references": ref_texts,
                    "predictions_tensor": img_preds.cpu().numpy().tolist(),
                    "references_tensor": img_refs.cpu().numpy().tolist(),
                    "metrics": {
                        "BLEU-1": corpus_bleu([ref_texts], [pred_texts[0].split()], 
                                            weights=(1, 0, 0, 0), smoothing_function=smoothie),
                        "BLEU-2": corpus_bleu([ref_texts], [pred_texts[0].split()], 
                                            weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie),
                        "BLEU-4": corpus_bleu([ref_texts], [pred_texts[0].split()], 
                                            smoothing_function=smoothie),
                        "METEOR": meteor_score([r.split() for r in ref_texts], pred_texts[0].split()),
                        "ROUGE-L": rouge.compute()['rougeL_fmeasure'].item(),
                        "CHRF++": chrf.compute().item(),
                        "Semantic_Similarity": sim['similarity'],
                        "Keyword_Overlap": sim['keyword_overlap'],
                        "Composite_Score": sim['composite_score'],
                        "CIDEr": cider.compute_score({0: ref_texts}, {0: [pred_texts[0]]})[0] if HAVE_CIDER else 0.0
                    }
                })
                
                # Reset metrics for next image
                rouge.reset()
                chrf.reset()
        
        # Calculate overall metrics
        overall_metrics = {
            "BLEU-1": bleu1.compute().item(),
            "BLEU-2": bleu2.compute().item(),
            "BLEU-4": bleu4.compute().item(),
            "METEOR": np.mean([r['metrics']['METEOR'] for r in all_results]),
            "ROUGE-L": np.mean([r['metrics']['ROUGE-L'] for r in all_results]),
            "CHRF++": np.mean([r['metrics']['CHRF++'] for r in all_results]),
            "Semantic_Similarity": np.mean([r['metrics']['Semantic_Similarity'] for r in all_results]),
            "Keyword_Overlap": np.mean([r['metrics']['Keyword_Overlap'] for r in all_results]),
            "Composite_Score": np.mean([r['metrics']['Composite_Score'] for r in all_results]),
            "CIDEr": np.mean([r['metrics']['CIDEr'] for r in all_results]) if HAVE_CIDER else 0.0
        }
        
        # Save JSON output
        output_data = {
            "overall_metrics": overall_metrics,
            "per_image_results": all_results[:1000]  # Limit to first 1000 for file size
        }
        
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, "detailed_predictions.json"), "w") as f:
            json.dump(output_data, f, indent=2, default=lambda x: x.tolist() if isinstance(x, (np.ndarray, torch.Tensor)) else float(x) if isinstance(x, (np.generic, torch.Tensor)) else x)
        
        # Save human-readable version
        with open(os.path.join(self.output_dir, "predictions.txt"), "w") as f:
            # Write overall metrics
            f.write("OVERALL EVALUATION METRICS:\n")
            f.write("=" * 50 + "\n")
            for metric, score in overall_metrics.items():
                f.write(f"{metric}: {score:.4f}\n")
            f.write("\n\n")
            
            # Write individual results
            f.write("DETAILED PREDICTIONS:\n")
            f.write("=" * 50 + "\n\n")
            for result in all_results[:100]:  # Limit to first 100 for readability
                f.write(f"Image: {result['image_id']}\n")
                f.write("Predictions:\n")
                for j, pred in enumerate(result['predicted']):
                    f.write(f"  {j+1}. {pred}\n")
                f.write("References:\n")
                for j, ref in enumerate(result['references']):
                    f.write(f"  {j+1}. {ref}\n")
                
                # Write metrics
                f.write("\nMetrics for this image:\n")
                for metric, score in result['metrics'].items():
                    f.write(f"  {metric}: {score:.4f}\n")
                
                f.write("\n" + "=" * 80 + "\n\n")
        
        # Print summary
        print("\nTESTING COMPLETE - SUMMARY METRICS:")
        print("=" * 50)
        for metric, score in overall_metrics.items():
            print(f"{metric}: {score:.4f}")
        print("=" * 50)
        print(f"\nDetailed results saved to: {os.path.join(self.output_dir, 'detailed_predictions.json')}")
        print(f"Human-readable results saved to: {os.path.join(self.output_dir, 'predictions.txt')}")
        
        # Clear stored predictions
        del self.test_predictions

    def _plot_train_val_loss(self, filename="train_val_loss.png"):
        """Plot training and validation loss curves by epoch."""
        plt.figure(figsize=(12, 6))
        
        # Ensure equal length of data
        min_length = min(len(self.train_losses), len(self.val_losses))
        epochs = range(min_length)
        
        plt.plot(epochs, self.train_losses[:min_length], label="Training Loss", color='blue')
        plt.plot(epochs, self.val_losses[:min_length], label="Validation Loss", color='orange')
        
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        
        os.makedirs(self.output_dir, exist_ok=True)
        plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight')
        plt.close()

    def _plot_test_loss(self, filename="test_loss.png"):
        """Plot test loss by batch."""
        if not hasattr(self, 'test_loss') or not self.test_loss:
            return
            
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(self.test_loss)), self.test_loss, label="Test Loss", color='red')
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.title("Test Loss Progress")
        plt.legend()
        plt.grid(True)
        
        os.makedirs(self.output_dir, exist_ok=True)
        plt.savefig(os.path.join(self.output_dir, filename), bbox_inches='tight')
        plt.close()

    def on_test_end(self):
        """Called when the test ends."""
        if hasattr(self, 'test_loss') and self.test_loss:
            self._plot_test_loss()
        
        # Plot final training/validation curves if available
        if (hasattr(self, 'train_losses') and self.train_losses and 
            hasattr(self, 'val_losses') and self.val_losses):
            self._plot_train_val_loss()


    def _update_metrics(self, preds, targets, prefix):
        """Optimized metric update for validation"""
        if prefix == "val":
            self.val_bleu.update(preds, targets)
            
            pred_texts = ids_to_captions(preds, self.tokenizer, True)
            target_texts = [ids_to_captions(t, self.tokenizer, True) for t in targets]
            
            for pred, refs in zip(pred_texts, target_texts):
                self.chrf.update(pred, ' '.join(refs))

    # Update on_validation_epoch_end:
    def _save_metrics(self, phase):
        """Save simplified metrics to JSON file"""
        metrics = {
            "bleu": float(self.val_bleu.compute() if phase == "validation" else self.test_bleu.compute()),
            "chrf": float(self.chrf.compute()),
            "loss": float(torch.mean(torch.tensor(self.val_loss if phase == "validation" else self.test_loss))),
        }
    
        os.makedirs(self.output_dir, exist_ok=True)
        with open(os.path.join(self.output_dir, f"{phase}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    def configure_optimizers(self):
        if self.optimizer == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(), betas=(self.momentum, 0.999), lr=self.learning_rate
            )
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(), betas=(self.momentum, 0.999), lr=self.learning_rate
            )
        else:
            optimizer = torch.optim.SGD(
                self.parameters(), lr=self.learning_rate, momentum=self.momentum
            )
            
        if self.scheduler:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, patience=3, min_lr=1e-6
            )
            return {
                "optimizer": optimizer,
                "scheduler": scheduler,
                "monitor": "val_loss",
            }
        else:
            return {"optimizer": optimizer}

    @classmethod
    def default_config(cls):
        return {
            "max_length": 25,
            "batch_size": 64,
            "wordvec_dim": 768,
            "hidden_size": 576,
            "wd_embedder_init": "xavier",
            "image_encoder": "resnext50_32x4d",
            "encoder_init": "xavier",
            "rnn_type": "lstm",
            "num_rnns": 1,
            "num_rnn_layers": 3,
            "rnn_nonlinearity": None,
            "rnn_init": None,
            "rnn_dropout": 0.1,
            "rnn_bidirectional": False,
            "fc_init": "xavier",
            "label_smoothing_epsilon": 0.05,
            "inference_beam_width": 10,
            "inference_beam_alpha": 0.9,
            "learning_rate": 3e-4,
            "optimizer": "adam",
            "scheduler": "plateau",
            "momentum": 0.9,
            "output_dir": "output",
        }

def semantic_similarity(predicted, references):
    """Simplified semantic similarity for non-English"""
    all_words = set(predicted.split())
    ref_words = set(' '.join(references).split())
    
    intersection = len(all_words & ref_words)
    union = len(all_words | ref_words)
    
    jaccard = intersection / union if union > 0 else 0
    
    return {
        'similarity': jaccard,
        'keyword_overlap': intersection / max(1, len(all_words)),
        'composite_score': jaccard
    }

def evaluate_captions(preds, gts, tokenizer):
    """Evaluate captions using multiple metrics (without METEOR)"""
    pred_texts = ids_to_captions(preds, tokenizer, skip_special_tokens=True)
    gt_texts = [ids_to_captions(gt, tokenizer, skip_special_tokens=True) for gt in gts]
    
    metrics = {
        'BLEU-1': 0.0,
        'BLEU-2': 0.0,
        'BLEU-4': 0.0,
        'ROUGE-L': 0.0,
        'CHRF++': 0.0,
        'Semantic_Similarity': 0.0,
        'Keyword_Overlap': 0.0,
        'Composite_Score': 0.0,
    }
    
    rouge = ROUGEScore()
    chrf = CHRFScore()
    
    for pred, refs in zip(pred_texts, gt_texts):
        metrics['BLEU-1'] += corpus_bleu([refs], [pred.split()], weights=(1, 0, 0, 0), smoothing_function=smoothie)
        metrics['BLEU-2'] += corpus_bleu([refs], [pred.split()], weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        metrics['BLEU-4'] += corpus_bleu([refs], [pred.split()], weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
        
        rouge.update(pred, ' '.join(refs))
        metrics['ROUGE-L'] += rouge.compute()['rougeL_fmeasure']
        
        chrf.update(pred, ' '.join(refs))
        metrics['CHRF++'] += chrf.compute()
        
        sim = semantic_similarity(pred, refs)
        metrics['Semantic_Similarity'] += sim['similarity']
        metrics['Keyword_Overlap'] += sim['keyword_overlap']
        metrics['Composite_Score'] += sim['composite_score']
    
    n = len(pred_texts)
    for k in metrics:
        metrics[k] /= n
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Image Captioning with Flickr30K Dataset")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing Flickr30K dataset')
    parser.add_argument('--caption_file', type=str, default='captions.txt', 
                       help='Text file with captions in format "image#num caption"')
    
    # Model arguments
    parser.add_argument('--model_type', choices=['rnn', 'lstm'], default='lstm',
                  help='Type of RNN to use (rnn or lstm)')
    parser.add_argument('--encoder', type=str, default='resnext50', help='Image encoder architecture')
    parser.add_argument('--hidden_size', type=int, default=1024, help='Size of hidden layers') # 576
    parser.add_argument('--wordvec_dim', type=int, default=512, help='Dimension of word embeddings') # 768
    parser.add_argument('--num_rnns', type=int, default=1, help='Number of parallel RNNs')
    parser.add_argument('--num_layers', type=int, default=3, help='Number of RNN layers')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate') # 0.1
    parser.add_argument('--bidirectional', action='store_true', help='Use bidirectional RNN')
    parser.add_argument('--optimizer', choices=['adam', 'adamw', 'sgd'], default='adamw',
                      help='Optimizer to use (adam, adamw, or sgd)')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--max_length', type=int, default=25, help='Maximum caption length')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for optimizer')
    parser.add_argument('--beam_width', type=int, default=10, help='Beam width for inference')
    parser.add_argument('--beam_alpha', type=float, default=0.9, help='Beam search length penalty')
    parser.add_argument('--label_smoothing', type=float, default=0.05, help='Label smoothing epsilon')
    
    # Runtime arguments
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loader workers')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory for outputs')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume training')
    parser.add_argument('--test_only', action='store_true', help='Only run testing')
    parser.add_argument('--eval_interval', type=int, default=1, help='Evaluation interval during training')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    data_dir = os.path.abspath(args.data_dir)  # Get absolute path
    caption_file = os.path.join(data_dir, args.caption_file)

    # Setup data module
    data_module = CombinedDataModule(
        flickr_txt=caption_file,
        flickr_dir=data_dir,
        batch_size=args.batch_size,
        val_size=1024,
        test_size=1024,
        num_workers=args.workers,
        max_caption_length=args.max_length,
    )
    data_module.setup()
    
    # Model configuration
    config = {
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "wordvec_dim": args.wordvec_dim,
        "hidden_size": args.hidden_size,
        "wd_embedder_init": "xavier",
        "image_encoder": args.encoder,
        "encoder_init": "xavier",
        "rnn_type": args.model_type,
        "num_rnns": args.num_rnns,
        "num_rnn_layers": args.num_layers,
        "rnn_nonlinearity": None,
        "rnn_init": None,
        "rnn_dropout": args.dropout,
        "rnn_bidirectional": args.bidirectional,
        "fc_init": "xavier",
        "label_smoothing_epsilon": args.label_smoothing,
        "inference_beam_width": args.beam_width,
        "inference_beam_alpha": args.beam_alpha,
        "learning_rate": args.lr,
        "optimizer": args.optimizer,
        "scheduler": "plateau",
        "momentum": args.momentum,
        "output_dir": args.output_dir,
    }
    
    # Initialize model
    if args.resume:
        model = CaptioningRNN.load_from_checkpoint(
            args.resume, 
            datamodule=data_module,
            config=config  # Pass the config when resuming
        )
    else:
        model = CaptioningRNN(data_module, config)

    # Setup trainer
    trainer = pl.Trainer(
        default_root_dir=args.output_dir,
        max_epochs=args.epochs,
        accelerator="auto",  # Automatically selects GPU if available
        devices=1 if args.gpu and torch.cuda.is_available() else "auto",
        check_val_every_n_epoch=args.eval_interval,
    )
    
    if not args.test_only:
        # Training
        print("Starting training...")
        trainer.fit(model)
        
        # Save final model
        trainer.save_checkpoint(os.path.join(args.output_dir, "final_model.ckpt"))
    
    print("Starting testing...")
    trainer.test(model)
    
    # Metrics are automatically logged by Lightning
    # print("\nTesting complete. Predictions saved to ./output/detailed_predictions.json")

if __name__ == "__main__":
    main()

