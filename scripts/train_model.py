import argparse
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import os
import h5py
import random

class NarrativeUnitDataset(Dataset):
    def __init__(self, embedding_dir, transform=None, target_transform=None):
        self.embedding_dir = embedding_dir
        with h5py.File(self.embedding_dir, 'r') as hf:
            self.len = len(hf)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        with h5py.File(self.embedding_dir, 'r') as hf:
            group = hf[str(idx)]
            chapters = len(group)
        
        random_chapter_num = random.randrange(chapters-1)

        rand_chapter_embedding = group[str(random_chapter_num)]

        example_type = random.choice("Positive", "Negative")

        if example_type == "Positive":
            # TODO: write positive example code
        else:
            # TODO: write negative example code

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
    

#
# This script does *nothing* except print out its arguments and touch any files
# specified as outputs (thus fulfilling a build system's requirements for
# success).
#
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", dest="outputs", nargs="+", help="Output files")
    args, rest = parser.parse_known_args()

    print("Building files {} from arguments {}".format(args.outputs, rest))
    for fname in args.outputs:
        with open(fname, "wt") as ofd:
            pass
