import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from os import path
import os
import timm
import urllib
import matplotlib.pyplot as plt
import torchshow as ts
from torchvision.datasets import ImageFolder
from typing import Any, Callable, cast, Dict, List, Optional, Tuple

mu = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

lbl_dict_names = dict(
    n01440764='tench',  # 0
    n02102040='English springer',  # 217
    n02979186='cassette player',  # 482
    n03000684='chain saw',  # 491
    n03028079='church',  # 497
    n03394916='French horn',  # 566
    n03417042='garbage truck',  # 569
    n03425413='gas pump',  # 571
    n03445777='golf ball',  # 574
    n03888257='parachute'  # 701
)

lbl_dict_ids = dict(
    n01440764=0,  # 0
    n02102040=217,  # 217
    n02979186=482,  # 482
    n03000684=491,  # 491
    n03028079=497,  # 497
    n03394916=566,  # 566
    n03417042=569,  # 569
    n03425413=571,  # 571
    n03445777=574,  # 574
    n03888257=701  # 701
)

lbl_dict = dict([(0, 0), (1, 217), (2, 482), (3, 491), (4, 497), (5, 566), (6, 569), (7, 571), (8, 574), (9, 701)])


class CustomImageFolder(ImageFolder):

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        #print(11111)
        return sample, target


def get_loaders(dir):
    #valdir = path.join(args.data_dir, 'val')
    valdir = 'C:/Users/furkan/Desktop/projects/combine-attack/data/imagenette2-320-tiny50/val'
    val_dataset = CustomImageFolder(dir,
                                       transforms.Compose([transforms.Resize(224),
                                                           transforms.CenterCrop(224),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize(mean=mu, std=std)
                                                           ]))
    val_dataset.__getitem__(1)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10, shuffle=False,
                                             num_workers=0, pin_memory=True)
    return val_loader


if __name__ == '__main__':

    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    model = model.cuda()
    model.eval()

    url, filename = (
    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
    urllib.request.urlretrieve(url, filename)
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]

    images = get_loaders()
    print(type(images))

    #for i, ((X, y), (x1, y1)) in enumerate(zip(images, images)):
    for i, (X, y) in enumerate(images):
        print(i)
        y = torch.ones(10) * lbl_dict[int(y[0])]
        y = y.type(torch.long)

        X, y = X.cuda(), y.cuda()

        with torch.no_grad():
            out = model(X)

        probabilities = torch.nn.functional.softmax(out[0], dim=0)
        print(probabilities.shape)

        #ts.show(X[0])
        top5_prob, top5_catid = torch.topk(probabilities, 5)
        for i in range(top5_prob.size(0)):
            print(categories[top5_catid[i]], top5_prob[i].item())
