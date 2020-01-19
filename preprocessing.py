import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
import problem_unittests as tests


data_dir: str = 'processed_celeba_small/'


def get_dataloader(batch_size, image_size, data_dir='processed_celeba_small/'):
    """
    Batch the neural network data using DataLoader
    :param batch_size: The size of each batch; the number of images in a batch
    :param image_size: The square size of the image data (x, y)
    :param data_dir: Directory where image data is located
    :return: DataLoader with batched data
    """

    transform = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])

    training_set = datasets.ImageFolder(data_dir, transform=transform)

    training_loader = torch.utils.data.DataLoader(dataset=training_set, batch_size=batch_size, shuffle=True)

    return training_loader


# Define function hyper parameters
batch_size = 128
img_size = 32

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
# Call your function and get a dataloader
celeba_train_loader = get_dataloader(batch_size, img_size)


def scale(x, feature_range=(-1, 1)):
    """"
    Scale takes in an image x and returns that image, scaled
    with a feature_range of pixel values from -1 to 1.
    This function assumes that the input x is already scaled from 0-1.
    """
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x
    min, max = feature_range

    x = (max - min) * x + min

    return x
