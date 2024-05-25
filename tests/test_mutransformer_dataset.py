# test_mutransformer_dataset.py

from mutransformer.dataset import prepare_data
import os


def test_prepare_data():
    # Initialize dataset
    trainloader, val_loader = prepare_data(batch_size=4)

    # Test __getitem__ functionality
    for i in range(len(trainloader)):
        sample = trainloader[i]


