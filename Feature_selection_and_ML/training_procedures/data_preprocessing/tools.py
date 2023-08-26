from torch.utils.data import DataLoader
import torch
import numpy as np 

def get_target_list(loader: DataLoader, target_means=[0], target_stds=[1]):

    """Gets the list of targets of a dataloader.

    Args:
        loader (Dataloader): The pytorch_geometric dataloader.
        target_means (np.array): An array of target means.
        target_stds (np.array): An array of target stds.

    Returns:
        list: The list of targets.
    """

    targets = []
    for batch in loader:
        targets.extend(get_target_list_from_batch(batch, target_means=target_means, target_stds=target_stds))

    return targets


def get_target_list_from_batch(batch, target_means=[0], target_stds=[1]):

    """Gets the list of targets of a batch.

    Args:
        batch (Batch): The pytorch_geometric batch.
        target_means (np.array): An array of target means.
        target_stds (np.array): An array of target stds.

    Returns:
        list: The list of targets.
    """

    targets = (batch[1].view(-1).numpy() * target_stds + target_means).tolist()

    return targets
