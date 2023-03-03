from torch.utils.data import Dataset


class CustomDataset(Dataset):

    def __init__(self, x, y):

        """Constructor for a custom dataset. This object will serve as the data provider for the torch functions.

        Args:
            x (tensor): A torch tensor of the features values.
            y (tensor): A torch tensor of the target values.
        """

        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
