import torch 
from torch.utils.data import Dataset, DataLoader


class NinaProDataset(Dataset):

    def __init__(self, X, y):

        """
        X: numpy array of shape [n_windows, window_size, 12]
        y: numpy array of shapre [n_windows]

        """

        self.X = torch.tensor(X, dtype = torch.float32)  # shape: [n_windows, window_size, 12]
        self.y = torch.tensor(y, dtype = torch.long)     # shape: [n_windows]


    

    def __len__(self):
        return len(self.X)


    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dataloaders(X_train, y_train, X_test, y_test, batch_size):

    train_dataset = NinaProDataset(X_train, y_train)
    test_dataset = NinaProDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    print(f'Train batches: {len(train_loader)}')
    print(f'Test batches:  {len(test_loader)}')

    return train_loader, test_loader