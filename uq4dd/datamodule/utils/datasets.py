from torch.utils.data import Dataset


class DrugDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return row['Drug'], row['Operator'], row['Y']

