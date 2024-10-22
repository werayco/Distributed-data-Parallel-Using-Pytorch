import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader

world_size = torch.cuda.device_count()
print(world_size)

loss = nn.MSELoss()

# Define our model
class SimpleModel(nn.Module):
    def __init__(self, drop_prop):
        super(SimpleModel, self).__init__()
        self.dense = nn.Linear(3, 8)
        self.dropOut1 = nn.Dropout(drop_prop)

        self.dense2 = nn.Linear(8, 16)
        self.dropOut2 = nn.Dropout(drop_prop)

        self.final = nn.Linear(16, 1)

    def forward(self, x):
        dense1 = self.dense(x)
        dense1 = torch.relu(dense1)
        dense1 = self.dropOut1(dense1)

        dense2 = self.dense2(dense1)
        dense2 = torch.relu(dense2)
        dense2 = self.dropOut2(dense2)

        final = self.final(dense2)
        final = torch.sigmoid(final)

        return final

# Define dataset classes
class featDataSet(Dataset):
    def __init__(self,row,col):
        self.data = torch.rand(row, col)  
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

class targetDataSet(Dataset):
    def __init__(self,row):
        self.data = torch.rand(row, 1)  # Adjust the size for a larger dataset
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

# Training function
def train(epochs: int, batch_size: int, device):
    # Instantiate the model
    model = SimpleModel(0.5)
    
    # Use DataParallel to split the work between two GPUs
    model = nn.DataParallel(model)
    model = model.to(device)
    
    # Define datasets and dataloader
    feat_dataset = featDataSet(3000,3)
    target_dataset = targetDataSet(3000)

    data_loader = DataLoader(feat_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    
    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        for data in data_loader:
            data = data.to(device)
            target = target_dataset[:len(data)].to(device)

            # Forward pass
            y_pred = model(data)

            # Compute loss
            losss = loss(y_pred, target)
            optimizer.zero_grad()
            losss.backward()
            optimizer.step()

        print(f'Epoch {epoch+1}/{epochs} completed.loss:{losss}')        


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 20
    batch_size = 8

    train(epochs, batch_size, device)
