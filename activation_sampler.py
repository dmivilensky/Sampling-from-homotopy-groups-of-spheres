import torch
from torch import nn
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import DataLoader
from dataset import NormalClosure


class NormalClosureModel(nn.Module):
    def __init__(self, kernel_size=3, conv_channels=1, 
                 hidden_size=4, generators_number=2, max_length=10):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=generators_number, out_channels=conv_channels, kernel_size=kernel_size).to(torch.double)
        self.linear1 = nn.Linear(conv_channels * (max_length - kernel_size + 1), hidden_size).to(torch.double)
        self.relu1 = nn.ReLU().to(torch.double)
        self.linear2 = nn.Linear(hidden_size, 1).to(torch.double)
        self.sigmoid = nn.Sigmoid().to(torch.double)
    
    def forward(self, word):
        out = self.conv(word)
        out = out.reshape(out.shape[0], -1)
        out = self.linear1(out)
        out = self.relu1(out)
        out = self.linear2(out)
        return self.sigmoid(out)


if __name__ == "__main__":
    generators_number = 2
    max_length = 10
    dataset_length = 1000
    batch = 100
    epochs = 50

    dataset = NormalClosure([[1]], dataset_length, generators_number=generators_number, max_length=max_length)
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True)
    eval_dataloader = DataLoader(dataset, batch_size=dataset_length)

    model = NormalClosureModel()
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    for _ in range(epochs):
        model.train()
        for word, indicator in dataloader:
            loss = criterion(model(word), indicator)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        estimations = []
        model.eval()
        word, indicator = next(iter(eval_dataloader))
        estimation = model(word)
        loss = criterion(estimation, indicator).item()
        print('loss:', round(loss, 4), end=' ')
        y_true = indicator.data
        y_pred = (estimation > 0.5).double().data
        print('f1', round(f1_score(y_true, y_pred), 4), end=' ')
        print()

print()
print('TN', 'FP')
print('FN', 'TP')
print(confusion_matrix(y_true, y_pred))
