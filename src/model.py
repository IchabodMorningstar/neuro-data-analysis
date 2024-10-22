import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import lightning as L
from torch import atan2, round, pi
# from features import position_to_x_and_y

class FNN(L.LightningModule):
    def __init__(self, input_size, hidden_sizes, output_size, **kwargs):
        super().__init__()
        layers = [nn.Linear(input_size, hidden_sizes[0]), nn.Tanh()]
        for h in range(len(hidden_sizes) - 1):
            layers += [nn.Linear(hidden_sizes[h], hidden_sizes[h + 1]), nn.Tanh()]
        layers += [
            nn.Linear(hidden_sizes[-1], output_size),
        ]
        self.layers = nn.Sequential(*layers)
        self.best_train_acc = None
        self.best_val_acc = None
        self.lr = kwargs.get("lr", 1e-3)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def get_loss(self, y_hat, y):
        pass

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.get_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)

        pp = self.get_predicted_positions(y_hat)
        acc = (pp == y[:, -1]).sum() / len(y)
        self.log("train_acc", acc)
        self.best_train_acc = acc if self.best_train_acc is None else max(self.best_train_acc, acc)
        self.log("best_train_acc", self.best_train_acc)
        return loss

    def get_predicted_positions(self, y_hat):
        pass

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.get_loss(y_hat, y)
        self.log("val_loss", loss)

        pp = self.get_predicted_positions(y_hat)
        acc = (pp == y[:, -1]).sum() / len(y)
        self.log("val_acc", acc)
        self.best_val_acc = acc if self.best_val_acc is None else max(self.best_val_acc, acc)
        self.log("hp_metric", self.best_val_acc)
        self.log("best_val_acc", self.best_val_acc)

        return pp

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        pp = self.get_predicted_positions(y_hat)
        acc = (pp == y[:, -1]).sum() / len(y)
        self.log("test_acc", acc)
        return acc

class CrossEntropyFNN(FNN):
    def __init__(self, input_size, hidden_size, output_size, **kwargs):
        super().__init__(input_size, hidden_size, output_size, **kwargs)
        self.save_hyperparameters()
        self.ce = nn.CrossEntropyLoss()

    # def __init__(self, input_size, hidden_size, output_size, NUM_SAMPLES):
    #     super().__init__(input_size, hidden_size, output_size)
    #     self.save_hyperparameters("input_size", "hidden_size", "NUM_SAMPLES")
    #     self.bce = nn.BCELoss()

    def forward(self, x):
        x = self.layers(x)
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        # x = F.softmax(x, dim=1)
            
        return x

    def get_loss(self, y_hat, y):
        # print(y_hat)
        # print(y_hat[0])
        # print(y)
        return self.ce(y_hat, y[:, :-1])

    def get_predicted_positions(self, y_hat):
        return torch.argmax(y_hat, dim=1)


# COORDS = torch.tensor(
#     [position_to_x_and_y(p) for p in range(1, 9)], dtype=torch.float32
# )

# class CartesianFNN(FNN):
#     def __init__(self, input_size, hidden_size, output_size, **kwargs):
#         super().__init__(input_size, hidden_size, output_size, **kwargs)
#         self.save_hyperparameters()

#     def setup(self, stage):
#         self.COORDS = COORDS.to(self.device)

#     def forward(self, x):
#         x = self.layers(x)
#         x = x @ self.COORDS[:, :2]
#         return x

#     def get_loss(self, y_hat, y):
#         return F.mse_loss(y_hat, y[:, :2])

#     def get_predicted_positions(self, y_hat):
#         return torch.tensor(
#             [
#                 (round(atan2(pr[1], pr[0]) % (2 * pi) / (pi / 4))) % 8 + 1
#                 for pr in y_hat
#             ],
#             device=self.device,
#         )


class DataModule(L.LightningDataModule):
    def __init__(self, train, val, test, train_batch_size=32, val_batch_size=32, one_hot_max=.6):
        super().__init__()

        self.train = train[torch.randperm(train.shape[0])]
        self.m = torch.mean(train[:, :-1], dim=0)
        self.s = torch.std(train[:, :-1], dim=0)

        self.val = val
        self.test = test
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.one_hot_max = one_hot_max

    def _create_dataset(self, dataset):
        return TensorDataset((dataset[:, :-1] - self.m) / self.s, 
                             self._create_y(dataset[:, -1], self._get_soft_one_hot))
    
    def _get_soft_one_hot(self, pos):
        return torch.tensor([self.one_hot_max if pos == x else ((1 - self.one_hot_max) / 2 if abs(x - pos) == 1 or abs(x - pos) == 7 else 0) 
                             for x in range(0, 8)])

    def _create_y(self, l, func):
        return torch.hstack([torch.stack([func(y) for y in l]), l.unsqueeze(1)])

    def train_dataloader(self): 
        return DataLoader(self._create_dataset(self.train), batch_size=self.train_batch_size, 
                          num_workers=7, persistent_workers=True)
        
    def val_dataloader(self):
        return DataLoader(self._create_dataset(self.val), batch_size=self.val_batch_size, 
                          num_workers=7, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self._create_dataset(self.test), num_workers=7, persistent_workers=True)