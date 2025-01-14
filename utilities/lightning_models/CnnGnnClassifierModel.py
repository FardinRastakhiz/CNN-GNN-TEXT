import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np


class CnnGnnClassifierModel:
    def __init__(
        self,
        model,
        num_classes,
        optimizer=None,
        loss_func=None,
        learning_rate=0.01,
        batch_size=64,
        lr_scheduler=None,
        user_lr_scheduler=False,
        min_lr=0.0,
        device=None,
    ):
        self.model = model
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.min_lr = min_lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.optimizer = self._get_optimizer(optimizer)
        self.lr_scheduler = (
            self._get_lr_scheduler(lr_scheduler) if user_lr_scheduler else None
        )
        self.loss_func = loss_func
        self.train_losses = []
        self.val_losses = []
        self.train_acc = []
        self.val_acc = []

    def forward(self, x, *args, **kwargs):
        return self.model(x.x, torch.zeros((2, 0)), x.token_subsampling_probabilities, x.token_indices, x.token_sentiments, x.token_lengths, x.num_tokens, x.character_length, x.token_embeddings)

    def train_one_epoch(self, train_loader):
        self.model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            X, y = batch
            X, y = X.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()
            y_out = self.forward(X)
            loss = self.loss_func(y_out, y)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            preds = torch.argmax(y_out, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        epoch_acc = correct / total
        self.train_losses.append(epoch_loss / len(train_loader))
        self.train_acc.append(epoch_acc)

        return epoch_loss / len(train_loader), epoch_acc

    def validate_one_epoch(self, val_loader):
        self.model.eval()
        epoch_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                X, y = batch
                X, y = X.to(self.device), y.to(self.device)

                y_out = self.forward(X)
                loss = self.loss_func(y_out, y)

                epoch_loss += loss.item()
                preds = torch.argmax(y_out, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        epoch_acc = correct / total
        self.val_losses.append(epoch_loss / len(val_loader))
        self.val_acc.append(epoch_acc)

        return epoch_loss / len(val_loader), epoch_acc

    def fit(self, train_loader, val_loader, num_epochs):
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.validate_one_epoch(val_loader)

            if self.lr_scheduler:
                self.lr_scheduler.step(val_loss)

            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

    def _get_optimizer(self, optimizer):
        return (
            optimizer
            if optimizer is not None
            else torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        )

    def _get_lr_scheduler(self, lr_scheduler):
        return (
            lr_scheduler
            if lr_scheduler is not None
            else ReduceLROnPlateau(
                self.optimizer, patience=5, factor=0.5, mode="min", min_lr=self.min_lr
            )
        )


# Example Usage
if __name__ == "__main__":
    # Assume `MyModel` is your model class
    class MyModel(torch.nn.Module):
        def __init__(self, num_classes):
            super(MyModel, self).__init__()
            self.fc = torch.nn.Linear(10, num_classes)

        def forward(
            self,
            x,
            zeros,
            token_subsampling_probabilities,
            token_indices,
            token_sentiments,
            token_lengths,
            num_tokens,
            character_length,
            token_embeddings,
        ):
            return self.fc(x)

    model = MyModel(num_classes=3)
    loss_func = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    classifier = CnnGnnClassifierModel(
        model=model,
        num_classes=3,
        loss_func=loss_func,
        optimizer=optimizer,
        learning_rate=0.01,
        batch_size=32,
        user_lr_scheduler=True,
    )

    # Mock DataLoaders
    train_data = [(torch.rand(10), torch.tensor(0)) for _ in range(100)]
    val_data = [(torch.rand(10), torch.tensor(0)) for _ in range(30)]
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    classifier.fit(train_loader, val_loader, num_epochs=10)
