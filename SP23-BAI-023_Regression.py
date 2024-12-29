import torch
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, r2_score


data = fetch_california_housing()
X, y = data.data, data.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)



def relu(x):
    return torch.max(torch.tensor(0), x)


class ManualRegression(torch.nn.Module):
    def __init__(self):

        super(ManualRegression, self).__init__()
        self.weights = torch.randn(8, 1, requires_grad=True)
        self.bias = torch.randn(1, requires_grad=True)
        self.w1 = torch.randn(8, 10, requires_grad=True)
        self.b1 = torch.randn(10, requires_grad=True)
        self.w2 = torch.randn(10,5 , requires_grad=True)
        self.b2 = torch.randn(5, requires_grad=True)
        self.w_out = torch.randn(5, 1, requires_grad=True)
        self.b_out = torch.randn(1, requires_grad=True)

    def forward(self, x):
        x1 = torch.matmul(x, self.w1) + self.b1
        x2 = relu(x1)
        x3 = torch.matmul(x2, self.w2) + self.b2
        x4 = relu(x3)
        x5 = torch.matmul(x4, self.w_out) + self.b_out
        return x5

    def parameters(self):
        return [self.w1, self.b1, self.w2, self.b2, self.w_out, self.b_out]

class MSELoss:
    def __call__(self, y_pred, y_true):
        return torch.mean((y_pred - y_true) ** 2)


class SGD:
    def __init__(self, parameters, lr):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        with torch.no_grad():
            for param in self.parameters:
                if param.grad is not None:
                    param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()


class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def train_model(model, train_data, n_epochs, batch_size, loss_fn, optimizer):
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    losses = []
    val_losses = []

    for epoch in range(n_epochs):
        epoch_losses = []

        for x_batch, y_batch in train_loader:
            model.train()
            # Forward pass
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        train_loss = sum(epoch_losses) / len(epoch_losses)
        losses.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = loss_fn(val_outputs, y_test_tensor).item()
            val_losses.append(val_loss)

        print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return model, losses, val_losses

def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
       
        y_pred = model(X_test)
       
        mse = torch.mean((y_pred - y_test) ** 2).item()

        mae = mean_absolute_error(y_test.numpy(), y_pred.numpy())

        r2 = r2_score(y_test.numpy(), y_pred.numpy())

    return mse, mae, r2


model = ManualRegression()
loss_fn = MSELoss()
optimizer = SGD(model.parameters(), lr=0.01)


train_data = CustomDataset(X_train_tensor, y_train_tensor)

# Train the model
trained_model, training_losses, val_losses = train_model(
    model, train_data, n_epochs=100, batch_size=64, loss_fn=loss_fn, optimizer=optimizer
)

plt.figure(figsize=(10, 5))
plt.plot(training_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curves')
plt.legend()
plt.show()

# Model testing (on test data)
mse, mae, r2 = evaluate_model(trained_model, X_test_tensor, y_test_tensor)

print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R²: {r2:.4f}")

