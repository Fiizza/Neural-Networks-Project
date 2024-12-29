import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay




transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.5,), (0.5,))  
])


train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

def relu(x):
    return torch.max(torch.tensor(0), x)


class ManualClassification(torch.nn.Module):
    def __init__(self):
        super(ManualClassification, self).__init__()
        
        self.w1 = torch.randn(28*28, 64, requires_grad=True)
        self.b1 = torch.randn(64, requires_grad=True)
        self.w2 = torch.randn(64, 32, requires_grad=True)
        self.b2 = torch.randn(32, requires_grad=True)
        
        self.w_out = torch.randn(32, 10, requires_grad=True)  
        self.b_out = torch.randn(10, requires_grad=True)

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten input images
        x1 = torch.matmul(x, self.w1) + self.b1
        x2 = relu(x1)
        x3 = torch.matmul(x2, self.w2) + self.b2
        x4 = relu(x3)
        
        x7 = torch.matmul(x4, self.w_out) + self.b_out
        return x7

    def parameters(self):
        return [self.w1, self.b1, self.w2, self.b2,  self.w_out, self.b_out]
    

def train_model(model, train_data, val_data, n_epochs, batch_size, loss_fn, optimizer):
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=False)
    
    training_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(n_epochs):
        model.train()
        epoch_train_losses = []
        
        for x_batch, y_batch in train_loader:
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_train_losses.append(loss.item())
        
        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        training_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        epoch_val_losses = []
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                y_pred = model(x_batch)
                loss = loss_fn(y_pred, y_batch)
                
                epoch_val_losses.append(loss.item())
                
                _, predicted = torch.max(y_pred, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
        
        avg_val_loss = sum(epoch_val_losses) / len(epoch_val_losses)
        val_accuracy = correct / total
        
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        print("Epoch:", epoch + 1, "/", n_epochs)
        print("Train Loss:", round(avg_train_loss, 4))
        print("Val Loss:", round(avg_val_loss, 4))
        print("Val Accuracy:", round(val_accuracy * 100, 2), "%")
        print()  
    
    return model, training_losses, val_losses, val_accuracies


def evaluate_model(model, test_loader, loss_fn):
    model.eval()  
    y_true = []
    y_pred = []
    correct = 0
    total = 0
    test_losses = []  
    
    with torch.no_grad(): 
        for x_batch, y_batch in test_loader:
            y_true.extend(y_batch.numpy())  
            y_pred_batch = model(x_batch) 
            _, predicted = torch.max(y_pred_batch, 1)  
            
            \
            loss = loss_fn(y_pred_batch, y_batch)
            test_losses.append(loss.item())
            
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()  
            y_pred.extend(predicted.numpy())  

    test_loss = sum(test_losses) / len(test_losses)  
    test_accuracy = correct / total  
    
    cm = confusion_matrix(y_true, y_pred)

    # Generate classification report (includes precision, recall, F1-score)
    report = classification_report(y_true, y_pred)
    
    return test_loss, test_accuracy, cm, report

model = model = ManualClassification() 
loss_fn = nn.CrossEntropyLoss()  
optimizer = optim.Adam(model.parameters(), lr=0.005)


trained_model, training_losses, val_losses, val_accuracies = train_model(
    model, train_dataset, test_dataset, n_epochs=30, batch_size=32, loss_fn=loss_fn, optimizer=optimizer
)

# Plot learning curves
plt.figure(figsize=(10, 5))
plt.plot(training_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Learning Curves')
plt.legend()
plt.show()

# Evaluate the model on the test set
test_loss, test_accuracy, cm, classification_report_str = evaluate_model(trained_model, test_loader, loss_fn)

print("Test Loss:", round(test_loss, 4))
print("Test Accuracy:", round(test_accuracy * 100, 2), "%")
print("Classification Report:\n", classification_report_str)

# Plot confusion matrix
def plot_confusion_matrix(cm):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(10)])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix')
    plt.show()

plot_confusion_matrix(cm)

