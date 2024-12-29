import torch
import torch.nn as NN
import torch.optim as optimizer
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay

TrainDataSet=datasets.FashionMNIST(root="",train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))]))
TestDataSet=datasets.FashionMNIST(root="",train=False,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))]))

print(TrainDataSet)
print(TestDataSet)
TrainDataLoader=DataLoader(TrainDataSet,batch_size=15,shuffle=True)
TestDataLoader=DataLoader(TestDataSet,batch_size=15,shuffle=False)

class ClassificationModel(NN.Module):
    def __init__(self,hidden1Neurons,hidden2Neurons,hidden3Neurons):
        super(ClassificationModel, self).__init__()
        self.InputLayer=NN.Linear(28 *28,hidden1Neurons)
        self.HiddenLayer1=NN.Linear(hidden1Neurons,hidden2Neurons)
        self.HiddenLayer2=NN.Linear(hidden2Neurons, hidden3Neurons)
        self.OutputLayer=NN.Linear(hidden3Neurons, 10)
        self.Flatten=NN.Flatten()

    def forward(self, x):
        x=self.Flatten(x)
        x=torch.relu(self.InputLayer(x))
        x=torch.relu(self.HiddenLayer1(x))
        x=torch.relu(self.HiddenLayer2(x))
        x=self.OutputLayer(x)
        return x

def trainClassificationModel(model,trainDataLoader,valLoader,n_epoch,loss_fn,optimizer):
    trainLosses=[]
    valLosses=[]
    trainaccuracies=[]
    valaccuracies=[]

    for epoch in range(n_epoch):
        model.train()
        epoch_train_loss=0
        correctvalues=0
        totalvalues = 0
        for XBatch,yBatch in trainDataLoader:
            optimizer.zero_grad()
            predictedy_=model(XBatch)
            loss=loss_fn(predictedy_,yBatch)
            loss.backward()
            optimizer.step()

            epoch_train_loss+=loss.item()
            _, predicted=torch.max(predictedy_.data,1)
            totalvalues+=yBatch.size(0)
            correctvalues+=(predicted==yBatch).sum().item()

        avgtrainloss=epoch_train_loss/len(trainDataLoader)
        trainaccuracy=correctvalues/ totalvalues *100
        trainLosses.append(avgtrainloss)
        trainaccuracies.append(trainaccuracy)

        model.eval()
        epoch_val_loss=0
        correct_val=0
        total_val=0
        with torch.no_grad():
            for XBatch,yBatch in valLoader:
                predictedy_=model(XBatch)
                loss=loss_fn(predictedy_,yBatch)
                epoch_val_loss+=loss.item()

                _, predicted=torch.max(predictedy_.data,1)
                total_val+=yBatch.size(0)
                correct_val+=(predicted==yBatch).sum().item()

        avgvalloss = epoch_val_loss / len(valLoader)
        val_accuracy = correct_val / total_val * 100
        valLosses.append(avgvalloss)
        valaccuracies.append(val_accuracy)

        print(f"Epoch {epoch + 1},Train Loss:{avgtrainloss:.3f},Val Loss:{avgvalloss:.3f}, "
              f"Train Acc:{trainaccuracy:.2f}%, Val Acc:{val_accuracy:.2f}%")

    return model,trainLosses,valLosses, trainaccuracies, valaccuracies

Parameters = [
    {"hidden1Neurons":128,"hidden2Neurons":64,"hidden3Neurons":32,"lr":0.001},
    {"hidden1Neurons":256,"hidden2Neurons":128, "hidden3Neurons":64,"lr":0.005},
    {"hidden1Neurons":64,"hidden2Neurons":32, "hidden3Neurons":16,"lr":0.0005},
]

modelNo=1

for Param in Parameters:
    print(f"\nTraining Model{modelNo} with Parameters:{Param}\n")
    model = ClassificationModel(Param["hidden1Neurons"],Param["hidden2Neurons"],Param["hidden3Neurons"])
    loss_fn=NN.CrossEntropyLoss()
    optimizer_=optimizer.Adam(model.parameters(),lr=Param["lr"])
    n_epoch=5
    trained_model,train_losses,val_losses,train_accuracies,val_accuracies=trainClassificationModel(model,TrainDataLoader,TestDataLoader,n_epoch,loss_fn,optimizer_)

    epochs=np.arange(1,len(train_losses)+1)
    plt.figure(figsize=(7,5))
    width=0.2
    plt.bar(epochs-width/2,train_losses,width=width,label='Training Loss',color='red',alpha=0.9)
    plt.bar(epochs+width/2,val_losses,width=width,label='Validation Loss',color='green',alpha=0.9)
    plt.title('Training vs. Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.show()
    plt.figure(figsize=(7,5))
    plt.bar(epochs - width/2,train_accuracies,width=width, label='Training Accuracy',color='skyblue',alpha=0.9)
    plt.bar(epochs + width/2,val_accuracies,width=width, label='Validation Accuracy',color='purple',alpha=0.9)
    plt.title('Training vs. Validation Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.xticks(epochs)
    plt.legend()
    plt.show()

    truevalues=[]
    predictedvalues=[]
    with torch.no_grad():
        for XBatch, yBatch in TestDataLoader:
            predictedy_=model(XBatch)
            _, predicted=torch.max(predictedy_.data,1)
            truevalues.extend(yBatch.numpy())
            predictedvalues.extend(predicted.numpy())

    print("\nClassification Report:\n")
    class_report=classification_report(truevalues,predictedvalues)
    print(class_report)
    confusionMatrix=confusion_matrix(truevalues,predictedvalues)
    displayCM=ConfusionMatrixDisplay(confusion_matrix=confusionMatrix)
    displayCM.plot()
    plt.title(f"Confusion Matrix for Model {modelNo}")
    plt.show()
    modelNo=modelNo+1
