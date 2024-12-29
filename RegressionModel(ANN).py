from google.colab import files
uploaded_data=files.upload()
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt

DataFrame=pd.read_excel('AirQualityUCI.xlsx')
DataFrame=DataFrame.drop(columns=['Date', 'Time'])
DataFrame.fillna(DataFrame.mean(), inplace=True)
X=DataFrame.loc[:, DataFrame.columns != 'NOx(GT)']
y=DataFrame['NOx(GT)']
y=abs(y)

X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.1,random_state=8)
y_train=y_train.to_numpy()
y_test=y_test.to_numpy()

standardSacler=StandardScaler()
X_train=standardSacler.fit_transform(X_train)
X_test=standardSacler.transform(X_test)
y_train = standardSacler.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test = standardSacler.transform(y_test.reshape(-1, 1)).flatten()

class AirQualityDataSet(Dataset):
    def __init__(self,X,y,transform=None):
        self.X=torch.tensor(X,dtype=torch.float32)
        self.y=torch.tensor(y,dtype=torch.float32)
        self.transform=transform

    def __getitem__(self,index):
       x = self.X[index]
       y = self.y[index]
       if self.transform:
           x=transform(x)
       return x,y

    def __len__(self):
        return len(self.X)

trainDataSet=AirQualityDataSet(X_train,y_train)
testDataSet=AirQualityDataSet(X_test,y_test)

trainDataLoader=DataLoader(trainDataSet,batch_size=15,shuffle=True)
testDataLoader=DataLoader(testDataSet,batch_size=15,shuffle=False)

class RegressionModel(nn.Module):
    def __init__(self):
         super(RegressionModel,self).__init__()
         self.W1 = torch.randn(12,128) * 0.01
         self.W1.requires_grad = True
         self.b1=torch.zeros(128, requires_grad=True)
         self.W2=torch.randn(128,64) * 0.01
         self.W2.requires_grad = True
         self.b2=torch.zeros(64,requires_grad=True)
         self.W3=torch.randn(64, 1) * 0.01
         self.W3.requires_grad = True
         self.b3=torch.zeros(1,requires_grad=True)

    def relu(self,x):
        return torch.max(x,torch.zeros_like(x))

    def forward(self,x):
        x=self.relu(torch.matmul(x,self.W1) + self.b1)
        x=self.relu(torch.matmul(x,self.W2) + self.b2)
        x=torch.matmul(x,self.W3) + self.b3
        return x

class SGD:
    def __init__(self,parameters,lr):
        self.parameters=parameters
        self.lr=lr

    def step(self):
        with torch.no_grad():
            for param in self.parameters:
                param-=self.lr*param.grad

    def zero_grad(self):
        for param in self.parameters:
            param.grad=None

def MSELoss(actual_y,predicted_y):
  error=actual_y-predicted_y
  squared_Error=(error)**2
  mse=torch.mean(squared_Error)
  return mse

R_model=RegressionModel()
loss_fn=MSELoss
optimizer=SGD(parameters=[R_model.W1,R_model.b1,R_model.W2,R_model.b2,R_model.W3,R_model.b3],lr=0.001)
n_epoch=50
batch_size=20

def trainRegressionModel(model,trainDataLoader,valDataLoader,batch_size,n_epoch,loss_fn,optimizer):
    train_losses=[]
    val_losses=[]
    for epoch in range(n_epoch):
        model.train()
        epoch_train_loss=0
        epoch_val_loss=0
        batch_num=0
        for XBatch, yBatch in trainDataLoader:
            predicted_y=model(XBatch)
            loss=loss_fn(predicted_y,yBatch.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss +=loss.item()
            batch_num +=1
        model.eval()
        with torch.no_grad():
            for XBatch,yBatch in valDataLoader:
                predicted_y=model(XBatch)
                loss=loss_fn(predicted_y,yBatch.unsqueeze(1))
                epoch_val_loss+=loss.item()

        avg_train_loss=epoch_train_loss/len(trainDataLoader)
        avg_val_loss=epoch_val_loss/len(valDataLoader)

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{n_epoch},Train Loss:{avg_train_loss:.2f},Val Loss:{avg_val_loss:.2f}")

    return model,train_losses,val_losses

model,train_losses,val_losses=trainRegressionModel(R_model,trainDataLoader,testDataLoader,batch_size,n_epoch,loss_fn,optimizer)

# Plot training vs validation loss
plt.figure(figsize=(7,5))
plt.plot(train_losses,label='Training Loss',color='Purple')
plt.plot(val_losses,label='Validation Loss',color='Blue')
plt.title("Training vs Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

torch.save(R_model.state_dict(),'RegressionModel.pt')
newRegressionModel=RegressionModel()
newRegressionModel.load_state_dict(torch.load('RegressionModel.pt'))
newRegressionModel.eval()

def EvaluateRegressionModel(model,testDataLoader):
    test_loss=0
    predictedvalues=[]
    truevalues=[]
    
    with torch.no_grad():
        for XBatch,yBatch in testDataLoader:
            predicted_y=model(XBatch)
            loss=MSELoss(predicted_y, yBatch.unsqueeze(1))
            test_loss+=loss.item()
            
            predictedvalues.extend(predicted_y.numpy())
            truevalues.extend(yBatch.numpy())
    predictedvalues=np.array(predictedvalues)
    truevalues=np.array(truevalues)
    meanSquaredError=mean_squared_error(truevalues,predictedvalues)
    meanAbsoluteError=mean_absolute_error(truevalues,predictedvalues)

    print(f"Mean Squared Error (MSE):{meanSquaredError:.2f}")
    print(f"Mean Absolute Error (MAE):{meanAbsoluteError:.2f}")
    return meanSquaredError,meanAbsoluteError
meanSquaredError,meanAbsoluteError=EvaluateRegressionModel(newRegressionModel,testDataLoader)