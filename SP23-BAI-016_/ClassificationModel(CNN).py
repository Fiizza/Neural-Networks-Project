import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report,confusion_matrix,ConfusionMatrixDisplay
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,Dense,MaxPooling2D,Flatten,Dropout
from tensorflow.keras.optimizers import Adam

(X_train,y_train),(X_test,y_test)=cifar10.load_data()
X_train=X_train.astype('float32')/255.0
X_test=X_test.astype('float32')/255.0
y_train=to_categorical(y_train,10)
y_test=to_categorical(y_test,10)

def CNNmodel(convLayers,denseLayers,dropoutLayers,learningRate):
    model=Sequential()
    for filters in convLayers:
        model.add(Conv2D(filters,(3, 3),activation='relu',padding='same'))
        model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    for units in denseLayers:
        model.add(Dense(units,activation='relu'))
        model.add(Dropout(dropoutLayers))
    model.add(Dense(10,activation='softmax'))
    optimizer=Adam(learning_rate=learningRate)
    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    return model

Parameters = [
    {"convLayers":[32,64],"denseLayers":[128],"dropoutLayers":0.3, "learningRate":0.001},
    {"convLayers":[32,64,128],"denseLayers":[128,64],"dropoutLayers":0.5, "learningRate":0.001},
    {"convLayers":[64,128],"denseLayers":[256],"dropoutLayers":0.4, "learningRate":0.0005},]

modelNO=1
for param in Parameters:
    print(f"\nTraining Model{modelNO} with Parameters:{param}\n")
    model=CNNmodel(**param)
    training=model.fit(X_train,y_train,epochs=5,batch_size=64,validation_data=(X_test, y_test))
    testLoss,testAccuracy=model.evaluate(X_test,y_test)
    print(f"Model{modelNO} - Test Loss:{testLoss:.2f},Test Accuracy: {testAccuracy:.2f}\n")

    ypredictions=model.predict(X_test)
    y_predictedclasses=np.argmax(ypredictions,axis=1)
    y_trueclasses=np.argmax(y_test,axis=1)
    print(f"\nClassification Report for Model{modelNO}:\n")
    classReport=classification_report(y_trueclasses,y_predictedclasses)
    print(classReport)
    cm=confusion_matrix(y_trueclasses,y_predictedclasses)
    displaycm=ConfusionMatrixDisplay(confusion_matrix=cm)
    displaycm.plot()
    plt.title(f"Confusion Matrix for Model {modelNO}")
    plt.show()

    trainaccuracy=training.history['accuracy']
    valaccuracy=training.history['val_accuracy']
    trainloss=training.history['loss']
    valloss=training.history['val_loss']

    epochs=range(1,len(trainloss)+1)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    x=np.arange(len(epochs))
    width=0.2
    plt.bar(x-width/2,trainloss,width,label='Train Loss',color='purple')
    plt.bar(x+width/2,valloss,width,label='Validation Loss',color='skyblue')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.xticks(x,epochs)
    plt.legend()
   
    plt.subplot(1,2,2)
    plt.bar(x-width/2,trainaccuracy,width,label='Train Accuracy',color='green')
    plt.bar(x+width/2,valaccuracy,width,label='Validation Accuracy',color='orange')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xticks(x, epochs)
    plt.legend()
    plt.tight_layout()
    plt.show()
    modelNO=modelNO+1