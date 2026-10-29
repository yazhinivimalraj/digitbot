import os
import keras
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
model_DIR="saved_model"
model_Path = os.path.join(model_DIR,"mnist_mlp.keras")
(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()
x_train=x_train.astype("float32")/255.0
x_test=x_test.astype("float32")/255.0
model=keras.Sequential([layers.Input(shape=(28,28)),layers.Flatten(),layers.Dense(256,activation="relu"),layers.Dropout(0.2),layers.Dense(10,activation="softmax"),])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.summary()
model.fit(x_train,y_train,epochs=10,batch_size=128,validation_split=0.1,)
loss,acc=model.evaluate(x_test,y_test,verbose=0)
print(f"\n test accuracy:{acc:.4f}")
os.makedirs(model_DIR,exist_ok=True)
model.save(model_Path)
print(f"model saved to :{model_Path}")