import cv2
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPool2D
import matplotlib.pyplot as plt
import numpy as np
from keras import layers
from keras.regularizers import L2
from keras.preprocessing.image import ImageDataGenerator
xlpath="C:\\Users\\hp\\Downloads\\PCOSGen-train\\PCOSGen-train\\class_label.xlsx"
data=pd.read_excel(xlpath)

# working on images
image=data['imagePath'].tolist()
imgfolderpath="C:\\Users\\hp\\Downloads\\PCOSGen-train\\PCOSGen-train\\images"
images=[]
labels=[]
image_paths=[]
for index, row in data.iterrows():
    image_path = os.path.join(imgfolderpath, row['imagePath'])
    image_paths.append(image_path)
for image_path, label in zip(image_paths, data['Healthy']):
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        img=cv2.resize(img,(150,150))
        img=img.astype('float32')/255.0
        images.append(img)
        labels.append(label)
    else:
        print(f"Image not found: {image_path}")
images = np.array(images)
labels = np.array(labels)
labels = to_categorical(labels, num_classes=2)
print(images)
print(labels)
print(images[0].shape)
#X_train,X_test,y_train,y_test=train_test_split(images,labels,test_size=0.2)
X=images
y=labels
model=Sequential()
'''model.add(Conv2D(512,kernel_size=(10,10),input_shape=(300,300,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(5,5)))
model.add(Conv2D(256,kernel_size=(5,5),activation='relu'))
model.add(MaxPooling2D(pool_size=(5,5)))'''
model.add(Conv2D(32,kernel_size=(3,3),input_shape=(150,150,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64,kernel_size=(5,5),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=2))
model.add(layers.Flatten())
model.add(Dense(1024,activation='relu'))
#model.add(Dense(16,activation='relu'))
model.add(Dense(2,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
'''datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
datagen.fit(X)'''
history=model.fit(X,y,epochs=20,batch_size=16,validation_split=0.2)
model.save('model.h5')
#loss1,acc1=model.evaluate(X_train,y_train)
#print(acc1*100)
'''loss,accuracy=model.evaluate(X_test,y_test)
print(accuracy*100)'''
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy Curves')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()