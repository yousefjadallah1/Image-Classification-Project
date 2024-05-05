import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras import models
from keras import datasets
from tkinter import Tk, filedialog




(traning_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
traning_images, testing_images = traning_images/255, testing_images/255

class_names = ['Plane', 'Car' , 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# this is for traning the model

# model = models.Sequential()
# model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
# model.add(layers.MaxPooling2D(2,2))
# model.add(layers.Conv2D(64, (3,3), activation='relu'))
# model.add(layers.MaxPooling2D(2,2))
# model.add(layers.Conv2D(64, (3,3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(64, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(traning_images, training_labels,epochs=10, validation_data=(testing_images, testing_labels))

# loss,accuracy = model.evaluate(testing_images, testing_labels)
# print(f"Loss: {loss}")
# print(f"Accuracy: {accuracy}")
# model.save('image_classsifier.model.h5')


# now for testing the model
def preprocess_image(image_path):
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    resized_img = cv.resize(img, (32, 32))  
    resized_img = resized_img / 255.0 
    return resized_img

def choose_image():
    root = Tk()
    root.withdraw() 
    file_path = filedialog.askopenfilename() 
    return file_path

model = models.load_model('image_classsifier.model.h5')

image_path = choose_image()

if image_path:
    img = preprocess_image(image_path)
    plt.imshow(img, cmap=plt.cm.binary)
    plt.show() 
    
    prediction = model.predict(np.expand_dims(img, axis=0))
    index = np.argmax(prediction)
    print(f'Model Prediction is {class_names[index]}, hope it is right :)')
    print("The Accuracy for this model: 0.699400007724762")
    
else:
    print("No image selected!")

