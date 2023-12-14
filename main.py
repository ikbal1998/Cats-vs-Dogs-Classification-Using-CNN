import os
from random import random
from random import seed
import shutil
import sys
from matplotlib import pyplot
from keras.preprocessing import image
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.models import load_model
from keras.callbacks import CSVLogger

data_path = "Dataset_Dogs_Cats/"

train_path = os.path.join(data_path,"train_new/")
test_path  = os.path.join(data_path, 'test_new/')
labeldirs = ['dogs/', 'cats/']

for folder_path in [train_path, test_path]:
    for sub_path in labeldirs:
        new_folder = os.path.join(folder_path, sub_path)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)

img_list = [filename for filename in os.listdir("Dataset_Dogs_Cats/train/")]

seed(1)
# define ratio of pictures to use for validation
test_ratio = 0.25
# Copy image files to destination folders
for i, f in enumerate(img_list):
    if random() < test_ratio:
        if f.startswith('dog'):
            dest_folder = os.path.join(test_path,"dogs")
        else:
            dest_folder = os.path.join(test_path, "cats")

    else:
        if f.startswith('cat'):
            dest_folder = os.path.join(train_path,"cats")
        else:
            dest_folder = os.path.join(train_path, "dogs")
    shutil.copy(os.path.join("Dataset_Dogs_Cats/train/", f), os.path.join(dest_folder, f))


def baseline_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(200, 200, 3),
                     kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def final_model():
 # load  model
 model = VGG16(include_top=False, input_shape=(224, 224, 3))
 # mark loaded layers as not trainable
 for layer in model.layers:
    layer.trainable = False
 # add new classifier layers
 flat1 = Flatten()(model.layers[-1].output)
 class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
 output = Dense(1, activation='sigmoid')(class1)

 # define new custom model: vgg16's input layers + new classifier layer
 model = Model(inputs=model.inputs, outputs=output)
 # compile model
 model.compile(optimizer="adam", loss='binary_crossentropy', metrics=['accuracy'])
 return model

def plot_graph(history):
    #plot CrossEntropy loss for train and test
    pyplot.subplot(211)
    pyplot.title('CrossEntropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='red', label='test')
    #plot accuracy for train and test
    pyplot.subplot(212)
    pyplot.title('Model Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='red', label='test')
    #save plot to file
    pyplot.savefig('vgg16_accuracy_plot.png')
    pyplot.close()

def predict_trial():
    model = load_model("saved_models/vgg16_model.h5")
    batch_sz = 64
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    test_set = test_datagen.flow_from_directory('Dataset_Dogs_Cats/test_new/',
                                              class_mode='binary', batch_size=batch_sz, target_size=(224, 224))
    test = test_set.filenames
    fig=pyplot.figure(figsize=(15, 6))
    columns = 6
    rows = 3
    for i in range(columns*rows):
        fig.add_subplot(rows, columns, i+1)
        img1 = image.load_img('Dataset_Dogs_Cats/test_new/'+test_set.filenames[np.random.choice(range(6000))], target_size=(224, 224))
        img = image.img_to_array(img1)
        img = img/255
        img = np.expand_dims(img, axis=0)
        prediction = model.predict(img, batch_size=None,steps=1) #gives all class prob.
        if(prediction[:,:]>0.5):
            value ='Dog :%1.2f'%(prediction[0,0])
            pyplot.text(20, 58,value,color='red',fontsize=10,bbox=dict(facecolor='white',alpha=0.8))
        else:
            value ='Cat :%1.2f'%(1.0-prediction[0,0])
            pyplot.text(20, 58,value,color='red',fontsize=10,bbox=dict(facecolor='white',alpha=0.8))
        pyplot.imshow(img1)
        pyplot.savefig('predicted_images.png')

    return 0

def evaluate_model():
    #define my model and the batch_sz
    model = final_model()
    csv_logger = CSVLogger('vgg16_training_log.log', separator=',', append=False)
    batch_sz = 64

    #generate the data batches for training dataset with augmentation techniques
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0,
                                       shear_range=0.1,
                                       zoom_range=0.1,
                                       horizontal_flip=True
                                       )

    # generate the data batches for testing dataset
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    #read the images
    train_set = train_datagen.flow_from_directory('Dataset_Dogs_Cats/train_new/',
                                           class_mode='binary', batch_size=batch_sz, target_size=(224, 224))
    test_set = test_datagen.flow_from_directory('Dataset_Dogs_Cats/test_new/',
                                          class_mode='binary', batch_size=batch_sz, target_size=(224, 224))

    #model fiting
    history = model.fit_generator(train_set, steps_per_epoch=train_set.samples//batch_sz,
                                  validation_data=test_set, validation_steps=test_set.samples//batch_sz, epochs=10, verbose=1, callbacks=[csv_logger])
    #evaluate the model
    loss, accuracy = model.evaluate_generator(test_set, steps=len(test_set), verbose=0)
    model.save('baseline_model.h5')
    plot_graph(history)
    return accuracy, history


accuracy, loss = evaluate_model()
predict_trial()
print("The accuracy of the model:",round(accuracy * 100,3))






















