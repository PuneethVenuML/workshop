


"""Creating the Dataset
Importing the basic libraries. We will import additional libraries when required"""
import glob
import numpy as np
import pandas as pd
import os
import shutil 
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
%matplotlib inline



'We have 25003 images each of cats and dogs'
train_folder = 'Deep Learning using Transfer Learning/ResNet50/Data/train/'
files = glob.glob(train_folder+'*') 
 
cat_files = [fn for fn in files if 'cat' in fn] 
dog_files = [fn for fn in files if 'dog' in fn] 
len(cat_files), len(dog_files)"


"""
Train data set will have 1500 images each of cats and dogs, 
Test data set will have 500 images each of cats and dogs and 
Validation data set will also have 500 images each of cats and dogs
"""

cat_train = np.random.choice(cat_files, size=200, replace=False) 
dog_train = np.random.choice(dog_files, size=200, replace=False) 
cat_files = list(set(cat_files) - set(cat_train)) 
dog_files = list(set(dog_files) - set(dog_train)) 
 
cat_val = np.random.choice(cat_files, size=50, replace=False) 
dog_val = np.random.choice(dog_files, size=50, replace=False) 
cat_files = list(set(cat_files) - set(cat_val)) 
dog_files = list(set(dog_files) - set(dog_val)) 
 
cat_test = np.random.choice(cat_files, size=50, replace=False) 
dog_test = np.random.choice(dog_files, size=50, replace=False) 
 
print('Cat datasets:', cat_train.shape, cat_val.shape, cat_test.shape) 
print('Dog datasets:', dog_train.shape, dog_val.shape, dog_test.shape)



"The dimension of our image will be 300 by 300 pixel"

IMG_WIDTH=300
IMG_HEIGHT=300
IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)
train_files = glob.glob(train_folder+'*')
train_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in np.concatenate((dog_train,cat_train),axis=0)]
train_imgs = np.array(train_imgs)
train_labels = [fn.split('\\')[-1].split('.')[0].strip() for fn in np.concatenate((dog_train,cat_train),axis=0)]

validation_files = glob.glob(train_folder+'*')
validation_imgs = [img_to_array(load_img(img, target_size=IMG_DIM)) for img in np.concatenate((dog_val,cat_val),axis=0)]
validation_imgs = np.array(validation_imgs)
validation_labels = [fn.split('\\')[-1].split('.')[0].strip() for fn in np.concatenate((dog_val,cat_val),axis=0)]
print('Train dataset shape:', train_imgs.shape,'\n','Validation dataset shape:', validation_imgs.shape)



"""
Pixel values for images are between 0 and 255. 
Deep Neural networks work well with smaller input values. 
Scaling each image with values between 0 and 1."""


train_imgs_scaled = train_imgs.astype('float32') 
validation_imgs_scaled = validation_imgs.astype('float32') 
train_imgs_scaled /= 255 
validation_imgs_scaled /= 255 
 
'# visualize a sample image '
print(train_imgs[0].shape) 
array_to_img(train_imgs[0])

"""Encoding text category labels of Cats and Dogs"""

# encode text category labels 
from sklearn.preprocessing import LabelEncoder 
 
le = LabelEncoder() 
le.fit(train_labels) 
train_labels_enc = le.transform(train_labels) 
validation_labels_enc = le.transform(validation_labels) 
 
print(train_labels[1495:1505], train_labels_enc[1495:1505])

train_datagen = ImageDataGenerator(rescale=1./255, zoom_range=0.3, rotation_range=50, \
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,\
                                       horizontal_flip=True, fill_mode='nearest')
val_datagen = ImageDataGenerator(rescale=1./255)

"""
Let’s see how some of the augmented images looks like. 
We will take two sample images from our training dataset to illustrate the same. 
The first image is an image of a cat and the second image is of a dog"""

img_id = 99
cat_generator = train_datagen.flow(train_imgs[img_id:img_id+1], 
 train_labels[img_id:img_id+1], 
 batch_size=1) 
cat = [next(cat_generator) for i in range(0,5)] 
fig, ax = plt.subplots(1,5, figsize=(16, 6))
print('Labels:', [item[1][0] for item in cat]) 
l = [ax[i].imshow(cat[i][0][0]) for i in range(0,5)]


img_id = 50 
dog_generator = train_datagen.flow(train_imgs[img_id:img_id+1], 
 train_labels[img_id:img_id+1], 
 batch_size=1) 
dog = [next(dog_generator) for i in range(0,5)] 
fig, ax = plt.subplots(1,5, figsize=(15, 6)) 
print('Labels:', [item[1][0] for item in dog]) 
l = [ax[i].imshow(dog[i][0][0]) for i in range(0,5)]



"""For our test generator, we need to send the original test images to the model for evaluation. 
We just scale the image pixels between 0 and 1 and do not apply any transformations."""

"""We just apply image augmentation transformations only to our training set images and validation images"""

train_generator = train_datagen.flow(train_imgs, train_labels_enc,batch_size=30)
val_generator = val_datagen.flow(validation_imgs, validation_labels_enc, batch_size=30)





from keras.applications.resnet import ResNet50
from keras.models import Model
import keras
restnet = ResNet50(include_top=False, weights='imagenet', input_shape=(IMG_HEIGHT,IMG_WIDTH,3))
output = restnet.layers[-1].output
output = keras.layers.Flatten()(output)
restnet = Model(restnet.input, outputs=output)
for layer in restnet.layers:
    layer.trainable = False
restnet.summary()


"""We now create our model using Transfer Learning using Pre-trained ResNet50 by 
adding our own fully connected layer and the final classifier using sigmoid activation function."""
input_shape=(IMG_HEIGHT,IMG_WIDTH,3)
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, InputLayer
from keras.models import Sequential
from tensorflow.keras import optimizers

model = Sequential()
model.add(restnet)
model.add(Dense(512, activation='relu', input_dim=input_shape))
model.add(Dropout(0.3))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=2e-5),
              metrics=['accuracy'])
model.summary()



"we now run the model"

history = model.fit_generator(train_generator, 
                              steps_per_epoch=100, 
                              epochs=1,
                              validation_data=val_generator, 
                              validation_steps=50, 
                              verbose=1)
"Saving the trained weights"

model.save(train_folder+'cats_dogs_tlearn_img_aug_cnn_restnet50.h5')


restnet.trainable = True
set_trainable = False
for layer in restnet.layers:
    if layer.name in ['res5c_branch2b', 'res5c_branch2c', 'activation_97']:
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False
layers = [(layer, layer.name, layer.trainable) for layer in restnet.layers]
pd.DataFrame(layers, columns=['Layer Type', 'Layer Name', 'Layer Trainable'])


model_finetuned = Sequential()
model_finetuned.add(restnet)
model_finetuned.add(Dense(512, activation='relu', input_dim=input_shape))
model_finetuned.add(Dropout(0.3))
model_finetuned.add(Dense(512, activation='relu'))
model_finetuned.add(Dropout(0.3))
model_finetuned.add(Dense(1, activation='sigmoid'))
model_finetuned.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['accuracy'])
model_finetuned.summary()


"We finally run the model"

history_1 = model_finetuned.fit_generator(train_generator, 
                                  steps_per_epoch=100, 
                                  epochs=2,
                                  validation_data=val_generator, 
                                  validation_steps=100, 
                                  verbose=1)
"saving the weights of the fine-tuned model"
model.save(train_folder +'cats_dogs_tlearn_finetune_img_aug_restnet50.h5')
