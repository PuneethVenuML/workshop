
import numpy as np
from numpy.random import default_rng
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import random

os.chdir('/Spine Fracture Detection/')

def sphere(shape, radius, position):
    """Generate an n-dimensional spherical mask."""
    # assume shape and position have the same length and contain ints
    # the units are pixels / voxels (px for short)
    # radius is a int or float in px
    assert len(position) == len(shape)
    n = len(shape)
    semisizes = (radius,) * len(shape)

    # genereate the grid for the support points
    # centered at the position indicated by position
    grid = [slice(-x0, dim - x0) for x0, dim in zip(position, shape)]
    position = np.ogrid[grid]
    # calculate the distance of all points from `position` center
    # scaled by the radius
    arr = np.zeros(shape, dtype=float)
    for x_i, semisize in zip(position, semisizes):
        # this can be generalized for exponent != 2
        # in which case `(x_i / semisize)`
        # would become `np.abs(x_i / semisize)`
        arr += (x_i / semisize) ** 2

    # the inner part of the sphere will have distance below or equal to 1
    return arr <= 1.0


def get_sphere(w,h,d,counts):
    """
    """
    first = True
    for i in range(counts):
        # sp_arr = sphere((w,h,d),10,(w/2,h/2,d/2))
        x_pos = round((w/2)- random.uniform(5,10))
        y_pos = round((h/2)- random.uniform(5,10))
        z_pos = round((d/2)- random.uniform(5,10))
        sp_arr = sphere((w,h,d),\
            round(random.uniform(5, 15)),(x_pos,y_pos,z_pos))
        sp_arr = np.expand_dims(sp_arr,0)
        sp_arr = sp_arr.astype(int)
        sp_arr = normalize(sp_arr)
        
        if first:
            sphere_out = sp_arr
            first=False
        else:
            sphere_out = np.concatenate((sphere_out,sp_arr),0)
    print(sphere_out.shape)
    
    return sphere_out


def get_plot(arr):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    verts, faces, normals, values = measure.marching_cubes(arr, 0.5)
    ax.plot_trisurf(
        verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap='Spectral',
        antialiased=False, linewidth=0.0)
    plt.show()


def get_cube(w,h,d,fraction,shape):
    """
    This cube returns either a cube;
    if concentric =True, cube will have a concentric within
    if concentric =False, cube will have uniform shade
    """
    count = 0
    for i in range(d):
        
        #Random dots on the plane(slice)
        slice = default_rng(42).random((w,h))
        
        if shape=='square':
            if (count>(d*0.2)) & (count<(d*0.8)):
                # fraction = random.random()
                slice[int(w*fraction):int(-(w*fraction)),int(h*fraction):int(-(h*fraction))] = 250
        
        if shape=='cylinder':
            if (count>(d*0.2)) & (count<(d*0.8)):
                cx=w/2;  cy=h/2; r=w/4
                x=np.arange(w); y=np.arange(h)
                mask = (x[np.newaxis,:]-cx)**2 + (y[:,np.newaxis]-cy)**2 < r**2
                slice[mask] = 250
                
        slice = np.expand_dims(slice,0)
        if count ==0:
            slice_out = slice
        else:
            slice_out = np.concatenate((slice_out,slice),axis=0)
        count +=1
    slice_out = np.swapaxes(slice_out,0,2)
    
    return slice_out

def normalize(volume):
    """Normalize the volume:"""
    min = volume.min()
    max = volume.max()
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    
    return volume


def get_training_cubes(w,h,d,counts,shape):
    """
    returns total volume for training, nd.array of shape (x,x,x,x)
    """
    first = True
    for i in range(counts):
        fraction = round(random.uniform(0.2, 0.5),2)
        cube = get_cube(w,h,d,fraction,shape)
        cube = normalize(cube)
        cube = np.expand_dims(cube,0)
        if first:
            cube_out = cube
            first=False
        else:
            cube_out = np.concatenate((cube_out,cube),0)
    print(cube_out.shape)
    
    return cube_out


print("Train Data")
w=128; h=128; d=64;

counts = 40
training_data_sph = get_sphere(w,h,d,counts)
labels_tr_sph = [0]*counts

counts=40
training_data_sq_cube = get_training_cubes(w,h,d,counts,shape='square')
labels_tr_sq = [1]*counts

counts=40
training_data_cylinder = get_training_cubes(w,h,d,counts,shape='cylinder')
labels_tr_cylinder = [2]*counts


print("Validation Data")
counts=10
validation_sph = get_sphere(w,h,d,counts)
labels_val_sph = [0]*counts

counts=10
validation_square = get_training_cubes(w,h,d,counts,shape='square')
labels_val_square = [1]*counts

counts=10
validation_data_cylinder = get_training_cubes(w,h,d,counts,shape='cylinder')
labels_val_cylinder = [2]*counts



"""
# Visualize the cubes how they look; Test to ensure they look as they supposed to look
"""

get_plot(training_data_sph[0])
get_plot(training_data_sq_cube[0])
get_plot(training_data_cylinder[0])


"""
Get Total Data
"""
train = np.concatenate((training_data_sph,training_data_sq_cube,training_data_cylinder),axis=0)
print('Total Train data',train.shape)

validation =  np.concatenate((validation_sph,validation_square,validation_data_cylinder),axis=0)
print('Total Validation data',validation.shape)

"""
Get Labels
"""
train_labels = np.asarray(labels_tr_sph+labels_tr_sq+labels_tr_cylinder)
print('Total train labels',train_labels.shape)

validation_labels =  np.asarray(labels_val_sph+labels_val_square+labels_val_cylinder)
print('Total validations labels',validation_labels.shape)


"""
Get Model
"""

def get_model(width=128, height=128, depth=64):

    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=32, kernel_size=3, activation="relu")(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Conv3D(filters=100, kernel_size=3, activation="relu")(x)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=200, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=3, activation="softmax")(x)

    # Define the model.
    model = keras.Model(inputs, outputs, name="3dcnn")
    return model


model = get_model(width=128, height=128, depth=64)

initial_learning_rate = 0.0001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
)
model.compile(
    loss="SparseCategoricalCrossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=lr_schedule),
    metrics=["acc"],
)

# Define callbacks.
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "3D CNN understanding using simple cubes of squares and randoms_v1.h5", save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(monitor="val_acc", patience=15)


model.fit(x=train,\
          y=train_labels,\
          epochs = 1,\
          verbose = 1,\
          validation_data=(validation,validation_labels),\
          callbacks=[checkpoint_cb, early_stopping_cb])



"""
Visualize 
"""

fig, ax = plt.subplots(1, 2, figsize=(20, 4))
ax = ax.ravel()

for i, metric in enumerate(["acc", "loss"]):
    ax[i].plot(model.history.history[metric])
    ax[i].plot(model.history.history["val_" + metric])
    ax[i].set_title("Model {}".format(metric))
    ax[i].set_xlabel("epochs")
    ax[i].set_ylabel(metric)
    ax[i].legend(["train", "val"])
    
    
"""
Load the saved model
"""

model = keras.models.load_model("3D CNN understanding using simple cubes of squares and randoms.h5")


print('Predict on Sphere Volumes')
counts=5
volume = get_sphere(w,h,d,counts)

for i in range(volume.shape[0]):
    res = model.predict(np.expand_dims(volume[i],0))
    print(i, res)

print('\n'*2)

print('Predict on random Volumes')
counts=5
volume = get_training_cubes(w,h,d,counts,shape='square')

for i in range(volume.shape[0]):
    res = model.predict(np.expand_dims(volume[i],0))
    print(i,res)

print('\n'*2)
    
print('Predict on cylinder Volumes')
counts=5
volume = get_training_cubes(w,h,d,counts,shape='cylinder')

for i in range(volume.shape[0]):
    res = model.predict(np.expand_dims(volume[i],0))
    print(i,res)
    
# (r-0,s-1,c-2)
