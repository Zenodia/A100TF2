# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np

print(tf.__version__)

def load_data():
    Xtrain = np.load('Xtrain.npy')
    Ytrain = np.load('Ytrain.npy')
    Xtest = np.load('Xtest.npy')
    Ytest = np.load('Ytest.npy')
    return Xtrain, Ytrain, Xtest,Ytest
Xtrain,Ytrain,Xtest,Ytest=load_data()

print(np.min(Xtrain),np.max(Xtrain), np.min(Ytrain),np.max(Ytrain),Xtrain.shape,Ytrain.shape,Xtest.shape,Ytest.shape)
def normalize(x,y,maxi=255.):
    x=x/maxi
    y=y/maxi
    return x,y
Xtrain,Ytrain=normalize(Xtrain,Ytrain)
Xtest,Ytest=normalize(Xtest,Ytest)
print(np.min(Xtrain),np.max(Xtrain), np.min(Ytrain),np.max(Ytrain))
import tempfile
import os
import tensorflow as tf
from tensorflow import keras
## note that Conv2DTranspose is not yet supported for quantization, need to only annotate the encoding bit of Unet to get quantization applied layers working properly
from tensorflow.keras.layers import BatchNormalization , Dropout
from tensorflow.keras.optimizers import Adam , Nadam,Adamax
from tensorflow.keras import backend as K
from tensorflow.keras.losses import binary_crossentropy

print("tensorflow version ", tf.__version__)

smooth = 1.

smooth = 1.

sz=96
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

### customized unet loss function in order to ensure sucess and fast training 
def combined_dice_binary_loss(y_true,y_pred):
    def dice_loss(y_true,y_pred):
        numerator= 2 * tf.reduce_sum( y_true * y_pred, axis=(1,2,3))
        denominator = tf.reduce_sum(y_true +y_pred, axis=(1,2,3))
        return tf.reshape(1-numerator/denominator, (-1,1,1))
    return binary_crossentropy(y_true,y_pred)+dice_loss(y_true,y_pred)


# quantized awared for the moment only accept sequential model api,and not yet support UpSampling,
# so i had to customize the unet model accordingly, also only quantize the encoder part, skip compeltely the deconder 
def setup_unet():
    model=keras.Sequential([
    keras.layers.InputLayer(input_shape=(96, 96,1)),
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'), # conv1 32
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'), # conv1 32
    keras.layers.MaxPooling2D(pool_size=(2, 2)), # maxpool pool1
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'), # 53
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'), #64
    keras.layers.MaxPooling2D(pool_size=(2, 2)), # maxpool pool2
    keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'), # 128
    keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'), #128
    keras.layers.MaxPooling2D(pool_size=(2, 2)), # maxpool 3 
    keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),#256
    keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),#256
    keras.layers.MaxPooling2D(pool_size=(2, 2)), # maxpool pool 4
    keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),#512
    keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same'),#512 
    keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last"),
    keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
    keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last"),
    keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last"),
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last"),
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    keras.layers.Conv2D(1, (1, 1), activation='sigmoid') ])
    return model
base_model=setup_unet()
base_model.compile(optimizer='Adam', loss=[combined_dice_binary_loss], metrics=[dice_coef])

# fit model
base_model.fit(
  Xtrain,
  Ytrain,
  epochs=20,
validation_data=(Xtest,Ytest),
)
