import numpy as np
np.random.seed(865)

from keras.models import Model
from keras.layers import Input, BatchNormalization, merge, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import  Dropout, concatenate, Conv2DTranspose, Lambda, Reshape
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils.np_utils import to_categorical
from scipy.misc import imsave
from os import path, makedirs
import argparse
import keras.backend as K
import logging
import pickle
import tifffile as tiff
import os
import sys
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D
from keras.layers import Input, add, concatenate
from keras.models import Model
from keras.optimizers import RMSprop
from keras.losses import binary_crossentropy
from clr_callback import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def dice_coef(y_true, y_pred):
    smooth = 1.0
    y_true_flat = K.flatten(y_true)
    y_pred_flat = K.flatten(y_pred)
    intersection = K.sum(y_true_flat * y_pred_flat)
    return (2.0 * intersection + smooth) / (K.sum(y_true_flat) + K.sum(y_pred_flat) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -1.0 * dice_coef(y_true, y_pred)


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

class UNet():

    def __init__(self):
        self.net = None

    def load_data(self,nb_rows,nb_cols,image_path,mask_path):
        imgs, msks = tiff.imread(image_path+'/train-volume.tif'), tiff.imread(mask_path+'/train-labels.tif') / 255
        montage_imgs = np.empty((nb_rows * imgs.shape[1], nb_cols * imgs.shape[2]), dtype=np.float32)
        montage_msks = np.empty((nb_rows * imgs.shape[1], nb_cols * imgs.shape[2]), dtype=np.int8)
        idxs = np.arange(imgs.shape[0])
        np.random.shuffle(idxs)
        idxs = iter(idxs)
        for y0 in range(0, montage_imgs.shape[0], imgs.shape[1]):
            for x0 in range(0, montage_imgs.shape[1], imgs.shape[2]):
                y1, x1 = y0 + imgs.shape[1], x0 + imgs.shape[2]
                idx = next(idxs)
                montage_imgs[y0:y1, x0:x1] = imgs[idx]
                montage_msks[y0:y1, x0:x1] = msks[idx]
        return montage_imgs, montage_msks

    def compile(self, loss=bce_dice_loss,classes=2):
        K.set_image_dim_ordering('tf')
        x = inputs = Input(shape=(512,512), dtype='float32')
        x = Reshape((512,512) + (1,))(x)

        down1 = Conv2D(44, 3, activation='relu', padding='same')(x)
        b1 = BatchNormalization()(down1)
        b1 = Dropout(rate=0.3)(b1)
        down1 = Conv2D(44, 3, activation='relu', padding='same')(b1)
        b2 = BatchNormalization()(down1)
        down1pool = MaxPooling2D((2, 2), strides=(2, 2))(b2)
        down1pool = Dropout(rate=0.3)(down1pool)
        down2 = Conv2D(88, 3, activation='relu', padding='same')(down1pool)
        b3 = BatchNormalization()(down2)
        down2 = Conv2D(88, 3, activation='relu', padding='same')(b3)
        b4 = BatchNormalization()(down2)
        down2pool = MaxPooling2D((2,2), strides=(2, 2))(b4)
        down2pool = Dropout(rate=0.3)(down2pool)
        down3 = Conv2D(176, 3, activation='relu', padding='same')(down2pool)
        b5 = BatchNormalization()(down3)
        down3 = Conv2D(176, 3, activation='relu', padding='same')(b5)
        b6 = BatchNormalization()(down3)
        down3pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
        down3pool = Dropout(rate=0.3)(down3pool)

        # stacked dilated convolution at the bottleneck
        dilate1 = Conv2D(176,3, activation='relu', padding='same', dilation_rate=1)(down3pool)
        b7 = BatchNormalization()(dilate1)
        dilate2 = Conv2D(176,3, activation='relu', padding='same', dilation_rate=2)(b7)
        b8 = BatchNormalization()(dilate2)
        dilate3 = Conv2D(176,3, activation='relu', padding='same', dilation_rate=4)(b8)
        b9 = BatchNormalization()(dilate3)
        dilate4 = Conv2D(176,3, activation='relu', padding='same', dilation_rate=8)(b9)
        b10 = BatchNormalization()(dilate4)
        dilate5 = Conv2D(176,3, activation='relu', padding='same', dilation_rate=16)(b10)
        b11 = BatchNormalization()(dilate5)
        dilate6 = Conv2D(176,3, activation='relu', padding='same', dilation_rate=32)(b11)

        dilate_all_added = add([dilate1, dilate2, dilate3, dilate4, dilate5, dilate6])

        up3 = UpSampling2D((2, 2))(dilate_all_added)
        up3 = Conv2D(88,3, activation='relu', padding='same')(up3)
        up3 = concatenate([down3, up3])
        b12 = BatchNormalization()(up3)
        b12 = Dropout(rate=0.3)(b12)
        up3 = Conv2D(88,3, activation='relu', padding='same')(b12)
        b13 = BatchNormalization()(up3)
        up3 = Conv2D(88,3, activation='relu', padding='same')(b13)

        up2 = UpSampling2D((2, 2))(up3)
        up2 = Conv2D(44,3, activation='relu', padding='same')(up2)
        up2 = concatenate([down2, up2])
        b14 = BatchNormalization()(up2)
        b14 = Dropout(rate=0.3)(b14)
        up2 = Conv2D(44,3, activation='relu', padding='same')(b14)
        b15 = BatchNormalization()(up2)
        up2 = Conv2D(44,3, activation='relu', padding='same')(up2)

        up1 = UpSampling2D((2, 2))(up2)
        up1 = Conv2D(22,3, activation='relu', padding='same')(up1)
        up1 = concatenate([down1, up1])
        b16 = BatchNormalization()(up1)
        b16 = Dropout(rate=0.3)(b16)
        up1 = Conv2D(22,3, activation='relu', padding='same')(b16)
        b17 = BatchNormalization()(up1)
        up1 = Conv2D(22,3, activation='relu', padding='same')(b17)
        b18 = BatchNormalization()(up1)
        x = Conv2D(classes, 1, activation='softmax')(b18)
        x = Lambda(lambda x: x[:, :, :, 1], output_shape=(512,512))(x)
        self.net = Model(inputs=inputs, outputs=x)
        self.net.compile(optimizer=RMSprop(), loss=loss, metrics=[dice_coef])
        self.net.summary()
        return


    def train(self,lr,max_lr):
        gen_trn = self.batch_generator(imgs=X_train, msks=Y_train, batch_size=3)
        gen_val = self.batch_generator(imgs=X_val, msks=Y_val, batch_size=3)
        clr_triangular = CyclicLR(mode='triangular')
        clr_triangular._reset(new_base_lr=lr, new_max_lr=max_lr)
        cb = [clr_triangular,EarlyStopping(monitor='val_loss', min_delta=1e-3, 
        patience=300, verbose=1, mode='min'),
            ModelCheckpoint('weights/' + 'U-net.weights',
            monitor='val_loss', save_best_only=True, verbose=1),
            TensorBoard(log_dir='./logs',write_grads=True, batch_size=3, 
            write_graph=True, write_images=True)]
    
        self.net.fit_generator(generator=gen_trn, steps_per_epoch=10, epochs=epochs,
                               validation_data=gen_val, validation_steps=10, verbose=1, callbacks=cb)

        return

    def batch_generator(self, imgs, msks, batch_size,transform=True):

        H, W = imgs.shape
        wdw_H, wdw_W = (512,512)
        mean, std_dev = np.mean(imgs), np.std(imgs)
        normalize = lambda x: (x - mean) / (std_dev + 1e-10)

        while True:

            img_batch = np.zeros((batch_size,) + (512,512), dtype=imgs.dtype)
            msk_batch = np.zeros((batch_size,) + (512,512), dtype=msks.dtype)

            for batch_idx in range(batch_size):
                # Sample a random window.
                y0, x0 = np.random.randint(0, H - wdw_H), np.random.randint(0, W - wdw_W)
                y1, x1 = y0 + wdw_H, x0 + wdw_W

                img_batch[batch_idx] = imgs[y0:y1, x0:x1]
                msk_batch[batch_idx] = msks[y0:y1, x0:x1]

            img_batch = normalize(img_batch)
            yield img_batch, msk_batch


if __name__ == "__main__":
    arg_list = argparse.ArgumentParser()
    arg_list.add_argument('--epochs', help='Number of epochs to run', default='300', type=int)
    arg_list.add_argument('--classes', help='Number of output classes', default='2', type=int)
    arg_list.add_argument('--lr', help='Learning rate', default='0.00001', type=float)
    arg_list.add_argument('--max_lr', help='Learning rate', default='0.0005', type=float)
    arg_list.add_argument('--image_path', help='Path to stack of training images', default='images', type=str)
    arg_list.add_argument('--mask_path', help='Path to stack of ground truth annotations', default='masks', type=str)

    args = vars(arg_list.parse_args())

    epochs = args['epochs']
    lr = args['lr']
    max_lr = args['max_lr']
    classes = args['classes']
    image_path = args['image_path']
    mask_path = args['mask_path']

    model_init = UNet()
    (X_train, Y_train) = model_init.load_data(nb_cols=5,nb_rows=6,
    image_path=image_path,mask_path=mask_path)

    (X_val, Y_val) = model_init.load_data(nb_cols=6,nb_rows=5,
    image_path=image_path,mask_path=mask_path)
    model_init.compile()
    model_init.train(lr=lr,max_lr=max_lr)
