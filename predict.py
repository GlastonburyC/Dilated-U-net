import keras
from dilated_unet import UNet
from keras.models import load_model
from dilated_unet import bce_dice_loss
from dilated_unet import dice_coef
import tifffile as tiff
keras.losses.bce_dice_loss = bce_dice_loss
keras.metrics.dice_coef = dice_coef
import numpy as np
import mahotas as mh
from matplotlib.image import imsave

model = load_model('weights/normal_unet_noadd.weights')

mean, std_dev = np.mean(test), np.std(test)
normalize = lambda x: (x - mean) / (std_dev + 1e-10)

test = tiff.imread('test/test-volume.tif')
#test.shape() (30,512,512)
mean, std_dev = np.mean(test), np.std(test)
normalize = lambda x: (x - mean) / (std_dev + 1e-10)

test = normalize(test)

test_preds = model.predict(test)

#  test_preds = np.array(test_preds * 255,dtype='uint8')

# for i in range(0,30):
#   test_preds[i,] = test_preds[i,] > 160


imsave('test/test_predictions/test_preds.tif',test_preds)
