# Dilated-U-net - Stacked Electron Microscopy (EM) segmentation.

Dilated/Atrous convolutions have the ability to increase the receptive field of a network expontentially. Therefore, whilst segmentation requires fine-grain classification accuracy (pixel-wise level), the ability of a network to learn features based on a wide receptive field capturing alot of the input space could be beneficial.

Here I implement a standard U-net against my U-net architecture whose bottleneck/center layers are replaced with expontentially growing dilated convoltions.

Standard U-net:

`python dilatedUnet.py --lr 0.00001 --max_lr 0.01 --epochs 500 --dilate 0 --weight_path weights/standard_unet.weights`

Dilated U-net:

`python dilatedUnet.py --lr 0.00001 --max_lr 0.01 --epochs 500 --dilate 1 --weight_path weights/dilated_unet.weights`

The dataset used is the ISBI-2012 stack of electron-microscopy images (n=30) - Representing a challenge to learn from so little data.

<img src="images/train-volume-p1c1pmolsqq5ugdl17011cu4skf.gif" alt="Training images" width="256" height="256"/><img src="images/train-labels-p1c1pngvp9u1148fmnh1i8o5dq.gif" alt="Ground truth labels" width="256" height="256"/>


Cyclical learning rates, much like sudden learning rate drops as compared to exponential lr decay have been shown to improve both final validation accuracy and also decrease training time to convergence ([LN Smith, 2015](https://arxiv.org/abs/1506.01186)). Here I use a triangular lr schedule as implemented by [bckenstler](https://github.com/bckenstler/CLR).

<img src="https://github.com/bckenstler/CLR/blob/master/images/triangularDiag.png?raw=true"/>

