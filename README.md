# Dilated-U-net - Stacked Electron Microscopy (EM) segmentation.

Dilated/Atrous convolutions have the ability to increase the receptive field of a network expontentially. Therefore, whilst segmentation requires fine-grain classification accuracy (pixel-wise level), the ability of a network to learn features based on a wide receptive field capturing alot of the input space could be beneficial.

Here I implement a standard U-net against my U-net architecture whose bottleneck/center layers are replaced with expontentially growing dilated convoltions.

The dataset used is the ISBI-2012 stack of electron-microscopy images (n=30) - Representing a challenge to learn from so little data.

<img src="images/train-volume-p1c1pmolsqq5ugdl17011cu4skf.gif" alt="Training images" width="256" height="256"/><img src="images/train-labels-p1c1pngvp9u1148fmnh1i8o5dq.gif" alt="Ground truth labels" width="256" height="256"/>


Cyclical learning rates, much like sudden learning rate drops as compared to exponential lr decay have been shown to improve both final validation accuracy and also decreaser training time ([LN Smith, 2015](https://arxiv.org/abs/1506.01186)).

<img src="https://github.com/bckenstler/CLR/blob/master/images/triangularDiag.png?raw=true"/>

