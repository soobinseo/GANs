# GANs

## Description
A tensorflow implementation of GANs with variable loss function including Standard GAN, Least-Squared GAN (LSGAN), Wasserstein GAN (WGAN), improved Wasserstein GAN, and DRAGAN

## Datasets
I used [celebA dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and cropped the center face by 64X64 pixels.

## Results
- Standard GAN

<p>
  <img src="https://raw.githubusercontent.com/soobin3230/GANs/master/png/standard_1.png"/>
  <img src="https://raw.githubusercontent.com/soobin3230/GANs/master/png/standard_2.png"/>
  <img src="https://raw.githubusercontent.com/soobin3230/GANs/master/png/standard_3.png"/>
  <img src="https://raw.githubusercontent.com/soobin3230/GANs/master/png/standard_4.png"/>
  <img src="https://raw.githubusercontent.com/soobin3230/GANs/master/png/standard_5.png"/>
  <img src="https://raw.githubusercontent.com/soobin3230/GANs/master/png/standard_6.png"/>
</p>

- LSGAN

<p>
  <img src="https://raw.githubusercontent.com/soobin3230/GANs/master/png/ls_1.png"/>
  <img src="https://raw.githubusercontent.com/soobin3230/GANs/master/png/ls_2.png"/>
  <img src="https://raw.githubusercontent.com/soobin3230/GANs/master/png/ls_3.png"/>
  <img src="https://raw.githubusercontent.com/soobin3230/GANs/master/png/ls_4.png"/>
  <img src="https://raw.githubusercontent.com/soobin3230/GANs/master/png/ls_5.png"/>
  <img src="https://raw.githubusercontent.com/soobin3230/GANs/master/png/ls_6.png"/>
</p>

- WGAN

<p>
  <img src="https://raw.githubusercontent.com/soobin3230/GANs/master/png/w_1.png"/>
  <img src="https://raw.githubusercontent.com/soobin3230/GANs/master/png/w_2.png"/>
  <img src="https://raw.githubusercontent.com/soobin3230/GANs/master/png/w_3.png"/>
  <img src="https://raw.githubusercontent.com/soobin3230/GANs/master/png/w_4.png"/>
  <img src="https://raw.githubusercontent.com/soobin3230/GANs/master/png/w_5.png"/>
  <img src="https://raw.githubusercontent.com/soobin3230/GANs/master/png/w_6.png"/>
</p>

- Improved WGAN

<p>
  <img src="https://raw.githubusercontent.com/soobin3230/GANs/master/png/impw_1.png"/>
  <img src="https://raw.githubusercontent.com/soobin3230/GANs/master/png/impw_2.png"/>
  <img src="https://raw.githubusercontent.com/soobin3230/GANs/master/png/impw_3.png"/>
  <img src="https://raw.githubusercontent.com/soobin3230/GANs/master/png/impw_4.png"/>
  <img src="https://raw.githubusercontent.com/soobin3230/GANs/master/png/impw_5.png"/>
  <img src="https://raw.githubusercontent.com/soobin3230/GANs/master/png/impw_6.png"/>
</p>

## Dependencies

1. tensorflow >= 1.0.0
1. numpy >= 1.12.0

## File description
  * `hyperparams.py` includes all hyper parameters that are needed.
  * `data.py` loads training data and crops them.
  * `modules.py` contains customized conv net and so on.
  * `networks.py` builds a generator and a discriminator.
  * `train.py` is for training.

## Training the network
  * STEP 1. Adjust hyper parameters in `hyperparams.py`, especially the hyperparameter "loss" that you want to train.
  * STEP 2. create 'data' directory, then download and extract celebA data at the directory.
  * STEP 3. Run `train.py` and show result through tensorboard.
