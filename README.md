# 3D convolutional neural networks for remote pulse rate measurement and mapping from facial video

Remote pulse rate measurement from facial video has gained particular attention over the last few years. Researches exhibit significant advancements and demonstrate that common video cameras correspond to reliable devices that can be employed to measure a large set of biomedical parameters without any contact with the subject. 

This repository contains the source codes related to a new framework for measuring and mapping pulse rate from video. The method, which relies on convolutional 3D networks, is fully automatic and does not require any special image pre-processing (see figure below). The network ensures concurrent mapping by producing a prediction for each local group of pixels. 

The volume of either uncompressed and labeled (with reference pulse rate values) video data being very limited to train this type of machine learning models, we propose a pseudo-PPG synthetic video generator (also contained in this repository).

![Alt text](illustrations/overview.png?raw=true "General overview")
*(top) Conventional approach: image processing operations are applied on the video stream to detect pixels or region(s) of interest (ROI). The signal is traditionally computed using a spatial averaging operation over the ROI before being processed with spectral or temporal filters. Finally, biomedical parameters like pulse rate are estimated from this signal. (bottom) The approach we propose consists in training an artificial intelligence model using only synthetic data. The input corresponds to a video stream (image sequence). The model predicts a pulse rate for each video patch and thus produces a map of predictions instead of a single estimation.*


## Reference

If you find this code useful or use it in an academic or research project, please cite it as: 

Frédéric Bousefsaf, Alain Pruski, Choubeila Maaoui, **3D convolutional neural networks for remote pulse rate measurement and mapping from facial video**, *Applied Sciences*, vol. 9, n° 20, 4364 (2019). [Link](https://www.mdpi.com/2076-3417/9/20/4364)

You can also visit my [website](https://sites.google.com/view/frederic-bousefsaf) for additional information.

## Scientific description

Please refer to the original publication to get all the details. A 3D CNN classifier structure has been developed for both extraction and classification of unprocessed video streams. The CNN acts as a feature extractor. Its final activations feeds two dense layers (multilayer perceptron) that are used to classify pulse rate (see figure below).

![Alt text](illustrations/network_architecture.png?raw=true "Network architecture")
*Model architecture. The network integrates a 3D convolution (blue) with its associated 3D pooling (green) layers. The stream converges to two fully dense layers (orange).*


A suitable amount of synthetic data ensures proper training and validation of machine learning models that contain a very large number of intrinsic variables. To this end, we propose a synthetic iPPG video streams generator. The procedure is five-fold (see figure below).


![Alt text](illustrations/synthetic_generator.png?raw=true "Synthetic generator")
*A waveform model, fitted to real iPPG pulse waves using Fourier series, is employed to construct a generic wave (a). A two seconds signal is produced from this waveform (b) and a linear, quadratic or cubic tendency is added (c). Note that both amplitude and frequency are controlled. The uni-dimensional pulse signal is then transformed to a video using vector repetition (d). Random noise is independently added to each image of the video stream (e). This step reproduces natural fluctuations due to camera noise that randomly appear in images.*


## Requirements
The codes were tested with Python 3.5/3.6 and Tensorflow + Keras frameworks.

Different packages must be installed to properly run the codes : 
- `pip install tensorflow` (or `tensorflow-gpu`)
- `pip install opencv-python`
- `pip install matplotlib`
- `pip install scipy`


## Usage

This repository contains two sub-repository: **train** and **predict** 

### train
`main.py` is a standalone program that creates the network architecture and the synthetic data before launching training. The code saves the architecture (.json) and weights (.h5) as well as statistics (training / validation loss and accuracy) at the end of the procedure.

**Variables and hyperparameters**
- `NB_VIDEOS_BY_CLASS_TRAIN` and `NB_VIDEOS_BY_CLASS_VALIDATION`: number of synthetic videos (a video is a tensor of size 25 x 25 x 60) that will be generated and used for training and validation.
- `EPOCHS`: 5000 by default.
- `CONTINUE_TRAINING`: `False` to start training from scratch (weights are randomly initialized), `True` to resume training.
- `SAVE_ALL_MODELS`: `False` to save only the model that presents the highest validation accuracy.


### predict
`main.py` first display an UI that allows selection of the folder that contains the images. The program will load all the files in the selected directory so it must contain only image files. (e.g. 0000.png; 0001.png; 0002.png...).

**Variables and hyperparameters**
- `DIR_SAVE`: save directory. The predictions will be saved in a Matlab format (.mat) in this path.
- `USE_RANDOM_PIXEL_LOCATION`: 1 by default. 0 to disable shuffle of pixel positions. 

***Please note that the codes provided in these subdirectories are temporary. We will update the latest versions soon.***
