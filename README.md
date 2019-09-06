# 3D convolutional neural networks for remote pulse rate measurement and mapping from facial video

Remote pulse rate measurement from facial video has gained particular attention over the last few years. Researches exhibit significant advancements and demonstrate that common video cameras correspond to reliable devices that can be employed to measure a large set of biomedical parameters without any contact with the subject. 

This repository contains the source codes related to a new framework for measuring and mapping pulse rate from video. The method, which relies on convolutional 3D networks, is fully automatic and does not require any special image pre-processing (see figure below). The network ensures concurrent mapping by producing a prediction for each local group of pixels. 

The volume of either uncompressed and labeled (with reference pulse rate values) video data being very limited to train this type of machine learning models, we propose a pseudo-PPG synthetic video generator (also contained in this repository).

![Alt text](illustrations/overview.png?raw=true "General overview")
.center[A figure caption.]

## Reference

If you find this code useful or use it in an academic or research project, please cite it as: 
Frédéric Bousefsaf, Alain Pruski, Choubeila Maaoui, **3D convolutional neural networks for remote pulse rate measurement and mapping from facial video**, Journal, 2019.


## Scientific description
Please refer to the original publication to get all the details.

(top) Conventional approach: image processing operations are applied on the video stream to detect pixels or region(s) of interest (ROI). The signal is traditionally computed using a spatial averaging operation over the ROI before being processed with spectral or temporal filters. Finally, biomedical parameters like pulse rate are estimated from this signal. (bottom) The approach we propose consists in training an artificial intelligence model using only synthetic data. The input corresponds to a video stream (image sequence). The model predicts a pulse rate for each video patch and thus produces a map of predictions instead of a single estimation.



![Alt text](illustrations/network_architecture.png?raw=true "General overview")

![Alt text](illustrations/synthetic_generator.png?raw=true "General overview")



## Usage
This repository contains two sub-repository:





