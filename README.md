# MVIT

## Introduction
A multiscale vision transformer for classification of Cholec80 surgical video data.
**Framework**: Torch

## Dataset
It uses the cholec80 dataset containing 80 surgical videos of Cholesystectomy surgery and classifying data into following 7 surgical phases.  

**Surgical Phases**
- Preparation
- CalotTriangleDissection
- ClippingCutting
- GallbladderDissection
- GallbladderPackaging
- CleaningCoagulation
- GallbladderRetraction

The dataset video has length ranging from 0.5 hours to 2.5 hours with 25fps frame rate. The frame size of each video is 854x480. The input size of model used here is 224x224. So for the classification purpose each videos are initially converted into 224x224 size by centre cropping and scaling using ffmpeg video codec. This make the preprocessing of the video much faster and reduce overall size of the dataset. Resultant video data possess a key frame in the interval of around 10 seconds.

## Dataset into Tensor
The torchvision.io.VideoReader class is utilized for converting the video data into corresponding tensor.
