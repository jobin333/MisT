# MisT:Multi-stride History Aware Transformer

## Introduction
A Transformer based architecture for surgical phase classification on Cholec80,M2CAI16 and AutoLaparo Dataset
**Framework**: Torch

## Datasets
The dataset used for this works are Cholec80, M2CAI16 and AutoLaparo

### Cholec80
The cholec80 dataset contains 80 surgical videos of Cholesystectomy surgery and classifying data into following 7 surgical phases.  

**Surgical Phases**
- Preparation
- Calot Triangle Dissection
- Clipping Cutting
- Gallbladder Dissection
- Gallbladder Packaging
- Cleaning Coagulation
- Gallbladder Retraction

The dataset video has length ranging from 0.5 hours to 2.5 hours with 25fps frame rate. The frame size of each video is 854x480. The input size of model used here is 224x224. So for the classification purpose each videos are initially converted into 224x224 size by centre cropping and scaling using ffmpeg video codec. This make the preprocessing of the video much faster and reduce overall size of the dataset. Resultant video data possess a key frame in the interval of around 10 seconds.


### M2CAI16
The cholec80 dataset contains 80 surgical videos of Cholesystectomy surgery and classifying data into following 8 surgical phases.  

**Surgical Phases**
- Preparation
- Calot Triangle Dissection
- Clipping Cutting
- Gallbladder Dissection
- Gallbladder Packaging
- Cleaning Coagulation
- Gallbladder Retraction
- Trocar Placement



### M2CAI16
The procedure of laparoscopic hysterectomy is divided into 7 phases:

**Surgical Phases** 
- Preparation
- Dividing Ligament and Peritoneum
- Dividing Uterine Vessels and Ligament
- Transecting the Vagina
- Specimen Removal
- Suturing
- Washing



## Dataset into Tensor
The torchvision.io.VideoReader class is utilized for converting the video data into corresponding tensor.
