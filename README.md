# CNN_ASR

This is my first repository on Github :hatching_chick:

This project is based on the tutorial on [Building a Dead Simple Speech Recognition Engine using ConvNet in Keras](https://blog.manash.me/building-a-dead-simple-word-recognition-engine-using-convnet-in-keras-25e72c19c12b)

## Intro
Input: speech -----> CNN -----> Output: text

## Data
audio_data dirctory contains three folders: bed, cat and happy. Each folder contains around 1700 audio files.

[Speech Commands Data Set v0.01](https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data)

## First thing to do
Audio embedding - MFCCs
### Prepocessing audio files
1. Read audio file
2. Take audio input from one channel only
3. Perform downsampling
4. Compute MFCC using librosa
5. Pad the output vectors 
#### preprocess - wav2mfcc
### Save mfcc vectors in a numpy file
#### preprocess - save_data_to_array
### Prepare train set and test set
#### preprocess - get_train_test

## Building CNN model
Convert the input data into 3D

Encode the output into one-hot
#### cnn
#### predict
