# Overview
#* Work with the audio data
#* Represent an audio data - Time Domain and Spectrogram
#* Build a deep learning model while working with audio data

# Understanding the Problem Statement

'''
According to the National Crime Records Bureau, nearly 24,012 people die each day due to a delay in getting medical assistance. Many accident victims wait for help at the site, and a delay costs them their lives. The reasons could range from ambulances stuck in traffic to the fire brigade not being able to reach the site on time due to traffic jams. 

The solution to the above problem is to create a system that automatically detects the emergency vehicle prior to reaching the traffic signals and change the traffic signals accordingly.


'''

# Dataset
'''
Download the dataset from [here](https://drive.google.com/file/d/1VBI_X6GyYvf8j3T70-_hVDyhR_sUzeCr/view?usp=sharing)

<br>

## Import Libraries

Let us first import the libraries into our environment

* **Librosa** is an open source library in Python that is used for audio and music analyis

* **Scipy** is a python library for scientific & technical computing. It contains modules for signal processing, image processing, and linear algebera


'''

# For audio processing
import librosa
import scipy
import pandas as pd


print(librosa.__version__)
print(scipy.__version__)

# For playing audio
import IPython.display as ipd

# For array processing
import numpy as np

# For visualization 
import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', 200)


import zipfile
import os

# Specify the path to the zip file
zip_file_path = 'data/audio.zip'

# Specify the directory to extract to
extract_to_dir = 'data/unzipped_contents'

# Create a directory to extract to if it doesn't exist
os.makedirs(extract_to_dir, exist_ok=True)

# Open the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    # Extract all the contents into the directory
    zip_ref.extractall(extract_to_dir)
    
    # List the contents of the extracted folder
    print(f"Contents of the zip file '{zip_file_path}':")
    for file_name in zip_ref.namelist():
        print(file_name)

# import emergency vehicle data
path='data/audio/emergency.wav'
emergency,sample_rate = librosa.load(path, sr = 16000)


# import non-emergency vehicle data
path='data/audio/nonemergency.wav'
non_emergency,sample_rate = librosa.load(path, sr =16000)

'''
We have used the sampling rate (sr) of 16000 to read the above audio data. An audio wave of 2 seconds with a sampling rate of 16,000 will have 32,000 samples.
'''

#__Find the duration of the audio clips__
duration1 = librosa.get_duration(y=emergency, sr=16000)



duration2 = librosa.get_duration(y=non_emergency, sr=16000)


print("Duration of an emergency and Non Emergency (in min):",duration1/60,duration2/60)

'''
## Preparing Data

Let us break the audio into chunks of 2 seconds. So, let us define the function for the same task

'''
def prepare_data(audio_data, num_of_samples=32000, sr=16000):
  
  data=[]
  for offset in range(0, len(audio_data), sr):
    start = offset
    end   = offset + num_of_samples
    chunk = audio_data[start:end]
    
    if(len(chunk)==32000):
      data.append(chunk)
    
  return data

# prepare audio chunks
emergency = prepare_data(emergency)
non_emergency = prepare_data(non_emergency)

print("No. of Chunks of Emergency and Non Emergency:",len(emergency),len(non_emergency))

ipd.Audio(emergency[136],rate=16000)

ipd.Audio(non_emergency[10],rate=16000)

## Visualization of Audio Data
plt.figure(figsize=(14,4))
plt.plot(np.linspace(0, 2, num=32000),emergency[103])
plt.title('Emergency')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()


plt.figure(figsize=(14,4))
plt.plot(np.linspace(0, 2, num=32000),non_emergency[102])
plt.title('Non Emergency')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.show()

#__Combine Emergecy and Non Emergency chunks__
audio = np.concatenate([emergency,non_emergency])

# assign labels 
labels1 = np.zeros(len(emergency))
labels2 = np.ones(len(non_emergency))

# concatenate labels
labels = np.concatenate([labels1,labels2])

print(audio.shape)

#**Split into train and validation set**
from sklearn.model_selection import train_test_split
x_tr, x_val, y_tr, y_val = train_test_split(np.array(audio),np.array(labels),
                                            stratify=labels,test_size = 0.1,
                                            random_state=777,shuffle=True)

print(x_tr.shape, x_val.shape)

x_tr_features  = x_tr.reshape(len(x_tr),-1,1)
x_val_features = x_val.reshape(len(x_val),-1,1)

print("Reshaped Array Size",x_tr_features.shape)

'''
## Model Architecture

Let's define the model architecture using conv1D layers  and the time domain features.

'''

from keras.layers import Input, Conv1D, Dropout, MaxPooling1D, GlobalMaxPool1D, Dense
from keras.models import Model
from keras.callbacks import ModelCheckpoint

# CNN based deep learning model architecture
def conv_model(x_tr):
  
  inputs = Input(shape=(x_tr.shape[1],x_tr.shape[2]))

  #First Conv1D layer
  conv = Conv1D(8, 13, padding='same', activation='relu')(inputs)
  conv = Dropout(0.3)(conv)
  conv = MaxPooling1D(2)(conv)

  #Second Conv1D layer
  conv = Conv1D(16, 11, padding='same', activation='relu')(conv)
  conv = Dropout(0.3)(conv)
  conv = MaxPooling1D(2)(conv)

  # Global MaxPooling 1D
  conv = GlobalMaxPool1D()(conv)

  #Dense Layer 
  conv = Dense(16, activation='relu')(conv)
  outputs = Dense(1,activation='sigmoid')(conv)

  model = Model(inputs, outputs)
  
  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
  model_checkpoint = ModelCheckpoint('best_model.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
  
  return model, model_checkpoint

model, model_checkpoint = conv_model(x_tr_features)

model.summary()

# model training
history = model.fit(x_tr_features, y_tr ,epochs=10, 
                    callbacks=[model_checkpoint], batch_size=32, 
                    validation_data=(x_val_features,y_val))

# load the best model weights
model.load_weights('best_model.hdf5')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# check model's performance on the validation set
_, acc = model.evaluate(x_val_features,y_val)
print("Validation Accuracy:",acc)

# input audio

ind=35
test_audio = x_val[ind]
ipd.Audio(test_audio,rate=16000)

# classification
feature = x_val_features[ind]
prob = model.predict(feature.reshape(1,-1,1))
if (prob[0][0] < 0.5):
  pred='emergency'
else:
  pred='non emergency' 

print("Prediction:",pred)

# reshape chunks
x_tr_features  = x_tr.reshape(len(x_tr),-1,160)
x_val_features = x_val.reshape(len(x_val),-1,160)

print("Reshaped Array Size",x_tr_features.shape)

from keras.layers import LSTM

# LSTM based deep learning model architecture
def lstm_model(x_tr):
  
  inputs = Input(shape=(x_tr.shape[1],x_tr.shape[2]))

  #lstm
  x = LSTM(128)(inputs)
  x = Dropout(0.3)(x)
  
  #dense
  x= Dense(64,activation='relu')(x)
  x= Dense(1,activation='sigmoid')(x)
  
  model = Model(inputs, x)

  model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])
  
  return model

model = lstm_model(x_tr_features)
model.summary()

mc = ModelCheckpoint('best_model.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history=model.fit(x_tr_features, y_tr, epochs=10, 
                  callbacks=[mc], batch_size=32, 
                  validation_data=(x_val_features,y_val))


# load best model weights
model.load_weights('best_model.hdf5')

