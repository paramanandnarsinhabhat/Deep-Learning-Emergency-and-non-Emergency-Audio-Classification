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

