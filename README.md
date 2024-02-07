
# Emergency and Non-Emergency Audio Classification

## Overview
This project aims to work with audio data to build a deep learning model that can differentiate between emergency and non-emergency vehicle sounds. The motivation behind this project is to create a system that could potentially save lives by detecting emergency vehicles and managing traffic systems accordingly to minimize delays in medical assista     nce.

## Problem Statement
Every day, delays in emergency services contribute to the loss of life. Traffic congestion can prevent ambulances and fire brigades from reaching those in urgent need of help. This project attempts to mitigate such issues by providing a means to automatically detect emergency vehicles and alter traffic signals to allow for quicker response times.

## Dataset
The dataset can be downloaded from the following link:
[Emergency Audio Dataset](https://drive.google.com/file/d/1VBI_X6GyYvf8j3T70-_hVDyhR_sUzeCr/view?usp=sharing)

## Setup and Installation
To set up this project, you need to install the necessary dependencies listed in `requirements.txt`. You can install them using the following command:

```bash
pip install -r requirements.txt
```

## File Structure
- `audio.zip`: Compressed file containing the dataset.
- `notebook/`: Contains Jupyter notebooks with the data analysis and model building process.
- `source/`: Contains the source code for the project.
- `best_model.hdf5`: The saved model that achieved the best validation accuracy.
- `requirements.txt`: The list of Python libraries required for the project.

## Usage
After cloning the repository and navigating to the project directory, run the following command to unzip the dataset:

```bash
python -m zipfile -e data/audio.zip data/unzipped_contents
```

Load the emergency and non-emergency audio data using Librosa, process the audio files, and use the provided models for classification.

## Models
The project includes several deep learning models using convolutional and recurrent architectures. The models are built using Keras with a TensorFlow backend.

## Visualization
The project includes scripts to visualize the audio data in the time domain and as spectrograms to understand the differences between emergency and non-emergency audio signals.

## Contributions
Contributions to this project are welcome. You can contribute by improving the models, suggesting new features, or reporting issues.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```
