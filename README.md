# HMS---Harmful-Brain-Activity-Classification-Silver-Medal
This repository contains code for detecting and classifying epileptic seizures and other harmful brain activities using EEG data. This project was part of a Kaggle competition where we achieved a silver medal, ranking within the top 3% of participants.

Project Overview
Objective: Improve the efficiency and accuracy of EEG monitoring for ICU patients by detecting and classifying epileptic seizures.
Dataset: Utilized both official spectrogram data provided by the competition organizers and self-extracted Mel-spectrogram features from raw EEG data.
Model: EfficientNetB0 architecture, trained with group k-fold cross-validation.
Evaluation Metric: Kullback-Leibler divergence.
File Descriptions
1. eeg_gen_spec
Purpose: Extract features from EEG data and convert them into Mel-spectrograms.

Details:

This script uses the librosa package to generate Mel-spectrogram features from raw EEG data.
Mel-spectrogram parameters:

mel_spec = librosa.feature.melspectrogram(y=x, sr=200, hop_length=len(x)//256, n_fft=1024, n_mels=128, fmin=0, fmax=20, win_length=128)
python eeg_gen_spec.py --input data/raw_eeg/ --output data/mel_spectrograms/

2. train_eff0_stage1

Purpose: Train the EfficientNetB0 model using 5-fold cross-validation.

Details:
This script handles the initial stage of training with group k-fold cross-validation.
It splits the samples into five segments based on patient IDs to ensure robust testing and prevent overfitting.
python train_eff0_stage1.py --data_path data/combined_spectrograms/ --k_folds 5

3. train_eff0_stage2

Purpose: Conduct secondary training on selected data where the prediction vote is greater than 10.

Details:

This script refines the model by training on a subset of data identified as high-confidence predictions from the first stage.
python train_eff0_stage2.py --data_path data/high_confidence/ --model_path models/eff0_stage1/

4. infer

Purpose: Load the trained model and make predictions on new data.

Details:

This script loads all necessary models and preprocessing steps to make final predictions on the test dataset.

It ensures that all feature extraction and model weights are properly loaded for inference.

python infer.py --model_path models/eff0_final/ --input data/test/ --output results/predictions.csv


Installation

Install the required libraries using pip:
pip install tensorflow keras librosa numpy pandas scikit-learn matplotlib


Results

Score: Achieved a score of 0.347.

Rank: Secured a ranking within the top 3% of participants.

Award: Silver medal for outstanding performance.

Contributors


Wenxuan "Gwen" Xiao
