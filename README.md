# 2020_ROT_SPECIFIC_ANN_EMG_120
Code example for the paper "A User-Specific Hand Gesture Recognition ModelBased on Feed-Forward Neural Networks, EMGs andCorrection of Sensor Orientation".

## Introduction
This file describes briefly the steps needed to run the Matlab code of the paper entitled “A User-Specific
Hand Gesture Recognition Model Based on Feed-Forward Neural Networks, EMGs and Correction of
Sensor Orientation.”
## Description
Dataset and Code for Reading the Data and Evaluating the Results: This folder contains the dataset EPN120
used in the paper for training and testing, and also contains the code to upload the EMGs to Matlab and to
evaluate the results of classification and recognition.
A total of 120 users compose the dataset. The 120 users were split into 2 subsets: one for training and the
other for testing. For each user, we collected 50 EMGs for each of the 5 gestures analyzed and 10 EMGs for
the hand relaxed. From these 50 EMGs, 25 were selected for model design and training, and the other 25
were selected for testing. For the hand relaxed, 5 EMGs were used for model design and training, and the
other 5 were selected for testing. For the testing EMGs of the users from the subset of testing, we do not
include the ground truths to avoid cheating. If a person wants her/his model to be evaluated with our dataset,
she/he needs to send the results in .mat file to email address indicated above following the same structure that
we use in our code.
The remaining folders contain the Matlab code of the paper. To run this code, we suggest using Matlab
2020a or newer versions. Each script contains a description of its function as well as the copyright
information. PLEASE, USE THIS MATERIAL FOR RESEARCH PURPOSES ONLY. If you want to use
the dataset and the code for commercial purposes, please contact to the correspondent author of the paper,
Marco E. Benalcázar, to the email address marco.benalcazar@epn.edu.ec.
## Instructions
1.Download the dataset from https://laboratorio-ia.epn.edu.ec/en/resources/dataset/2020_emg_dataset_120
and paste all the content in the folder "Dataset and Code for Reading the Data and Evaluating the Results".
1.Run the script main.m. After running this script, you will have to wait about 1 hour to obtain the
results. You can observe the progress of the code by observing the command window. In our case,
the code was executed using a personal computer with an AMD FX-8370 Eight-Core Processor of
4GHz and 16GB of RAM.
If you use this material, please cite the following paper:
Marco E. Benalcázar, Angel L. Valdivieso Caraguay, and Lorena I. Barona López, “A User-Specific Hand
Gesture Recognition Model Based on Feed-Forward Neural Networks, EMGs and Correction of Sensor
Orientation,” ICPR 2020.
