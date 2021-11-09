# pFedHN_LSTM
Personalized Federated HyperNetworks for Heart Rate Prediction using LSTM

"""
This project is inspired by the pFedHN model by https://github.com/AvivSham/pFedHN, and the heart rate prediction model by https://github.com/nijianmo/fit-rec

The idea of this project is to create a personalized federated training setting using HyperNetworks, using prediction model is the LSTM suggested by Jianmo Ni in https://github.com/nijianmo/fit-rec

The LSTM model is originally written in Keras, and therefore not consistent with the pFedHN model. We use the heart rate prediction LSTM model architecture by Jianmo Ni and implement that model with pFedHN model in PyTorch.
"""

Setup and running guide:

1. Simply download and setup pFedHN as in https://github.com/AvivSham/pFedHN, then copy the scripts to the directory /experiments/pfedhn

2. Try executing "HRFL_trainer.py" and modifying the data path

