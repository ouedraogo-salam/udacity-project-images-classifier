#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                             
# PROGRAMMER: OUEDRAOGO Salam
# DATE CREATED: 26/07/2023                                  
# REVISED DATE: 28/07/2023    

# Imports python modules

from training_inputs_args import training_inputs_args
from ImageTraining import CNNModelTrainning

# Get training input args from the user using command line
in_args = training_inputs_args()

# CNN model class instanciation
cnn_model = CNNModelTrainning(arch=in_args.arch,\
                              data_dir=in_args.data_dir,\
                              epochs=in_args.epochs,\
                              gpu=in_args.gpu,\
                              hidden_units=in_args.hidden_units,\
                              learning_rate=in_args.learning_rate,\
                             save_dir = in_args.save_dir )
# Training the cnn model
cnn_model.training_model()