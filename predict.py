#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#                                                                             
# PROGRAMMER: OUEDRAOGO Salam
# DATE CREATED: 26/07/2023                                  
# REVISED DATE: 28/07/2023    

# Imports python modules
from predict_inputs_args import predict_inputs_args
from PredictionImage import PredictionImage
import json
    

inputs_args = predict_inputs_args()
# open the category_names file using python json module
with open(inputs_args.category_names, 'r') as f:
    cat_to_name = json.load(f)
    pass

# Instanciation of prediction image class
predictImage = PredictionImage(checkpoint=inputs_args.checkpoint,\
                               input_image=inputs_args.input,\
                               category_names=cat_to_name,\
                               gpu=inputs_args.gpu, top_k=inputs_args.top_k)

print('\nThe predicted image output')
print(predictImage.predict())