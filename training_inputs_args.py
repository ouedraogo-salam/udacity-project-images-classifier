# PROGRAMMER: OUEDRAOGO Salam
# DATE CREATED: 26/07/2023                                  
# REVISED DATE: 28/07/2023    
# Imports python modules

import argparse
def training_inputs_args():
    """
        Retrieves and parses  command line arguments provided by the user when
        they run the program from a terminal window. This function uses Python's 
        argparse module to created and defined these command line arguments. If 
        the user fails to provide some or all of the arguments, then the default 
        values are used for the missing arguments. 
        Command Line Arguments:
          1. Images Folder as data_dir required
          2. CNN model saving dir as --save_dir with default value /opt/
          3. CNN Model Architecture as --arch with default value 'vgg'
          4. Learning rate as --learning_rate with default value 0.001
          5. Number of Hidden Layers units as --hidden_units with default value 4096
          6. Number of epochs for the training as --epochs with default 5
          7. Disable/Enable GPU mode as --gpu with default value False
          
        This function returns these arguments as an ArgumentParser object.
        Parameters:
         None - simply using argparse module to create & store command line arguments
        Returns:
         parse_args() -data structure that stores the command line arguments object  
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('data_dir', type=str,\
                                 help='path to the folder containing images')
    argument_parser.add_argument('--save_dir', type=str,default='/opt',\
                                 help='path to save the trained model')
    argument_parser.add_argument('--arch', type=str, default='vgg19',\
                                 help='CNN model architecture the')
    argument_parser.add_argument('--learning_rate', type=float,default=0.001,\
                                 help='learning rate for training the model')
    argument_parser.add_argument('--hidden_units', type=int, default=4096,\
                                 help='Number of hidden layer units')
    argument_parser.add_argument('--epochs', type=int, default=5,\
                                 help='number of epochs')
    argument_parser.add_argument('--gpu', type=bool, default=False, const=True, nargs='?',\
                                 help='Enabling GPU mode during training process')
    
    return argument_parser.parse_args()