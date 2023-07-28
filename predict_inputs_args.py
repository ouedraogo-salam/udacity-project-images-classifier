# PROGRAMMER: OUEDRAOGO Salam
# DATE CREATED: 26/07/2023                                  
# REVISED DATE: 28/07/2023    

# Imports python modules

import argparse
def predict_inputs_args():
    """
        Retrieves and parses  command line arguments provided by the user when
        they run the program from a terminal window. This function uses Python's 
        argparse module to created and defined these command line arguments. If 
        the user fails to provide some or all of the arguments, then the default 
        values are used for the missing arguments. 
        Command Line Arguments:
          1. Path to the input image as input required
          2. Path to the Model checkpoint as checkpoint with default value /opt
          3. The k most probable classes as --top_k with default value 5
          4. Label to category name as --categories_names with default value cat_to_name.json
          5. Disable/Enable GPU mode as --gpu with default value False
          
        This function returns these arguments as an ArgumentParser object.
        Parameters:
         None - simply using argparse module to create & store command line arguments
        Returns:
         parse_args() -data structure that stores the command line arguments object  
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('input', type=str,\
                                 help='path to the input image')
    argument_parser.add_argument('checkpoint', type=str, default='/opt',\
                                 help='path to the model checkpoint')
    argument_parser.add_argument('--top_k', type=int, default=5,\
                                 help='predict the k most probable classes')
    argument_parser.add_argument('--category_names', type=str, default="cat_to_name.json",\
                                 help='label to category name')
    argument_parser.add_argument('--gpu', type=bool, default=False, const=True, nargs='?',\
                                 help='Enabling GPU mode during training process')
    
    return argument_parser.parse_args()