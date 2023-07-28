# Imports the necessaries libraies and module
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from collections import OrderedDict
from workspace_utils import active_session, keep_awake
from PIL import Image

class PredictionImage:
    """ 
        Description:
            Image predict class for implementing all methode need for predict image.
            All these parameters are provided by user using command line.
        Parameters:
            -- checkpoint: path to the checkpoint model,
            -- input_image: path to the input image,
            -- category_names: categories of label, 
            -- gpu: enable or disable the gpu mode,
            -- top_k: k probables predited class
    """
    def __init__(self, checkpoint, input_image, category_names, gpu, top_k):
        self.checkpoint=checkpoint
        self.input_image=input_image
        self.category_names=category_names
        self.gpu=gpu, 
        self.top_k=top_k
        pass
    
    def predict(self):
        ''' Predict the class (or classes) of an image using a trained deep learning model.
            Parameters:
                None: None parameters are provided
            Returns :
                predicted classes and probabilities
        '''
        
        model = self.load_checkpoint()
        if self.gpu and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
        model.to(device)
        # Preprocess the image
        processed_image = self.process_image(self.input_image)
        processed_image = processed_image.unsqueeze(0)  # Add batch dimension

        # Move the image tensor to the same device as the model
        processed_image = processed_image.to(device)

        # Disable gradients to speed up the prediction process
        with torch.no_grad():
            model.eval()
            # Make the prediction
            output = model.forward(processed_image)
            # Get the top K probabilities and class indices
            top_probs, top_indices = torch.topk(F.log_softmax(output, dim=1), self.top_k, dim=1)

        # Convert indices to class labels using the model's class_to_idx dictionary
        idx_to_class = {v: k for k, v in model.class_to_idx.items()}
        top_classes = [idx_to_class[idx.item()] for idx in top_indices[0]]
        top_classes_label = [self.category_names[x] for x in top_classes]
        
        # Return probabilities and top classes label
        return top_probs[0].tolist(), top_classes_label
    
    def load_checkpoint(self):
        """
        Loading model checkpoint provided by the user with the command line;
        Torchvision method are used here to achieve it.
        parameters:
            None: None value are provided to this function.
        return:
            model: return the model load using the provided checkpoint
        """
        checkpoint = torch.load(self.checkpoint)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        model.class_to_idx = checkpoint['class_to_idx']
        return model
    
    def process_image(self, image_path):
        ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
            returns an Numpy array
        '''
        # Open the image using PIL
        with Image.open(image_path) as image:

            # Resize the image while maintaining the aspect ratio
            shortest_side = 256
            image.thumbnail((shortest_side, shortest_side))

            # Crop the center 224x224 portion of the image
            width, height = image.size
            left = (width - 224) / 2
            top = (height - 224) / 2
            right = left + 224
            bottom = top + 224
            image = image.crop((left, top, right, bottom))

            # Convert the image to a Numpy array
            np_image = np.array(image)

            # Normalize the color channels to floats in the range [0, 1]
            np_image = np_image / 255.0

            # Normalize the image with the specified means and standard deviations
            means = np.array([0.485, 0.456, 0.406])
            stds = np.array([0.229, 0.224, 0.225])
            np_image = (np_image - means) / stds

            # Reorder the dimensions so that color channel is the first dimension
            np_image = np_image.transpose((2, 0, 1))

            # Convert the Numpy array to a PyTorch tensor
            tensor_image = torch.tensor(np_image, dtype=torch.float)
        return tensor_image
