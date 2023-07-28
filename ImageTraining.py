# PROGRAMMER: OUEDRAOGO Salam
# DATE CREATED: 26/07/2023                                  
# REVISED DATE: 28/07/2023    

# Imports python modules
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from workspace_utils import active_session

class CNNModelTrainning:
    def __init__(self, arch, data_dir, epochs, gpu, hidden_units, learning_rate, save_dir):
        self.arch = arch
        self.epochs = epochs
        self.gpu = gpu
        self.hidden_units = hidden_units
        self.output_units = 102
        self.learning_rate = learning_rate
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.image_datasets = None
       
        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'valid_test': transforms.Compose([
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }
        pass
    
    def load_datasets(self):
        """ 
        Description:
            Load datasets from the provide directoty as a train directory and a validator directory.
            The function use torchvision methode to transform both training and validating directories.
        
        parameters:
            None: None parameters are provided to the function
        
        Returns:
            return the transform data as dataloaders and validloaders.        
        """
        
        train_dir = self.data_dir + '/train'
        valid_dir = self.data_dir + '/valid'
        image_datasets = datasets.ImageFolder(train_dir, transform=self.data_transforms['train'])
        self.image_datasets = image_datasets

        valid_datasets = datasets.ImageFolder(valid_dir, transform=self.data_transforms['valid_test'])

        
        dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=64, shuffle=True)
        validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size=64)
        
        return dataloaders, validloaders
    def models_architecture(self):
        """
            select the provided model architecture by users with command line;
            the allowed models are :  resnet18, alexnet, vgg16 and vgg19.
            parameters:
                None: None parameters are provided to this function
            return:
                return the number of input features of the model and the model itself.
        """
        resnet18 = models.resnet18(pretrained=True)
        alexnet = models.alexnet(pretrained=True)
        vgg16 = models.vgg16(pretrained=True)
        vgg19 = models.vgg19(pretrained=True)

        models_dict = {'resnet': resnet18, 'alexnet': alexnet, 'vgg16': vgg16, 'vgg19':vgg19}
        in_features = 0
        # Checking for the provided arch in_features
        if self.arch.startswith('vgg'):
            in_features = models_dict[self.arch].classifier[0].in_features
        elif self.arch =='resnet18':
            in_features = models_dict[self.arch].fc.in_features
        elif self.arch == 'alexnet':
            in_features = models_dict[self.arch].classifier[1].in_features
        # return statement
        return models_dict[self.arch], in_features
    
    def build_model(self):
        # TODO: Build and train your network
        model, in_features = self.models_architecture()
        
        for params in model.parameters():
            params.requires_grad = False
            pass

        classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(in_features, self.hidden_units)),
                                               ('relu', nn.ReLU()),
                                               ('dropout', nn.Dropout(p=0.5)),
                                                ('fc2', nn.Linear(self.hidden_units, self.output_units)),
                                               ('log_softmax', nn.LogSoftmax(dim=1))]
                                              ))
        model.classifier = classifier
        return model
    
    def training_model(self):
        """
        function for training the cnn model for the images classifier. 
        Use torchvision model and methodes to train the model        
        """
        
        if self.gpu and torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        
        train_losses, val_losses = [], []
        tot_train_loss = 0
        DELAY = INTERVAL = 12 * 60
        
        model = self.build_model()
        training_loader, valid_loader = self.load_datasets()
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=self.learning_rate )
        model.to(device)
        print('--'*10, 'Training Started', '--'*10)
        for epoch in range(self.epochs): 
            with active_session(DELAY, INTERVAL):

                running_loss = 0
                model.train()

                for inputs, labels in training_loader:

                    inputs, labels = inputs.to(device), labels.to(device)
                    output = model.forward(inputs)
                    train_batch_loss = criterion(output, labels)

                    optimizer.zero_grad()
                    train_batch_loss.backward()
                    optimizer.step()

                    tot_train_loss += train_batch_loss.item()
                    running_loss += train_batch_loss.item()

                else:
                    tot_val_loss = 0
                    val_correct = 0  # Number of correct predictions on the validation set
                    model.eval() 
                    with torch.no_grad():

                        for inputs, labels in valid_loader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            val_output = model.forward(inputs)
                            val_batch_loss = criterion(val_output, labels)

                            tot_val_loss += val_batch_loss.item()

                            ps = torch.exp(val_output)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            val_correct += equals.sum().item()

                        # Get mean loss to enable comparison between train and valid sets
                        train_loss = tot_train_loss / len(training_loader.dataset)
                        val_loss = tot_val_loss / len(valid_loader.dataset)

                        # At completion of epoch
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                        tot_train_loss = 0
                        print("Epoch: {}/{}.. ".format(epoch+1, self.epochs),
                              "Training Loss: {:.4f}.. ".format(train_loss),
                              "Test Loss: {:.4f}.. ".format(val_loss),
                              "Test Accuracy: {:.4f}".format(val_correct / len(valid_loader.dataset)))

        self.save_checkpoint(model, optimizer, training_loader)
        print('--'*10, 'Training Ended', '--'*10)
        return None
    
    def save_checkpoint(self, model, optimizer, training_loader):
        """ The save checkpoint function for saving the trainning model and a bound of others information"""
        checkpoint = {
            'model': model,
            'state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(), # the information about the optimizer
            'class_to_idx': self.image_datasets.class_to_idx,
            'epochs': self.epochs
        }
        torch.save(checkpoint, self.save_dir+'/model_checkpoint.pth')
        return None
        

        