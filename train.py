'''
Author: Asheem Chhetri
Version: 2
Date: 27 dec-2018
Train a new network on a data set with train.py

Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg13"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu
'''
import argparse
import os
import sys
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import time
import json

def get_input_args():
    parser =  argparse.ArgumentParser()
    
    parser.add_argument('data_directory', help='location of input source for images')
    parser.add_argument('--save_dir', help='loaction to save checkpoint', default='checkpoints')
    parser.add_argument('--arch', type=str, default='densenet121', help='model architecture to use: <densenet121> or <vgg16>')
    parser.add_argument('--gpu', action='store_true', help='use GPU')
    parser.add_argument('--epochs', type=int, default='6', help='number of epochs to use')
    parser.add_argument('--hidden_units', nargs='+', type=int, default=[500, 200], help='hidden layers <only two layers allowed>')
    parser.add_argument('--learn_rate', type=float, default='0.001', help='learn rate')
    parser.add_argument('--drop_rate', type=float, default='0.5', help='drop out rate')
    parser.add_argument('--verification_loader_data', type=str, default='validation', help='can be either <valdiation> or <test>')
    return parser.parse_args()

def data_load(image_source):
    # Loading the data
    data_dir = image_source
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


    test_transforms = transforms.Compose([transforms.Resize(256),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # ----------------------------------------------------------------------------------------------
    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    # ----------------------------------------------------------------------------------------------
    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size= 64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=32, shuffle=False)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)
    
    return train_data, validation_data, test_data, trainloader, validationloader, testloader

def model_building(arch_name, hidden_layer, learn_rate, drop_out):
    input_size = {'densenet121':1024, 'vgg16': 25088}
    output_size = 102
    model = None
    if(arch_name == 'densenet121'):
        model = models.densenet121(pretrained=True)
    elif(arch_name == 'vgg16'):
        model = models.vgg16(pretrained=True)
    else:
        print("This architecture {} is not supported. Please retry with either: densenet121 or vgg16".format(arch_name))
        return None
    # Freezing the parameters
    for param in model.parameters():
        param.require_grad = False
    
    #     print(input_size[arch_name])
    #     print("{}: {},{}".format(hidden_layer, hidden_layer[0], hidden_layer[1]))
    #     print(type(hidden_layer[0]))
    #     print(learn_rate)
    #     print(drop_out)
    print("\n------------------------------------------------\n")
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size[arch_name], hidden_layer[0])),
                          ('relu', nn.ReLU()),
                          ('dropout',  nn.Dropout(drop_out)),
                          ('fc2', nn.Linear(hidden_layer[0], hidden_layer[1])),
                          ('relu', nn.ReLU()),
                          ('fc3', nn.Linear(hidden_layer[1], output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

    model.classifier = classifier
    return model, classifier

def train_network(device_source, model, epochs, criterion, optimizer, training_data, test_data, valid_flag):
    print("We will be using this device: {}\n".format(device_source.upper()))
    print("Training started...\n")
    trainloader = training_data
    data_to_test = test_data
    loss_set = []
    accuracy_set =[]
    
    start = time.time()
    # print_every affects the print statement(i.e. 50 => 2 line per epoch, 10 => 10 line per epoch, etc. (100/print_every))
    print_every = 50
    steps = 0

    # GPU mode enabled based on device_source
    model.to(device_source)

    for e in range(epochs):
        running_loss = 0
        model.train()
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1

            inputs, labels = inputs.to(device_source), labels.to(device_source)
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                # Make sure network is in eval mode for inference
                model.eval()

                # Turn off gradients for validation, saves memory and computations
                with torch.no_grad():
                    test_loss, accuracy = validation(device_source, model, data_to_test, criterion)
                loss_set.append(test_loss/len(data_to_test))
                accuracy_set.append(accuracy/len(data_to_test))
                if valid_flag:
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Validation Loss: {:.3f}.. ".format(test_loss/len(data_to_test)),
                          "Validation Accuracy: {:.3f}".format(accuracy/len(data_to_test)))    

                else:
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Test Loss: {:.3f}.. ".format(test_loss/len(data_to_test)),
                          "Test Accuracy: {:.3f}".format(accuracy/len(data_to_test)))

                running_loss = 0
                model.train()
    end = time.time()
    diff = end - start
    print("\nTotal time taken: {:.0f} minutes {:.0f} seconds".format(diff//60, diff%60))
    return loss_set, accuracy_set

def validation(device_source, model, data_to_test, criterion):
    test_loss = 0
    accuracy = 0
    model.to(device_source)
    for images, labels in data_to_test:
        images, labels = images.to(device_source), labels.to(device_source)

        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy

def save_checkpoint(arch, input_size, epochs, model, optimizer, classifier, criterion, learn_rate, train_data, save_location):
    checkpoint = {
              'input_size': 1024,
              'output_size': 102,
              'arch': arch,
              'classifier': classifier,
              'state_dict': model.state_dict(),
              'epochs': epochs,
              'optimizer': optimizer.state_dict(),
              'learn_rate': learn_rate,
              'class_to_idx': train_data.class_to_idx,
             }

    torch.save(checkpoint, save_location+'/checkpoint.pth')

def main():
    with open('log.txt', 'w') as file:
        print("Writing to log file commenced...\n\n")
        file.write("We will record prelim data here for self check.\n\n")
        input_args = get_input_args()
        accuracy_record = []
        loss_record = []
        
        input_size = {'densenet121':1024, 'vgg16': 25088}
        save_dir = input_args.save_dir
        data_dir = input_args.data_directory
        gpu_use = input_args.gpu
        device = 'cpu'
     
        if gpu_use and torch.cuda.is_available():
            device = 'cuda'
        
        print("\n----------Architecture Information-------------\n")
        print("Image source dir name: ",input_args.data_directory)
        print("Checkpoint location dir name: ",save_dir) 
        print("Model Architecture being used: ",input_args.arch)
        print("GPU enabled: ",input_args.gpu)
        print("Is Cuda available: ",torch.cuda.is_available())
        print("Device going to be used: ",device.upper())
        print("EPOCHS value: ",input_args.epochs)
        print("Hidden Unit[layer1,layer2]: ",input_args.hidden_units)
        print("Learn Rate: ",input_args.learn_rate)
        print("Drop Out Rate: ",input_args.drop_rate)
        print("Data set to verify: ",input_args.verification_loader_data)
        print("\n------------------------------------------------\n")

        train_data, validation_data, test_data, trainloader, validationloader, testloader = data_load(input_args.data_directory)
        if input_args.verification_loader_data == "validation":
            data_to_check = validationloader
            data_flag = True
        elif input_args.verification_loader_data == "test":
            data_to_check = testloader
            data_flag = False
        else:
            print(">>>>> Data provided to be tested does NOT belong to test_datset or validation_dataset <<<<<\nPlease retry with correct dataset")
            sys.exit()
        # Creating a directory for checkpoint
        if not os.path.exists(save_dir): 
            print("Creating directory for checkpoint.\n Dir name: {}".format(save_dir))
            os.makedirs(save_dir)

        # Building the model
        model, classifier = model_building(input_args.arch, input_args.hidden_units, input_args.learn_rate, input_args.drop_rate)
        # print(model)

        # Setting up criterion and optimzer
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr = input_args.learn_rate)

        # Training the model
        if device == 'cuda':
            print("\nTraining commenced using CUDA as our device...\n")
            loss_record,accuracy_record = train_network('cuda', model, input_args.epochs, criterion, optimizer, trainloader, data_to_check, data_flag)
        else:
            print("\nTraining commenced using CPU as our device...\n")
            loss_record, accuracy_record = train_network('cpu', model, input_args.epochs, criterion, optimizer, trainloader, data_to_check, data_flag)
        
        # doing a sanity check on test or validation dataset, given what user provided for training
        if data_flag:
            model.eval()
            with torch.no_grad():
                new_loss_data, new_accuracy_data = validation(device, model, testloader, criterion)
        else:
            model.eval()
            with torch.no_grad():
                new_loss_data, new_accuracy_data = validation(device, model, validationloader, criterion)
        # Log file for my own record
        file.write('-------------Loss Record---------------\n\n')
        for i in loss_record: file.write(str(i)+'\n')
        file.write('\n-----------Accuracy Record-------------\n')
        for i in accuracy_record: file.write(str(i)+'\n')
        
        if data_flag:
            print("\n-----------Model Accuracy-------------\n")
            file.write("\n-----------Model Accuracy-------------\n")
            print("Validation Loss: {:.4f}\nAccuracy: {:.4f}\nAccuracy(in %): {:.2f}%".format(loss_record[-1], accuracy_record[-1], accuracy_record[-1] * 100))
            print("\n------------Test Result---------------\n")
            print("Test Accuracy: {:.2f} %".format((new_accuracy_data/len(testloader)) * 100))
            file.write("\nValidation Loss: {:.4f}\nAccuracy: {:.4f}\nAccuracy(in %): {:.2f}%".format(loss_record[-1], accuracy_record[-1], accuracy_record[-1] * 100))
            file.write('\n\n-----------Test Result-------------\n')
            file.write("Test Accuracy: {:.2f} %".format((new_accuracy_data/len(testloader)) * 100))
        else:
            print("\n------------Model Accuracy--------------\n")
            file.write("\n-----------Model Accuracy-------------\n")
            print("Test Loss: {:.4f}\nAccuracy: {:.4f}\nAccuracy(in %): {:.2f}%".format(loss_record[-1], accuracy_record[-1], accuracy_record[-1] * 100))
            print("\n----------Validation Result-------------\n")
            print("Validation Accuracy: {:.2f} %".format((new_accuracy_data/len(validationloader)) * 100))
            file.write("\nTest Loss: {:.4f}\nAccuracy: {:.4f}\nAccuracy(in %): {:.2f}%".format(loss_record[-1], accuracy_record[-1], accuracy_record[-1] * 100))
            file.write('\n\n-----------Validation Result-------------\n')
            file.write("Validation Accuracy: {:.2f} %".format((new_accuracy_data/len(validationloader)) * 100))
         
        # Log file writing ends
        
        print("\n---------Creating Checkpoint-----------\n")
        save_checkpoint(input_args.arch, input_size[input_args.arch], input_args.epochs, model, optimizer, classifier, criterion, input_args.learn_rate, train_data, save_dir)
if __name__ == "__main__":
    main()