'''
Predict flower name from an image with predict.py along with the probability of that name. That is, you'll pass in a single image /path/to/image and return the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint
Options:
Return top KK most likely classes: python predict.py input checkpoint --top_k 3
Use a mapping of categories to real names: python predict.py input checkpoint --category_names cat_to_name.json
Use GPU for inference: python predict.py input checkpoint --gpu

Eg: python predict.py flowers/test/58/image_02719.jpg
Eg: python predict.py flowers/test/10/image_07090.jpg
Eg: python predict.py flowers/test/58/image_02719.jpg --gpu
'''
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import OrderedDict
from torchvision import datasets, transforms, models
from PIL import Image
import time
import json

def get_input_args():
    parser =  argparse.ArgumentParser()
    
    parser.add_argument('data_directory', type=str, help='image location to test')
    parser.add_argument('--load_dir', help='loaction to retreive checkpoint from', default='checkpoints')
    parser.add_argument('--top_k', type=int, default=5, help='K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Mapping of categories to real name')
    parser.add_argument('--gpu', action='store_true', help='use GPU')

    return parser.parse_args()
def retreive_checkpoint(filepath, device):
    '''
    One issue that someone might face, how to open checkpoint made on gpu on a cpu device
    For more info and code reference for this issue, please refer to this pytorch community site
    https://discuss.pytorch.org/t/on-a-cpu-device-how-to-load-checkpoint-saved-on-gpu-device/349
    '''
    if device == "cuda":
        print("\nUsing CUDA as a device to retreive checkpoint...\n")
        checkpoint = torch.load(filepath)
    else:
        print("\nUsing CPU as a device to retreive checkpoint...\n")
        checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer = checkpoint['optimizer_dict']
    epochs = checkpoint['epochs']
    learn_rate = checkpoint['learn_rate']
    criterion = checkpoint['criterion']
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
#     print(img)
#     print(type(img))
#     print('\n-------------------------------------------------\n')
    img_mods = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])])
    # Apply transformations to the image
    img = img_mods(img).numpy()
    return img
'''
Note: This function will not work here or any image display. Still leaving it here to run it on local machine later.
'''
def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def invert_dict(model):
    inverted_dict = {}
    for k,v in model.class_to_idx.items():
        inverted_dict[v] = k
    return inverted_dict

def predict(device, image_path, model, topk, inverted_dict):
    '''
    Predict the class (or classes) of an image using a trained deep learning model.
    '''
    print("Device we are using is: {}\n".format(device.upper()))
    if device == 'cuda:0': model.to('cuda')
    else: model.to('cpu')
    
    img_data = torch.from_numpy(process_image(image_path)).unsqueeze_(0).float()
    model.eval()
    # Calculate the class probabilities (softmax) for img
    with torch.no_grad():
        output = model.forward(img_data)

    ps = torch.exp(output)
    prob, classes = ps.topk(topk)
    # print(prob)
    # print('Classes tensor was converted to numpy array, to help indexing the inverted_dictionary.')
    # print(classes)
    # Initial value is tensor, we must convert classes tensor to numpy, so we can index the dictionary using it.
    classes = classes.numpy()[0]
    prob = prob.numpy()[0]
    p_val = [i for i in prob]
    c_val = [inverted_dict[i] for i in classes]
    # print(c_val)
    return p_val, c_val

def sanity_check():
    pass
def main():
    input_args = get_input_args()
    with open(input_args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    img_location = input_args.data_directory
    checkpoint_loc = input_args.load_dir + '/checkpoint.pth'
    gpu_use = input_args.gpu
    device = 'cpu'
    if gpu_use and torch.cuda.is_available():
        device = 'cuda'
    
    print("\n-----------Prediction File Information-------------\n")
    print("Path to image: ",input_args.data_directory)
    print("Checkpoint location: ",input_args.load_dir)
    print("GPU enabled: ",input_args.gpu)
    print("Is Cuda available: ",torch.cuda.is_available())
    print("Device going to be used: ",device.upper())
    print("TOPK value: ",input_args.top_k)
    print("Category File name: ",input_args.category_names)
    print("\n---------------------------------------------------\n")

        
    # Retreiving the checkpoint
    model = retreive_checkpoint(checkpoint_loc, device)
    # print(model)
    # Only enable to view what was loaded from checkpoint
    # after_loading = model.state_dict().keys()
    # print(after_loading)
    # Opening an image using matplotlib. Won't work here
    # with Image.open(img_location) as image:
    #     print(image.size)
    #    plt.imshow(image)
    # Opening image using process_image
    # imshow(process_image(img_location))
    # Getting the inverted dictionary, to get mapping form index to class
    inverted_dictionary = invert_dict(model)
    # print(inverted_dictionary)
    # Prediction
    print("\n-------------Performing Prediction--------------------\n")
    prob,classes = predict(device, img_location, model, input_args.top_k, inverted_dictionary)
    flower_name = [cat_to_name[x] for x in classes]
    
    print("Probability:\t{}".format(prob))
    print("Classes:\t\t{}".format(classes))
    print("Names:\t\t{}".format(flower_name))
    print("\n\t\t\t   Tabular View\n")
    result = zip(prob, classes, flower_name)
    print("{}{}{}".format(' ',"-"*(35+20+14+12),' '))
    print("|%-35s|%-20s|%-14s|%-8s|"% ("Flower Name","Probability (4dp *)","Probability (%)","Class"))
    print("{}{}{}".format('|',"-"*(35+20+14+12),'|'))
    for p,c,f in result:
        print("|%-35s|%-20.4f|%-15.2f|%-8s|"% (f.title(),p,(p*100),c))
    print("{}{}{}".format(' ',"-"*(35+20+14+12),' '))
    print("\n*4dp: 4 decimal point")
    # print(classes)
    # print(flower_name)
    print("\n------------------Results-----------------------------------\n")
    print("Actual Flower Class family, based on input path:\t{}\n".format(cat_to_name[img_location.split('/')[2]].title()))
    print("Predicted Flower Class family:\t\t{}\n".format(flower_name[0].title()))
    print("\n------------------Finish------------------------------------\n")
if __name__ == '__main__':
    main()