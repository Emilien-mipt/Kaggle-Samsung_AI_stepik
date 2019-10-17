import torch
import torchvision
from torchvision import transforms
from modules.ImageFolderWithPaths import ImageFolderWithPaths

def form_dataset(test = False):
    # Apply augmentations
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224), # Cut the part of the image in the form of rectangular with size 224
        transforms.RandomHorizontalFlip(), # Flip it horizontally
        transforms.ToTensor(), # Turn to PyTorch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize: we want to bring images to the form of images on which ResNet was trained

    ])
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)), # Resize pictures to format 224*224.
        #224х224 - the size of ImageNet images, on which ResNet was pretrained
        transforms.ToTensor(), # Transform to Pytorch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize, as it was described above
    ])
    if not test:
        '''
          We iterate over the images and get the pairs: image - label
          Label is the name of the directory in which the image is stored
          We would like to feed tensors into the neural network, so we apply transformations (train / val_transforms)
        '''
        print("Form the dataset for training and validation")
        train_dataset = torchvision.datasets.ImageFolder('train', train_transforms)
        val_dataset = torchvision.datasets.ImageFolder('val', val_transforms)
        return train_dataset, val_dataset
    else:
        test_dataset = ImageFolderWithPaths('test', val_transforms)
        return test_dataset
