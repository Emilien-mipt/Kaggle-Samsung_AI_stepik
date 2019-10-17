import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import shutil
import time
from tqdm import tqdm
import copy

import torch
import torchvision
from torchvision import models

from modules.train import train
from modules.show import show_input
from modules.predict import predict
from modules.create_validation import create_val
from modules.form_datasets import form_dataset
from modules.submit import make_submit

def main():
    print(os.listdir("./platesv2/"))
# Any results you write to the current directory are saved as output.
    data_root = './platesv2/plates/'
    print(os.listdir(data_root))
# Create directory "val_dir" for validation
    train_dir = 'train'
    val_dir = 'val'
    class_names = ['cleaned', 'dirty']
    for dir_name in [train_dir, val_dir]:
        for class_name in class_names:
            os.makedirs(os.path.join(dir_name, class_name), exist_ok=True)
# Move each 6-th photo from "train_dir" to "val_dir" for validation
    create_val(data_root, train_dir, val_dir, class_names, 6)
# Apply augmentations and form datasets for training and validation
    train_dataset, val_dataset = form_dataset(test = False)
# We will feed the net with data in the form of batches
    batch_size = 8
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size)
# Let's have a look at the first batch
    X_batch, y_batch = next(iter(train_dataloader))
    for x_item, y_item in zip(X_batch, y_batch):
        show_input(x_item, title=class_names[y_item])
# Model - pretrained ResNet18, trained on ImageNet
    model = models.resnet18(pretrained=True)
# Disable grad for all conv layers
    for param in model.parameters():
        param.requires_grad = False
    print("Output of ResNet18 before FC layer, that we add later: ", model.fc.in_features)
# Add FC layer with 2 outputs: cleaned or dirty
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
# Put model on GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
# Loss function - binary Cross-Entropy
    loss = torch.nn.CrossEntropyLoss()
# Optimization method - Adam
    optimizer = torch.optim.Adam(model.parameters(), amsgrad=True, lr=1.0e-3)
# Decay LR by a factor of 0.1 every 7 epochs
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
# Training
    print("Begin training: ")
    train(model, train_dataloader, val_dataloader, loss, optimizer, scheduler, device, num_epochs = 10)

# Now let's make predictions
    test_dir = 'test'
# Make additional directory in test to let ImageFolder identify pictures in it and create iterator
    shutil.copytree(os.path.join(data_root, 'test'), os.path.join(test_dir, 'unknown'))
    test_dataset = form_dataset(test = True)
#Form batches on test
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size)
    model.eval()
# Make predictions
    test_predictions = []
    test_img_paths = []
    test_predictions, test_img_paths = predict(test_dataloader, model, device)
# Show predictions for the test
    inputs, labels, paths = next(iter(test_dataloader))

    for img, pred in zip(inputs, test_predictions):
        show_input(img, title=pred)
# Submit
    print("Making submission")
    make_submit(test_img_paths, test_predictions)


if __name__ == '__main__':
    main()
