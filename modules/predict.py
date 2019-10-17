import numpy as np
from tqdm import tqdm

import torch

def predict(test_dataloader, model, device):
    predictions = []
    test_img = []
    print("Make predictions for test: ")
    for inputs, labels, paths in tqdm(test_dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            preds = model(inputs)
        predictions.append(torch.nn.functional.softmax(preds, dim=1)[:,1].data.cpu().numpy())
        test_img.extend(paths)
    return np.concatenate(predictions), test_img
