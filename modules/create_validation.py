import os
from tqdm import tqdm
import shutil

def create_val(data_root, train_dir, val_dir, class_names, number):
    for class_name in class_names:
        source_dir = os.path.join(data_root, 'train', class_name)
        print("Class {}: copy every {} - th image from {} directory to {} directory:".format(class_name, number, train_dir, val_dir))
        for i, file_name in enumerate(tqdm(os.listdir(source_dir))):
            if i % number != 0:
                dest_dir = os.path.join(train_dir, class_name)
            else:
                dest_dir = os.path.join(val_dir, class_name)
            shutil.copy(os.path.join(source_dir, file_name), os.path.join(dest_dir, file_name))
