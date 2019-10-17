import torch
import torchvision

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    '''
        Since we don't have labels for test images,
        we write class, that is inherited from ImageFolder,
        and returns also the path to the image. We will use the test_img_paths
        later during submission 
    '''
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path
