import torch
import os
import numpy as np
from PIL import Image

class dataset_class(torch.utils.data.Dataset):
    """The dataset_class is wraper around our dataset"""

    def __init__(self, ImagesPath, transform=None):
        self.ImagesPath = ImagesPath
        self.transform = transform

    def __len__(self):
        return len(self.ImagesPath)


    def __getitem__(self, index):
        ImgPath = self.ImagesPath[index]
        img_name = os.path.basename(ImgPath)
        BasePath = os.path.join(os.path.dirname(os.path.dirname(ImgPath)), "mask")
        mask_name = "mask_" + img_name[:-3] + "png"
        MaskPath = os.path.join(BasePath, mask_name)

        img = Image.open(ImgPath).convert("RGB")
        mask = Image.open(MaskPath) # Convert to grayscale

        # Resize the image and mask to the desired size
        img = img.resize((256, 256), Image.ANTIALIAS)
        mask = mask.resize((256, 256), Image.ANTIALIAS)

        mask = torch.from_numpy(np.array(mask))

        if self.transform:
            img = self.transform(img)

        return img, mask
        