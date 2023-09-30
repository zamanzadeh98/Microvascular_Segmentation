import numpy as np
import glob
import os
import pathlib
import shutil
import shutil
from sklearn.model_selection import train_test_split
import sys

sys.path.append("/content/drive/MyDrive/ZamanPersonalUsage/vesselSegmentation/Codes/Preprocess")

def CopyFunc(Paths, mode):
    """This function copy each images and its corresponding
    mask to train and validation dir
    """
    counter = 0
    # Copying the train images
    for Path_ in Paths:
        try:
            subj = os.path.basename(Path_)

            # Image
            NewImgPath = f"../../Data/{mode}/image"

            # mask 
            #origin
            MaskDir = "../../RawData/masks"

    
            mask_name = "mask_" + subj[:-4] + ".png"
            mask_path = os.path.join(MaskDir, mask_name)
            # New dir
            NewMaskPath = f"../../Data/{mode}/mask"

            shutil.copy(mask_path, NewMaskPath)
            shutil.copy(Path_, NewImgPath)

        # If there was no annotation for an image
        # copy it in a folder for later usage
        except:
            NoAnn = "../../Data/NoAnnotation"
            shutil.copy(Path_, NoAnn)
            counter += 1
            continue
    print(counter)


# if __name__=="__main__":

ImgPaths = glob.glob("../../RawData/train/*.tif")
print(ImgPaths)
TrainImages, ValImages  = train_test_split(ImgPaths, test_size=0.3, random_state=0)


# Train
CopyFunc(TrainImages, "train")
# Validation
CopyFunc(ValImages, "validation")

