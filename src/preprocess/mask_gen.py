from tkinter.messagebox import YES
import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt


# When creating masks, it is crucial to ensure that each tissue type is
# represented by a unique color. This allows the model to differentiate
# between different tissues during training and prediction. If all 
# tissues are assigned the same color in the masks, it can lead to 
# ambiguity and difficulties in accurately modeling and segmenting 
# the different tissue regions.


jsonPath = "/content/drive/MyDrive/ZamanPersonalUsage/vesselSegmentation/RawData/polygons.jsonl"
with open(jsonPath, "r") as file:
    for line in file:
        json_data = json.loads(line)
        
        id_ = json_data["id"]
        annotations = json_data["annotations"]
        width, height = 512, 512

        # Create an empty mask
        mask = np.zeros((width, height), dtype=np.uint8)

        # Iterate over annotations and fill the mask
        for annot in annotations:
            tissue_label = annot["type"]
            print(tissue_label)
            if tissue_label == "blood_vessel":
                color = 1
            else: color = 0

            tissue_coordinate = annot["coordinates"][0]
            X, Y = zip(*tissue_coordinate)
            ROI = np.column_stack((X, Y))
            cv2.fillPoly(mask, [ROI], color)
            

        # Save the mask image
        mask_file_path = f"/content/drive/MyDrive/ZamanPersonalUsage/vesselSegmentation/RawData/masks/mask_{id_}.png"
        cv2.imwrite(mask_file_path, mask)
    
        

        