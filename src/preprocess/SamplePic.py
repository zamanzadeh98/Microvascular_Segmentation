import cv2
import os
import json
import numpy as np
import matplotlib.pyplot as plt


#When creating masks, it is crucial to ensure that each tissue type is
#  represented by a unique color. This allows the model to differentiate
#  between different tissues during training and prediction. If all 
# tissues are assigned the same color in the masks, it can lead to 
# ambiguity and difficulties in accurately modeling and segmenting 
# the different tissue regions.



jsonPath = "/content/drive/MyDrive/VesselSegmentation/RawData/polygons.jsonl"
with open(jsonPath, "r") as file:

    for counter, line in enumerate(file):
        json_data = json.loads(line)
        
        
        id_ = json_data["id"]
        annotations = json_data["annotations"]
        width, height = 512, 512

        # an empty mast
        mask = np.zeros((width, height), dtype=np.uint8)

        # since there might be multiple tissue in each pictures
        for annot in annotations:

            tissu_label = annot["type"]


            if tissu_label == "blood_vessel": color = 1
            else : color = 0

                
            # print(tissu_label)
            tissue_coordinate = annot["coordinates"][0]

            #Extracting (x,y) from list
            X, Y = zip(*tissue_coordinate)
            # print(X, Y)

            # Creating mask imgs
            ROI = np.column_stack((X, Y))
            # mask[ROI] = color
            cv2.fillPoly(mask, [ROI], color)



        # loading images
        path = "/content/drive/MyDrive/VesselSegmentation/RawData/train"
        img_path = os.path.join(path, id_)
        img_path = img_path + ".tif"
        print(img_path)
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        # Convert BGR image to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #overlay mask image to original image
        overlayed_img = cv2.bitwise_and(img, img, mask=mask)

        fig, ax = plt.subplots(1,3, figsize=(21,7))
        ax[0].imshow(img)
        ax[0].set_title("Original Image")
        ax[1].imshow(mask)
        ax[1].set_title("Mask")
        ax[2].imshow(overlayed_img)
        ax[2].set_title("Overlayed Image")
        plt.axis('off')
        plt.savefig(f"/content/drive/MyDrive/VesselSegmentation/Codes/pictures/{counter}.png")
        # plt.show()
        
        if counter > 5:
            break