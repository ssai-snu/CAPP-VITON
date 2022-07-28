import json
import pycocotools.mask as mask_util
import cv2
from PIL import Image, ImageDraw
import os
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
import numpy as np

def get_encoded_jsons(id, predictions):
    instances = predictions["instances"].to('cpu')
    json_list = instances_to_coco_json(instances, id)

    return json_list



def get_binary_mask(encoded_json_dict):
    mask = mask_util.decode(encoded_json_dict['segmentation'])[:, :]

    return mask



def get_binary_contour(instance_dict, binary_mask):
    instance_dict2 = instance_dict.copy()
    instance_dict2['segmentation'] = polygonFromMask(binary_mask)

    return instance_dict2

def get_binary_image(binary_mask):
    il = len(binary_mask)
    jl = len(binary_mask[0])

    for i in range(il):
        for j in range(jl):
            if binary_mask[i][j] == 1:
                binary_mask[i][j] = 255


    img = Image.fromarray(np.uint8(binary_mask)).convert('RGB')
    return img



# This function is for transferring binary masks to contour coordinates
def polygonFromMask(maskedArr):  # https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
    # please reference the site below to use findCountours function properly
    # https://opencv-python.readthedocs.io/en/latest/doc/15.imageContours/imageContours.html
    contours, _ = cv2.findContours(maskedArr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # print("1", maskedArr, type(maskedArr), maskedArr.shape)
    segmentation = []
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())
    RLEs = mask_util.frPyObjects(segmentation, maskedArr.shape[0], maskedArr.shape[1])
    RLE = mask_util.merge(RLEs)
    # RLE = mask.encode(np.asfortranarray(maskedArr))
    area = mask_util.area(RLE)
    [x, y, w, h] = cv2.boundingRect(maskedArr)

    return segmentation[0]  # , [x, y, w, h], area