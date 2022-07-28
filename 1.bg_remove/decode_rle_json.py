import json
import pycocotools.mask as mask_util
import ast
import cv2



def decoding(rle_json, decode_all):
    with open(rle_json, 'r') as f:
        targets = json.load(f) # targets: list of string (each string expresses json file as dict)
        # print(type(targets), targets, len(targets))
        # print(type(targets[0]))

        # convert targets to list of dictionary
        cnt = 0
        for target in targets:
            target = target.strip('\n') # delete \n
            target = ast.literal_eval(target) # change str to dict

            if decode_all:
                with open(rle_json[:-5] + '_decoded_({}).json'.format(cnt), 'w') as decoded_f:
                    mask = mask_util.decode(target['segmentation'])[:, :]
                    # target['segmentation'] = polygonFromMask(mask) # if you want to get contour, deannotate this line of codes
                    target['segmentation'] = mask.tolist()

                    decoded_f.write(json.dumps(target))
                    cnt += 1
            else:
                if target["category_id"] == 0:  # if target is person
                    with open(rle_json[:-5] + '_decoded_({}).json'.format(cnt), 'w') as decoded_f:
                        mask = mask_util.decode(target['segmentation'])[:, :]
                        # target['segmentation'] = polygonFromMask(mask) # if you want to get contour, deannotate this line of codes
                        print(mask, type(mask))
                        target['segmentation'] = mask.tolist()

                        decoded_f.write(json.dumps(target))
                        cnt += 1



# The all codes below reference to
# https://github.com/cocodataset/cocoapi/issues/476

'''  
maskedArr = mask_util.decode(compactRLESegmentation)
area = float((maskedArr > 0.0).sum())
def polygonFromMask(maskedArr):
  # adapted from https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
  contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  segmentation = []
  valid_poly = 0
  for contour in contours:
  # Valid polygons have >= 6 coordinates (3 points)
     if contour.size >= 6:
        segmentation.append(contour.astype(float).flatten().tolist())
        valid_poly += 1
  if valid_poly == 0:
     raise ValueError
  return segmentation
'''

# This function is for transferring binary masks to contour coordinates
def polygonFromMask(maskedArr):  # https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
    # please reference the site below to use findCountours function properly
    # https://opencv-python.readthedocs.io/en/latest/doc/15.imageContours/imageContours.html
    contours, _ = cv2.findContours(maskedArr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

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

