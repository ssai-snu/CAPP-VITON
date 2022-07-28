from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog
import os

#     category, try to keep
# CLASS_NAMES = ["onepiece", "blouse", "skirt", "coat", "jumper", "longpants", "shortpants"]
CLASS_NAMES = ['blouse', 'cardigan', 'coat', 'hoodie', 'jacket', 'jumper', 'jumpsuit',
               'knit', 'longpants', 'onepiece', 'shirt', 'shortpants', 'skirt', 'tshirt']

#
DATASET_ROOT = r'./datasets/gs_14cgy/'
# folder path
ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
# Training picture path
TRAIN_PATH = os.path.join(DATASET_ROOT, 'custom_train_cocofy')
# Test image path
VAL_PATH = os.path.join(DATASET_ROOT, 'custom_val_cocofy')
#
TRAIN_JSON = os.path.join(ANN_ROOT, 'custom_train_cocofy.json')
#
VAL_JSON = os.path.join(ANN_ROOT, 'custom_val_cocofy.json')
# Test the label file
# VAL_JSON = os.path.join(ANN_ROOT, 'test.json')

register_coco_instances("my_train", {}, TRAIN_JSON, TRAIN_PATH)
MetadataCatalog.get("my_train").set(thing_classes=CLASS_NAMES,
#                                    evaluator_type='coco',  # specify the evaluation method
                                    json_file=TRAIN_JSON,
                                    image_root=TRAIN_PATH)
register_coco_instances("my_val", {}, VAL_JSON, VAL_PATH)
MetadataCatalog.get("my_val").set(thing_classes=CLASS_NAMES,
#                                  evaluator_type='coco',  # specify the evaluation method
                                  json_file=VAL_JSON,
                                  image_root=VAL_PATH)