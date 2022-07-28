# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import random
import numpy as np

from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

from predictor import VisualizationDemo
from adet.config import get_cfg

#############################
import fruitsnuts_data
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
import json
from background_removal import remove_background
from get_json import get_encoded_jsons, get_binary_mask, get_binary_contour, get_binary_image
from PIL import Image
from insert_background import insert_bg
#############################
# constants
WINDOW_NAME = "COCO detections"


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)

    # Modified by yjcho.
    # If you want to test my custom dataset, you should deannotate this single line of codes.
    # cfg.DATASETS.TEST = ("my_val",)

    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.MEInst.INFERENCE_TH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")

    parser.add_argument("--step", default="step1")
    parser.add_argument(
        "--config-file",
        default="BlendMask/R_101_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument("--bg", default="inputs/background.txt", help="Background information")
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
             "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.3,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    print(parser)

    return parser


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)

    if args.input:
        if os.path.isdir(args.input[0]):
            args.input = [os.path.join(args.input[0], fname) for fname in os.listdir(args.input[0])]
        elif len(args.input) == 1:
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"

        cnt = 0
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
            img = read_image(path, format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: detected {} instances in {:.2f}s".format(
                    path, len(predictions["instances"]), time.time() - start_time
                )
            )

            if args.output:
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)

                ############## The codes belows are modified by yjcho.
                # It includes background removal, encoded json, and decoded json files.
                coco_format_list_of_dict = get_encoded_jsons(cnt, predictions)
                # print(type(coco_format_list_of_dict))

                if len(coco_format_list_of_dict) != 0:  # At least 1 instance should be detected...
                    # background removal

                    detected_object_classes = predictions["instances"].pred_classes.tolist()
                    cnt2 = 0
                    for i in range(len(coco_format_list_of_dict)):
                        encoded_dict = coco_format_list_of_dict[i]

                        ##### if you want to get items from certain category, use category_id and continue
                        if encoded_dict["category_id"] != 0:
                            continue
                        #####

                        background_removed_image = remove_background(img, predictions, i)
                        encoded_dict = get_encoded_jsons(cnt, predictions)[i]
                        mask = get_binary_mask(encoded_dict)

                        OUTPUT_FILE = os.path.splitext(out_filename)[0]

                        # write files
                        # if you want a certain file, please uncomment
                        cv2.imwrite(OUTPUT_FILE + '_cnt_' + str(i) + '_background_removed.jpg', background_removed_image)

                        #with open(OUTPUT_FILE + '_encoded_({}).json'.format(cnt2), 'w') as encoded_json:
                        #    json.dump(encoded_dict, encoded_json)

                        #    with open(OUTPUT_FILE + '_binary_mask_({}).json'.format(cnt2), 'w') as binary_json:
                        #        json.dump(mask.tolist(), binary_json)

                        #    with open(OUTPUT_FILE + '_contour_({}).json'.format(cnt2), 'w') as contour_json:
                        #        json.dump(get_binary_contour(encoded_dict, mask), contour_json)

                        if args.step == 'step1':
                            continue

                        #image = get_binary_image(mask)
                        #image.save(OUTPUT_FILE + '_binary_image_({}).jpg'.format(cnt2))
                        #bg_image_list = []
                        #with open ('bg_images/target.txt') as f:
                        #    lines = f.readlines()
                        #    for line in lines:
                        #        bg_image_list.append(line.replace('\n',''))

                        #bg_image = random.choice(bg_image_list)
                        #print(bg_image)
                        # bg_image = args.bg

                        #x = cv2.imread(bg_image, cv2.IMREAD_COLOR)

                        bg_f = open(args.bg, 'r')
                        bg_f_lines = bg_f.readlines()

                        for line in bg_f_lines:
                            bg_img_f, bg_pos_f = line.strip().split()
                            print(bg_pos_f)

                            bg_pos = open(bg_pos_f, 'r')
                            bg_pos_line = bg_pos.readline()
                            y_min, y_max, x = bg_pos_line.strip().split()

                            img_bg = read_image(bg_img_f, format="BGR")
                            img_bg = cv2.resize(img_bg, dsize=(1920, 1080), interpolation=cv2.INTER_AREA)
                            mask = predictions["instances"].pred_masks
                            mask = mask.cpu().numpy()
                            mask = np.array(mask, dtype=np.uint8)
                            bg_modified = insert_bg(img_bg, background_removed_image, mask[i], y_min=int(y_min), y_max=int(y_max), x=int(x))
                            cv2.imwrite(f"{OUTPUT_FILE}_{bg_img_f.split('/')[-1].split('.')[0]}_bg_insert.jpg", bg_modified)
                        cnt2 += 1

                cnt += 1
                ##########


            else:
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit