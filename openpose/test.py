from PIL import Image
import glob
import os
import json
import numpy as np
import cv2

# Linux 서버 세팅 이후 아래 경로들 변경 필요!!!!
openpose_bin = str(os.getcwd()) + "/" + "openpose/openpose_build/build/examples/openpose/openpose.bin"
openpose_dir = str(os.getcwd()) + "/" + "openpose/openpose_build"

input_path = str(os.getcwd()) + "/" + "output/background_step1/"
output_img = str(os.getcwd()) + "/" + "inputs/image/"
output_pose = str(os.getcwd()) + "/" + "inputs/openpose-img/"
output_posejson = str(os.getcwd()) + "/" + "inputs/openpose-json/"

input_org_img = str(os.getcwd()) + "/" + "inputs/org_img_gs/"
output_resized_img = str(os.getcwd()) + "/" + "inputs/org_img/"

if not os.path.exists(output_img):
    os.makedirs(output_img)
if not os.path.exists(output_pose):
    os.makedirs(output_pose)
if not os.path.exists(output_posejson):
    os.makedirs(output_posejson)

def run_openpose():
    os.chdir(openpose_dir)
    os.system(f'{openpose_bin} --image_dir {output_img} --write_json {output_posejson} --display 0 --render_pose 0')
    os.system(f'{openpose_bin} --image_dir {output_img} --write_images {output_pose} --display 0 --disable_blending')

def crop_img():
    original_files = glob.glob(input_path+"*_background_removed.jpg")
    for file in original_files:
        print("Resizing Image(512x512) :" + str(file))
        img = Image.open(file)
        set_height = 512
        if (img.height < set_height):
            multiplier = float(set_height / img.height)
            set_width = int(img.width / multiplier)
        else:
            multiplier = float(img.height / set_height)
            set_width = int(img.width / multiplier)
        resized = img.resize((set_width, set_height))

        pix = np.array(resized)
        print(pix.shape[0], pix.shape[1])
        x_min = pix.shape[1]
        x_max = 0
        y_min = pix.shape[0]
        y_max = 0

        bg = np.array([230, 230, 230])

        for y in range(pix.shape[0]):
            for x in range(pix.shape[1]):
                if np.array_equal(pix[y][x], bg):
                    continue
                else:
                    if x < x_min:
                        x_min = x
                    if x > x_max:
                        x_max = x
                    if y < y_min:
                        y_min = y
                    if y > y_max:
                        y_max = y

        x_center = (x_min + x_max) / 2
        x_min = x_center - ((y_max - y_min) / 2)
        x_max = x_center + ((y_max - y_min) / 2)
        resized = resized.crop((x_min, y_min, x_max, y_max)).resize((512, 512), Image.LANCZOS)
        resized.save(output_img + file.split('/')[-1])
        print(f"{x_min} {x_max} {y_min} {y_max}")

def resize_image():
    original_files = glob.glob(input_org_img+"*.jpg")
    for file in original_files:
        print("Resizing Image(512x512) :" + str(file))
        img = Image.open(file)
        set_height = 512
        if (img.height < set_height):
            multiplier = float(set_height / img.height)
            set_width = int(img.width / multiplier)
        else:
            multiplier = float(img.height / set_height)
            set_width = int(img.width / multiplier)
        resized = img.resize((set_width, set_height))
        print(set_width, set_height)
        resized.save(output_resized_img + file.split('/')[-1])

if __name__ == "__main__":
    crop_img()
    run_openpose()
    # resize_image()
