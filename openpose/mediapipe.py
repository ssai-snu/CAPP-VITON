import numpy as np
import time

import json
import cv2
import mediapipe as mp

BG_COLOR = (192, 192, 192)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

input_path = str(os.getcwd()) + "/" + "output/background_step1/"
output_img = str(os.getcwd()) + "/" + "inputs/image/"
output_pose = str(os.getcwd()) + "/" + "inputs/openpose-img/"
output_posejson = str(os.getcwd()) + "/" + "inputs/openpose-json/"

if not os.path.exists(output_img):
    os.makedirs(output_img)
if not os.path.exists(output_pose):
    os.makedirs(output_pose)
if not os.path.exists(output_posejson):
    os.makedirs(output_posejson)

def get_pose(file):
    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.5) as pose:
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        # Convert the BGR image to RGB before processing
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if not results.pose_landmarks:
            continue

        annotated_image = image.copy()
        # Draw segmentation on the image
        # To improve segmentation on the image
        # Bilateral filter to 'results.segmentation_mask' with 'image'
        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
        bg_image = np.zeros(image.shape, dtype=np.uint8)
        # Draw pose landmarks on the image
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
        # Plot pose world landmarks
        mp_drawing.plot_landmarks(
            results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)


        for landmarks in results.pose_landmarks:
            keypoints = [
                landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.NOSE].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_INNER].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_INNER].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_INNER].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_OUTER].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_OUTER].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_OUTER].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_INNER].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_INNER].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_INNER].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_OUTER].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_OUTER].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_OUTER].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.MOUTH_LEFT].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.MOUTH_RIGHT].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_PINKY].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_PINKY].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_INDEX].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_INDEX].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_THUMB].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_THUMB].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].visibility,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].x * image.width,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].y * image.height,
                landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX].visibility,
            ]

            pose_dict = {}
            keypoints_dict = {}
            pose_dict['people'] = [keypoints_dict]
            pose_dict['people'][0]['person_id'] = [-1]
            pose_dict['people'][0]['pose_keypoints_2d'] = keypoints

            return annotated_image, pose_dict

def crop_img(file):
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
    # print(pix.shape[0], pix.shape[1])
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
    # print(f"{x_min} {x_max} {y_min} {y_max}")

def main():
    count = 0
    start_time = time.time()
    original_files = glob.glob(input_path + "*.jpg")
    for file in original_files:
        crop_img(file)
        print(f'Running MediaPipe Pose Estimator on image: {file}')
        annotated_image, pose_dict = get_pose(file)

        with open(file.split('/')[-1].split('.')[0] + '_keypoints.json', 'w') as f:
            json.dump(pose_dict, w)
        cv2.imwrite(output_posejson + file.split('/')[-1].split('.')[0] + '_rendered.png', annotated_image)


if __name__ == "__main__":
    main()