import cv2
import numpy as np
from background_removal import blurring, get_edge


def insert_bg(img_bg, img_target, mask_target, y_min, y_max, x):
    """
    :param img_bg: background image (numpy array (height, width, rgb colors))
    :param img_target: target image (numpy array (height, width, rgb colors))
    :param mask_target: target's masks
    :param y_min:
    :param y_max:
    :param x:
    :return:
    """

    h_bg = img_bg.shape[0]
    w_bg = img_bg.shape[1]
    h_target = img_target.shape[0]
    w_target = img_target.shape[1]

    assert 5 <= x <= w_bg - 6, "x out of range. At least 5 pixels should be left as a blank"
    assert 5 <= y_min < y_max <= h_bg - 6, "y out of range. At least 5 pixels should be left as a blank"

    # find a bbox of target by coordinates
    temp = np.where(mask_target != 0)
    target_y_min, target_y_max = np.min(temp[0]), np.max(temp[0])
    target_x_min, target_x_max = np.min(temp[1]), np.max(temp[1])

    # find resize ratio
    assert target_y_max != target_y_min
    resize_ratio = (y_max - y_min) / (target_y_max - target_y_min)

    resized_h = int(h_target * resize_ratio)
    resized_w = int(w_target * resize_ratio)

    target_resized = cv2.resize(
        img_target,
        dsize=(resized_h, resized_w),
        interpolation=cv2.INTER_NEAREST
    )

    mask_resized = cv2.resize(
        mask_target,
        dsize=(resized_h, resized_w),
        interpolation=cv2.INTER_NEAREST
    )

    # find a bbox of resized_target
    temp = np.where(mask_resized != 0)
    target_y_min, target_y_max = np.min(temp[0]), np.max(temp[0])
    target_x_min, target_x_max = np.min(temp[1]), np.max(temp[1])

    # find parallel translation
    y_trans = y_min - target_y_min
    x_trans = int(x - (target_x_max + target_x_min) / 2)

    bg_modified = img_bg.copy()
    edge_list = get_edge(mask_resized)
    edge_list = [(edge[0] + y_trans, edge[1] + x_trans) for edge in edge_list]

    for y in range(target_y_min, target_y_max + 1):
        for x in range(target_x_min, target_x_max + 1):
            if mask_resized[y, x] != 0:
                try:
                    bg_modified[y + y_trans, x + x_trans] = target_resized[y, x]
                except IndexError:
                    pass

    blur_dict = {}

    for edge in edge_list:
        try:
            blurring(
                image=bg_modified,
                i=edge[0],
                j=edge[1],
                pixel_depth=2,
                blur_depth=1,  # change this value if you want to change the magnitude.
                w=h_bg,  # This part is not a mishap. image shape is reversed.
                h=w_bg,  # This part is not a mishap.
                blur_dict=blur_dict,
            )
        except IndexError:
            pass

    for coordinates, rgb in zip(blur_dict.keys(), blur_dict.values()):
        bg_modified[coordinates[0], coordinates[1]] = rgb

    return bg_modified


'''
def insert_bg(img_bg, img_target, mask_target, x_0, y_0, x_1, y_1):
    """
    :param img_bg: numpy image. shape:(height, width, 3)
    :param img_target:
    :param mask_target: ndarray
    :param x_0: a top left x coordinate (of target images)
    :param y_0: a top left y coordinate
    :param x_1: a bottom right x coordinate
    :param y_1: a bottom right y coordinate
    :return:
    """
    print(type(img_bg))
    h = img_bg.shape[0]
    w = img_bg.shape[1]
    print(w, h)
    assert 0 <= x_0 < x_1 <= w - 1 and 0 <= y_1 < y_0 <= h - 1

    bg_modified = img_bg.copy()
    mask_resized = cv2.resize(mask_target, (y_0 - y_1, x_1 - x_0), interpolation=cv2.INTER_AREA)
    target_resized = cv2.resize(img_target, (y_0 - y_1, x_1 - x_0), interpolation=cv2.INTER_AREA)
    print(mask_resized.shape, target_resized.shape)
    print(x_0, y_0, x_1, y_1)
    for i in range(y_1, y_0):
        for j in range(x_0, x_1):
            if mask_resized[i - y_1, j - x_0] != 0:
                print(img_target[i - y_1, j - x_0])
                print(target_resized[i - y_1, j - x_0])
                bg_modified[i, j] = target_resized[i - y_1, j - x_0]

    return bg_modified
'''
