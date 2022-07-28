import cv2
from PIL import Image, ImageFilter
import numpy as np
from scipy import ndimage as ndi
from scipy.ndimage import convolve
from skimage import morphology
import torch
from get_json import get_binary_mask

AFFORDABLE_ERROR = 70  # You can control stringency of performance of background removal by modifying this value
BACKGROUND_COLOR = 230  # This value controls the color of the background
NUM_REMOVE_LAYER = 10  # This value controls the number of additional removal
REMOVE_DEPTH = 10


def remove_background(image, predictions, index):
    """
    :param image: Image (numpy array, shape=(w,h,3))
    :param predictions: contains information about prediction such as mask
    :param index: the index of the detected instances.
    :return: a background removed, improved, and blurred image
    """

    mask = predictions["instances"].pred_masks
    mask = mask.cpu().numpy()
 

    test_image = image.copy()
    w = len(mask[0])
    h = len(mask[0, 0])
    # mask processing: Removing an island
    # mask[index] = morphology.remove_small_objects(mask[index].astype(bool), min_size=5000, connectivity=1).astype(int)
    # mask[index] = mask_biggest(mask[index].astype(bool), 1)
    mask[index] = mask_the_one(mask[index])

    edge_list = get_edge(mask[index])
    # 0. Remove background at first
    test_image[np.where(mask[index] == 0)] = BACKGROUND_COLOR

    # 1. Remove the extraneous pixels
    for num_iter in range(NUM_REMOVE_LAYER):
        remove_list = []

        for edge in edge_list:
            remove_around(
                mask=mask,
                image=test_image,
                i=edge[0],
                j=edge[1],
                index=index,
                remove_depth=REMOVE_DEPTH,
                w=w,
                h=h,
                remove_list=remove_list
            )

        for remove_item in remove_list:
            mask[index, remove_item[0], remove_item[1]] = 0  # update masks
            test_image[remove_item[0], remove_item[1]] = BACKGROUND_COLOR



    '''
    # There is no need to blur images here. blurring will be used in insert_background
    # 2. Blur the edges
    edge_list = get_edge(mask[index])  # Since mask is changed a little bit, we have to find edges again.
    blur_dict = {}

    for edge in edge_list:
        blurring(
            image=test_image,
            i=edge[0],
            j=edge[1],
            pixel_depth=2,
            blur_depth=1,  # change this value if you want to change the magnitude.
            w=w,
            h=h,
            blur_dict=blur_dict,
        )

    for coordinates, rgb in zip(blur_dict.keys(), blur_dict.values()):
        test_image[coordinates[0], coordinates[1]] = rgb
    '''


    predictions["instances"].pred_masks = torch.from_numpy(mask)  # overwrite mask to new mask

    return test_image


def remove_around(mask, image, i, j, index, remove_depth, w, h, remove_list):
    """
    :param mask: Binary Mask (Tensor, shape=(1,w,h))
    :param image: Image (numpy array, shape=(w,h,3))
    :param i: x coordinate of the pivot pixel
    :param j: y coordinate of the pivot pixel
    :param index: the index of detected instances.
    :param remove_depth: the number of additional pixels we have to consider when compare the pivot with around pixels
    :param w: width of the image
    :param h: height of the image
    :param remove_list: the list which will contain information about the coordinates of the pixels need to be removed
    :return: nothing
    """

    assert type(remove_depth) == int

    error_sum = 0
    num_valid_pixels = 0

    if 0 <= i - remove_depth and i + remove_depth <= w - 1 and 0 <= j - remove_depth and j + remove_depth <= h - 1:

        for x in range(i - remove_depth, i + remove_depth + 1):
            for y in range(j - remove_depth, j + remove_depth + 1):
                ''' This method has poor performance
                if mask[index, x, y] == 1 and compare_pixel(image[i, j], image[x, y]) > AFFORDABLE_ERROR:
                    image[x, y] = BACKGROUND_COLOR
                    mask_change[0, x, y] = 0
                '''

                if mask[index, x, y] == 1:
                    num_valid_pixels += 1
                    error_sum += compare_pixel(image[i, j], image[x, y])


    # print(error_sum / num_valid_pixels)
    if num_valid_pixels != 0 and error_sum / num_valid_pixels > AFFORDABLE_ERROR:
        # image[i, j] = BACKGROUND_COLOR  # if you want a smoother curve, use this line of codes
        remove_list.append((i, j))


def blurring(image, i, j, pixel_depth, blur_depth, w, h, blur_dict):
    """
    :param image: image file consists of rgb color
    :param i: pixel x coordinate index
    :param j: pixel y coordinate index
    :param pixel_depth: consider (pixel_depth + 1) * (pixel_depth + 1) region when blurs (pivot is i and j)
    :param blur_depth: additional regions to blur. (i - blur_depth, j - blur_depth)
                       to (i + blur_depth, j + blur_depth) will be blurred
    :param w: width of image
    :param h: height of image
    :param blur_dict: (key: coordinates of the blurred pixels) (value: list of three rgb values)
    :return: nothing
    """

    assert pixel_depth >= 1 and type(pixel_depth) == int
    assert blur_depth < pixel_depth and type(blur_depth) == int

    sum_pixel = [0, 0, 0]

    if 0 <= i - pixel_depth and i + pixel_depth <= w - 1 and 0 <= j - pixel_depth and j + pixel_depth <= h - 1:
        # 1. Blur the pivot pixel
        for x in range(i - pixel_depth, i + pixel_depth + 1):
            for y in range(j - pixel_depth, j + pixel_depth + 1):
                sum_pixel = [a + b for a, b in zip(sum_pixel, image[x, y])]

        # (i, j) value will be updated
        blur_dict[(i, j)] = [element / ((2 * pixel_depth + 1) ** 2) for element in sum_pixel]

        # 2. Blur the pixels around the pivot pixel
        for x in range(i - blur_depth, i + blur_depth + 1):
            for y in range(j - blur_depth, j + blur_depth + 1):
                sum_pixel = [0, 0, 0]
                if not (x == i and y == j) and (x, y) not in blur_dict.keys():  # existing key will not be updated
                    for x_0 in range(x - 1, x + 2):
                        for y_0 in range(y - 1, y + 2):
                            sum_pixel = [a + b for a, b in zip(sum_pixel, image[x_0, y_0])]

                    blur_dict[(x, y)] = [element / ((2 * 1 + 1) ** 2) for element in sum_pixel]

    # Consider weights?
    # In my opinion, there is a possibility that we have to consider weight when adding another background.


def compare_pixel(pixel_1, pixel_2):
    """
    :param pixel_1:  list of length 3 (rgb values)
    :param pixel_2:  list of length 3 (rgb values)
    :return: root-sum-square distance (You can change this method if you find a method more conducive to better results)
    """

    if pixel_1 is pixel_2:
        return 0

    r = (pixel_1[0].astype(int) - pixel_2[0].astype(int)) ** 2
    g = (pixel_1[1].astype(int) - pixel_2[1].astype(int)) ** 2
    b = (pixel_1[2].astype(int) - pixel_2[2].astype(int)) ** 2

    return (r + g + b) ** 0.5

def get_edge(mask):
    """
    Reference to (https://stackoverflow.com/questions/64932502/python-numpy-find-edges-of-a-2d-3d-mask)

    Use convolution to judge whether each pixel is edge or not
    :param mask: the mask of a certain instance. Usually, mask[index]
    :return: list that contains information about coordinates of the edges
    """

    fil = [[-1, -1, -1],
           [-1, 8, -1],
           [-1, -1, -1]]

    x, y = np.where(convolve(mask, fil, mode='constant') > 1)

    edge_list = zip(x, y)

    return edge_list


def mask_the_one(mask):
    """
    :param mask: 2 dimensional numpy array with binary mask information
    :return: mask that the islands are deleted
    """

    mask_image = (mask * 255).astype(np.uint8)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask_image, connectivity=8)
    sizes = stats[1:, -1];
    nb_components = nb_components - 1

    max_size = 0
    max_index = 0

    out = np.zeros((output.shape))

    for i in range(0, nb_components):
        if sizes[i] >= max_size:
            max_size = sizes[i]
            max_index = i

    out[output == max_index + 1] = 1

    return out

