import os
import numpy as np
import pandas as pd
import scipy.ndimage
import imageio
import logging
logger = logging.getLogger(__name__)

from argparse import ArgumentParser

def get_masks_and_sizes_of_connected_components(img_mask):
    '''
    Finds the connected components from the mask of the image
    '''
    mask, num_labels = scipy.ndimage.label(img_mask)

    mask_pixels_dict = {}
    for i in range(num_labels + 1):
        this_mask = (mask == i)
        if img_mask[this_mask][0] != 0:
            # Exclude the 0-valued mask
            mask_pixels_dict[i] = np.sum(this_mask)

    return mask, mask_pixels_dict


def get_mask_of_largest_connected_component(img_mask):
    '''
    Finds the largest connected component from the mask of the image
    '''
    mask, mask_pixels_dict = get_masks_and_sizes_of_connected_components(img_mask)
    largest_mask_index = pd.Series(mask_pixels_dict).idxmax()
    largest_mask = mask == largest_mask_index
    return largest_mask


def get_edge_values(img, largest_mask, axis):
    '''
    Finds the bounding box for the largest connected component
    '''
    assert axis in ['x', 'y']
    has_value = np.any(largest_mask, axis=int(axis == 'y'))
    edge_start = np.arange(img.shape[int(axis == 'x')])[has_value][0]
    edge_end = np.arange(img.shape[int(axis == 'x')])[has_value][-1] + 1
    return edge_start, edge_end


def crop(img):
    img_mask = img > 0
    try:
        largest_mask = get_mask_of_largest_connected_component(img_mask)
        y_edge_top, y_edge_bottom = get_edge_values(img, largest_mask, 'y')
        x_edge_left, x_edge_right = get_edge_values(img, largest_mask, 'x')
        return img[y_edge_top:y_edge_bottom, x_edge_left:x_edge_right]
    except:
        return img


def main(raw_dpath, cropped_dpath):
    for img_fname in os.listdir(raw_dpath):
        logger.info(f'Processing {img_fname}')
        raw_fpath = os.path.join(raw_dpath, img_fname)
        cropped_fpath = os.path.join(cropped_dpath, img_fname)
        img = crop(imageio.imread(raw_fpath, as_gray=True))
        imageio.imsave(cropped_fpath, img.astype('uint8'))


if __name__ == '__main__':
    parser = ArgumentParser(description='Remove background from raw images')
    parser.add_argument('--raw-dpath', type=str, required=True)
    parser.add_argument('--cropped-dpath', type=str, required=True)
    args = parser.parse_args()
    main(args.raw_dpath, args.cropped_dpath)