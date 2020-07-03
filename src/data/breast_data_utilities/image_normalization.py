import numpy as np

DEFAULT_NORMALIZATION_MODE = "standard:single"


def standard_normalize_single_image(image):
    image -= np.mean(image)
    image /= np.maximum(np.std(image), 10**(-5))


def linear_normalize_single_image(image):
    max_val = np.max(image)
    min_val = np.min(image)
    image -= min_val
    image /= (max_val - min_val)


def standard_normalize_multi_image(image_ls):
    # Approximate - assumes each image is the same size for normalization, for simplicity
    mean = np.mean([np.mean(image) for image in image_ls])
    std = np.maximum(
        np.sqrt(np.mean([np.var(image) for image in image_ls])),
        10**(-5),
    )
    for image in image_ls:
        image -= mean
        image /= std


def linear_normalize_multi_image(image_ls):
    max_val = np.max([np.max(image) for image in image_ls])
    min_val = np.min([np.min(image) for image in image_ls])
    for image in image_ls:
        image -= min_val
        image /= (max_val - min_val)


def normalize_each_image(image_ls, normalization_func):
    for image in image_ls:
        normalization_func(image)


def resolve_normalization(image_ls, normalization_mode):
    normalization_func_name, application_mode = normalization_mode.split(":")
    if normalization_func_name == "standard" and application_mode == "single":
        normalize_each_image(image_ls, standard_normalize_single_image)
    elif normalization_func_name == "standard" and application_mode == "multi":
        standard_normalize_multi_image(image_ls)
    elif normalization_func_name == "linear" and application_mode == "single":
        normalize_each_image(image_ls, linear_normalize_single_image)
    elif normalization_func_name == "linear" and application_mode == "multi":
        linear_normalize_multi_image(image_ls)
    else:
        raise KeyError(normalization_mode)