import sys
import numpy as np
import multiprocessing as mp
import datetime as dt

from . import reading_images
from . import augmentations
from . import image_normalization
from .breast_data_utils import deprecated

import warnings
import functools

def loader_feeder(data_list, q_data_to_load, number_of_loaders, parameters, random_seed, permute=True):

    random_number_generator = np.random.RandomState(random_seed)
    number_of_examples = len(data_list)

    if permute:
        indices = random_number_generator.permutation(number_of_examples)
    else:
        indices = range(0, number_of_examples)

    loader_feeder_with_given_index(data_list, q_data_to_load, number_of_loaders, parameters, indices)


def loader_feeder_with_given_index(data_list, q_data_to_load, number_of_loaders, parameters, indices):
    for index in indices:
        metadata = data_list[index]

        if parameters['verbose_printing']: print('Producing meta data for index: ' + str(index))
        q_data_to_load.put((index, metadata))
        if parameters['verbose_printing']: print('Meta data for index ' + str(index) + ' put in the queue')

    for i in range(0, number_of_loaders):
        if parameters['verbose_printing']: print('All indices processed. Producing ending datum: ' + str(None))
        q_data_to_load.put(None)
        if parameters['verbose_printing']: print('Datum ' + str(None) + ' put in the queue')


def load_datum_non_image(metadata, parameters):

    # load non-image features
    number_of_features = len(parameters['non_image_features'])
    non_image_features = np.zeros(shape = (1, number_of_features), dtype = np.float32)

    counter = 0
    for feature_name in parameters['non_image_features']:
        non_image_features[counter] = metadata[feature_name]
        counter = counter + 1

    # load label
    label = metadata['label']

    datum = (non_image_features, label)

    return datum

def flip_image(image, view, horizontal_flip, mode = 'training'):
    
    if mode == 'training':
        if horizontal_flip == 'NO':
            if (view == 'R-CC') or (view == 'R-MLO'):
                image = np.fliplr(image)
        elif horizontal_flip == 'YES':
            if (view == 'L-CC') or (view == 'L-MLO'):
                image = np.fliplr(image)
    elif mode == 'medical':
        if horizontal_flip == 'YES':
            image = np.fliplr(image)

    return image


def get_single_image(image, view, parameters, random_number_generator, augmentation,
                     horizontal_flip='NO', auxiliary_image=None, best_center=None):
    """
    if using breast_data with augmentation_center = 'best_center' or 'center_of_mass',
    call random_augmentation_best_center
    otherwise, call random_augmentation
    """
    pid = mp.current_process().pid

    def log_print(s):
        print("[{}:{:%Y-%m-%d %H:%M:%S}] {}".format(pid, dt.datetime.now(), s))

    image = flip_image(image, view, horizontal_flip)
    if auxiliary_image is not None:
        auxiliary_image = flip_image(auxiliary_image, view, horizontal_flip)

    image_transformation_library = parameters.get('image_transformation_library', 'scipy')

    if type(parameters['input_size']) == tuple:
        view_input_size = parameters['input_size']
    else:
        view_input_size = parameters['input_size'][view[2:]]

    if augmentation:
        if ('augmentation_center' in parameters) and (parameters['augmentation_center'] is not None):

            cropped_image, cropped_auxiliary_image = augmentations.random_augmentation_best_center(
                image=image,
                input_size=view_input_size,
                random_number_generator=random_number_generator,
                max_crop_noise=parameters['max_crop_noise'],
                max_crop_size_noise=parameters['max_crop_size_noise'],
                max_rotation_noise=parameters['max_rotation_noise'],
                library=image_transformation_library,
                auxiliary_image=auxiliary_image,
                best_center=best_center,
                view=view,
            )
            
        else:
            cropped_image, cropped_auxiliary_image = augmentations.random_augmentation(
                image=image,
                input_size=view_input_size,
                random_number_generator=random_number_generator,
                max_crop_noise=parameters['max_crop_noise'],
                max_crop_size_noise=parameters['max_crop_size_noise'],
                max_rotation_noise=parameters['max_rotation_noise'],
                library=image_transformation_library,
                auxiliary_image=auxiliary_image,
            )

    else:
        if ('augmentation_center' in parameters) and (parameters['augmentation_center'] is not None):
            cropped_image, cropped_auxiliary_image = augmentations.random_augmentation_best_center(
                image=image,
                input_size=view_input_size,
                random_number_generator=random_number_generator,
                max_crop_noise=(0, 0),
                max_crop_size_noise=0,
                max_rotation_noise=0,
                library=image_transformation_library,
                auxiliary_image=auxiliary_image,
                best_center=best_center,
                view=view,
            )

        else:
            cropped_image, cropped_auxiliary_image = augmentations.random_augmentation(
                image=image,
                input_size=view_input_size,
                random_number_generator=random_number_generator,
                max_crop_noise=(0, 0),
                max_crop_size_noise=0,
                max_rotation_noise=0,
                library=image_transformation_library,
                auxiliary_image=auxiliary_image,
            )

    return cropped_image, cropped_auxiliary_image


def load_datum(metadata, reader_buffer, loader_buffer, parameters, random_number_generator, augmentation, randomise_view):

    # load non-image features and the label
    non_image_features, label = load_datum_non_image(metadata, parameters)

    # load images
    L_CC_index = 0
    R_CC_index = 0
    L_MLO_index = 0
    R_MLO_index = 0

    if randomise_view == True:
        L_CC_index = random_number_generator.randint(low = 0, high = len(metadata['L-CC']))
        R_CC_index = random_number_generator.randint(low = 0, high = len(metadata['R-CC']))
        L_MLO_index = random_number_generator.randint(low = 0, high = len(metadata['L-MLO']))
        R_MLO_index = random_number_generator.randint(low = 0, high = len(metadata['R-MLO']))

    assert parameters['input_format'] in ['binary', 'hdf5'], 'Wrong input format'

    L_CC_buffer, R_CC_buffer, L_MLO_buffer, R_MLO_buffer = reader_buffer

    if parameters['input_format'] == 'binary':
        L_CC = reading_images.read_image_binary_buffer(parameters['data_prefix'] + metadata['L-CC'][L_CC_index] + '.bin', L_CC_buffer)
        R_CC = reading_images.read_image_binary_buffer(parameters['data_prefix'] + metadata['R-CC'][R_CC_index] + '.bin', R_CC_buffer)
        L_MLO = reading_images.read_image_binary_buffer(parameters['data_prefix'] + metadata['L-MLO'][L_MLO_index] + '.bin', L_MLO_buffer)
        R_MLO = reading_images.read_image_binary_buffer(parameters['data_prefix'] + metadata['R-MLO'][R_MLO_index] + '.bin', R_MLO_buffer)
    elif parameters['input_format'] == 'hdf5':
        L_CC = reading_images.read_image_mat(parameters['data_prefix'] + metadata['L-CC'][L_CC_index] + '.hdf5')
        R_CC = reading_images.read_image_mat(parameters['data_prefix'] + metadata['R-CC'][R_CC_index] + '.hdf5')
        L_MLO = reading_images.read_image_mat(parameters['data_prefix'] + metadata['L-MLO'][L_MLO_index] + '.hdf5')
        R_MLO = reading_images.read_image_mat(parameters['data_prefix'] + metadata['R-MLO'][R_MLO_index] + '.hdf5')
    else:
        raise KeyError(parameters['input_format'])

    # some images are flipped horizontally
    if 'horizontal_flip' in metadata:
        if metadata['horizontal_flip'] == 'YES':
            horizontal_flip = 'YES'
        elif metadata['horizontal_flip'] == 'NO':
            horizontal_flip = 'NO'
    else:
        horizontal_flip = 'NO'

    # This is a very sensitive part of the code. L_CC, R_CC, L_MLO and R_MLO are 16-bit integers at this point.
    # If only cropping is going to be used below, there is no need to cast them explicitly to float32 here.
    # However, if rotation or resizing will be used, the type has to be changed here explicitly.

    if (augmentation == True) and ((parameters['max_crop_size_noise'] > 0) or (parameters['max_rotation_noise'] > 0.0)):
        L_CC = L_CC.astype(np.float32)
        R_CC = R_CC.astype(np.float32)
        L_MLO = L_MLO.astype(np.float32)
        R_MLO = R_MLO.astype(np.float32)

    L_CC, _ = get_single_image(L_CC, 'L-CC', parameters, random_number_generator, augmentation, horizontal_flip)
    R_CC, _ = get_single_image(R_CC, 'R-CC', parameters, random_number_generator, augmentation, horizontal_flip)
    L_MLO, _ = get_single_image(L_MLO, 'L-MLO', parameters, random_number_generator, augmentation, horizontal_flip)
    R_MLO, _ = get_single_image(R_MLO, 'R-MLO', parameters, random_number_generator, augmentation, horizontal_flip)

    # Casting from uint16 to float32 is happening here if it didn't happen above yet.
    loader_buffer[:, :, 0] = L_CC
    loader_buffer[:, :, 1] = R_CC
    loader_buffer[:, :, 2] = L_MLO
    loader_buffer[:, :, 3] = R_MLO

    # setting a default normalisation if normalisation not specified
    normalization_mode = parameters.get('normalization_mode', 'standard:single')
    image_normalization.resolve_normalization(
        image_ls=[
            loader_buffer[:, :, 0],
            loader_buffer[:, :, 1],
            loader_buffer[:, :, 2],
            loader_buffer[:, :, 3],
        ],
        normalization_mode=normalization_mode,
    )

    datum = (non_image_features, loader_buffer, label)

    return datum

def load_datum_multiple_input_size(metadata, reader_buffer, cc_loader_buffer, mlo_loader_buffer, parameters, random_number_generator, augmentation, randomise_view):

    # load non-image features and the label
    non_image_features, label = load_datum_non_image(metadata, parameters)

    # load images
    L_CC_index = 0
    R_CC_index = 0
    L_MLO_index = 0
    R_MLO_index = 0

    if randomise_view == True:
        L_CC_index = random_number_generator.randint(low = 0, high = len(metadata['L-CC']))
        R_CC_index = random_number_generator.randint(low = 0, high = len(metadata['R-CC']))
        L_MLO_index = random_number_generator.randint(low = 0, high = len(metadata['L-MLO']))
        R_MLO_index = random_number_generator.randint(low = 0, high = len(metadata['R-MLO']))

    assert parameters['input_format'] in ['binary', 'hdf5'], 'Wrong input format'
    assert parameters['augmentation_center'] in ('best_center', 'center_of_mass', None), 'Wrong augmentation_center format'

    L_CC_buffer, R_CC_buffer, L_MLO_buffer, R_MLO_buffer = reader_buffer

    if parameters['input_format'] == 'binary':
        L_CC = reading_images.read_image_binary_buffer(parameters['data_prefix'] + metadata['L-CC'][L_CC_index] + '.bin', L_CC_buffer)
        R_CC = reading_images.read_image_binary_buffer(parameters['data_prefix'] + metadata['R-CC'][R_CC_index] + '.bin', R_CC_buffer)
        L_MLO = reading_images.read_image_binary_buffer(parameters['data_prefix'] + metadata['L-MLO'][L_MLO_index] + '.bin', L_MLO_buffer)
        R_MLO = reading_images.read_image_binary_buffer(parameters['data_prefix'] + metadata['R-MLO'][R_MLO_index] + '.bin', R_MLO_buffer)
    elif parameters['input_format'] == 'hdf5':
        L_CC = reading_images.read_image_mat(parameters['data_prefix'] + metadata['L-CC'][L_CC_index] + '.hdf5')
        R_CC = reading_images.read_image_mat(parameters['data_prefix'] + metadata['R-CC'][R_CC_index] + '.hdf5')
        L_MLO = reading_images.read_image_mat(parameters['data_prefix'] + metadata['L-MLO'][L_MLO_index] + '.hdf5')
        R_MLO = reading_images.read_image_mat(parameters['data_prefix'] + metadata['R-MLO'][R_MLO_index] + '.hdf5')

    # some images are flipped horizontally
    if 'horizontal_flip' in metadata:
        if metadata['horizontal_flip'] == 'YES':
            horizontal_flip = 'YES'
        elif metadata['horizontal_flip'] == 'NO':
            horizontal_flip = 'NO'
    else:
        horizontal_flip = 'NO'

    # This is a very sensitive part of the code. L_CC, R_CC, L_MLO and R_MLO are 16-bit integers at this point.
    # If only cropping is going to be used below, there is no need to cast them explicitly to float32 here.
    # However, if rotation or resizing will be used, the type has to be changed here explicitly.

    if (augmentation == True) and ((parameters['max_crop_size_noise'] > 0) or (parameters['max_rotation_noise'] > 0.0)):
        L_CC = L_CC.astype(np.float32)
        R_CC = R_CC.astype(np.float32)
        L_MLO = L_MLO.astype(np.float32)
        R_MLO = R_MLO.astype(np.float32)

    # Use pre-calculated center points if specified
    L_CC, _ = get_single_image(L_CC, 'L-CC', parameters, random_number_generator, augmentation, horizontal_flip, best_center = None if parameters['augmentation_center'] == None else metadata[parameters['augmentation_center']]['L-CC'][L_CC_index])
    R_CC, _ = get_single_image(R_CC, 'R-CC', parameters, random_number_generator, augmentation, horizontal_flip, best_center = None if parameters['augmentation_center'] == None else metadata[parameters['augmentation_center']]['R-CC'][R_CC_index])
    L_MLO, _ = get_single_image(L_MLO, 'L-MLO', parameters, random_number_generator, augmentation, horizontal_flip, best_center = None if parameters['augmentation_center'] == None else metadata[parameters['augmentation_center']]['L-MLO'][L_MLO_index])
    R_MLO, _ = get_single_image(R_MLO, 'R-MLO', parameters, random_number_generator, augmentation, horizontal_flip, best_center = None if parameters['augmentation_center'] == None else metadata[parameters['augmentation_center']]['R-MLO'][R_MLO_index])

    # Casting from uint16 to float32 is happening here if it didn't happen above yet.
    cc_loader_buffer[:, :, 0] = L_CC
    cc_loader_buffer[:, :, 1] = R_CC
    mlo_loader_buffer[:, :, 0] = L_MLO
    mlo_loader_buffer[:, :, 1] = R_MLO

    # setting a default normalisation if normalisation not specified
    normalization_mode = parameters.get('normalization_mode', 'standard:single')

    image_normalization.resolve_normalization([
        cc_loader_buffer[:, :, 0],
        cc_loader_buffer[:, :, 1]
    ], normalization_mode=normalization_mode)
    image_normalization.resolve_normalization([
        mlo_loader_buffer[:, :, 0],
        mlo_loader_buffer[:, :, 1]
    ], normalization_mode=normalization_mode)

    datum = (non_image_features, cc_loader_buffer, mlo_loader_buffer, label)

    return datum

@deprecated
def loader(shared_array, q_data_to_load, q_data_loaded, q_empty_indices, parameters, random_seed, augmentation, randomise_view):

    random_number_generator = np.random.RandomState(random_seed)

    loader_buffer = np.zeros(shape = (parameters['input_size'][0], parameters['input_size'][1], 4), dtype = np.float32) # there are four views and that won't change

    L_CC_buffer = np.zeros(shape = 5000 * 5000, dtype = np.uint16)
    R_CC_buffer = np.zeros(shape = 5000 * 5000, dtype = np.uint16)
    L_MLO_buffer = np.zeros(shape = 5000 * 5000, dtype = np.uint16)
    R_MLO_buffer = np.zeros(shape = 5000 * 5000, dtype = np.uint16)

    reader_buffer = (L_CC_buffer, R_CC_buffer, L_MLO_buffer, R_MLO_buffer)

    while True:
        item = q_data_to_load.get()
        if parameters['verbose_printing']: print('Meta data taken out of the queue.')

        if item is None:
            if parameters['verbose_printing']: print('Loader is finishing.')
            break

        data_index, metadata = item

        try:
            loaded_datum = load_datum(metadata, reader_buffer, loader_buffer, parameters, random_number_generator, augmentation, randomise_view)

            shared_array_index = q_empty_indices.get()
            if parameters['verbose_printing']: print('Datum will be put in the shared array at index: ' + str(shared_array_index))

            non_image_features, images, label = loaded_datum # images and loader_buffer are the same object

            # put this datum in the shared memory
            shared_array[shared_array_index, :, :, 0] = images[:, :, 0] # L_CC
            shared_array[shared_array_index, :, :, 1] = images[:, :, 1] # R_CC
            shared_array[shared_array_index, :, :, 2] = images[:, :, 2] # L_MLO
            shared_array[shared_array_index, :, :, 3] = images[:, :, 3] # R_MLO

            loaded_datum_minus_images = (non_image_features, label)

            if parameters['verbose_printing']: print('Loaded datum.')
            q_data_loaded.put((data_index, shared_array_index, loaded_datum_minus_images))
            if parameters['verbose_printing']: print('Loaded datum put in the queue.')
        except (FileNotFoundError, KeyError, AssertionError, MemoryError):
            if parameters['verbose_printing']: print('Loading datum unsuccessful ' + str(sys.exc_info()[0]))
            q_data_loaded.put((data_index, -1, None))
            if parameters['verbose_printing']: print(str(None) + ' put in the queue.')

        if parameters['verbose_printing']: print('There are currently ' + str(q_data_loaded.qsize()) + ' data waiting in the queue.')

def loader_multiple_input_size(shared_array_list, q_data_to_load, q_data_loaded, q_empty_indices, parameters, random_seed, augmentation, randomise_view):

    random_number_generator = np.random.RandomState(random_seed)

    cc_loader_buffer = np.zeros(shape = (parameters['input_size']['CC'][0], parameters['input_size']['CC'][1], 2), dtype = np.float32) # 2 types of views
    mlo_loader_buffer = np.zeros(shape = (parameters['input_size']['MLO'][0], parameters['input_size']['MLO'][1], 2), dtype = np.float32) # 2 types of views

    L_CC_buffer = np.zeros(shape = 5000 * 5000, dtype = np.uint16)
    R_CC_buffer = np.zeros(shape = 5000 * 5000, dtype = np.uint16)
    L_MLO_buffer = np.zeros(shape = 5000 * 5000, dtype = np.uint16)
    R_MLO_buffer = np.zeros(shape = 5000 * 5000, dtype = np.uint16)

    reader_buffer = (L_CC_buffer, R_CC_buffer, L_MLO_buffer, R_MLO_buffer)

    while True:
        item = q_data_to_load.get()
        if parameters['verbose_printing']: print('Meta data taken out of the queue.')

        if item is None:
            if parameters['verbose_printing']: print('Loader is finishing.')
            break

        data_index, metadata = item

        try:
            loaded_datum = load_datum_multiple_input_size(metadata, reader_buffer, cc_loader_buffer, mlo_loader_buffer, parameters, random_number_generator, augmentation, randomise_view)

            shared_array_index = q_empty_indices.get()
            if parameters['verbose_printing']: print('Datum will be put in the shared array at index: ' + str(shared_array_index))

            non_image_features, cc_images, mlo_images, label = loaded_datum # images and loader_buffer are the same object

            # put this datum in the shared memory
            shared_array_list[0][shared_array_index, :, :, 0] = cc_images[:, :, 0] # L_CC
            shared_array_list[1][shared_array_index, :, :, 0] = cc_images[:, :, 1] # R_CC
            shared_array_list[2][shared_array_index, :, :, 0] = mlo_images[:, :, 0] # L_MLO
            shared_array_list[3][shared_array_index, :, :, 0] = mlo_images[:, :, 1] # R_MLO

            loaded_datum_minus_images = (non_image_features, label)

            if parameters['verbose_printing']: print('Loaded datum.')
            q_data_loaded.put((data_index, shared_array_index, loaded_datum_minus_images))
            if parameters['verbose_printing']: print('Loaded datum put in the queue.')
        except (FileNotFoundError, KeyError, AssertionError, MemoryError):
            #raise
            if parameters['verbose_printing']: print('Loading datum unsuccessful ' + str(sys.exc_info()[0]))
            q_data_loaded.put((data_index, -1, None))
            if parameters['verbose_printing']: print(str(None) + ' put in the queue.')

        if parameters['verbose_printing']: print('There are currently ' + str(q_data_loaded.qsize()) + ' data waiting in the queue.')


def non_image_loader(q_data_to_load, q_data_loaded, parameters):

    while True:
        item = q_data_to_load.get()
        if parameters['verbose_printing']: print('Meta data taken out of the queue.')

        if item is None:
            if parameters['verbose_printing']: print('Loader is finishing.')
            break

        data_index, metadata = item

        try:
            loaded_datum = load_datum_non_image(metadata, parameters)
            loaded_datum_minus_images = loaded_datum

            if parameters['verbose_printing']: print('Loaded datum.')
            q_data_loaded.put((data_index, loaded_datum_minus_images))
            if parameters['verbose_printing']: print('Loaded datum put in the queue.')
        except (FileNotFoundError, KeyError):
            if parameters['verbose_printing']: print('Loading datum unsuccessful ' + str(sys.exc_info()[0]))
            q_data_loaded.put((data_index, -1, None))
            if parameters['verbose_printing']: print(str(None) + ' put in the queue.')

        if parameters['verbose_printing']: print('There are currently ' + str(q_data_loaded.qsize()) + ' data waiting in the queue.')
