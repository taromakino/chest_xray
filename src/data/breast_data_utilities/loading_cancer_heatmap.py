import datetime as dt
import sys
import numpy
import multiprocessing as mp
import imageio
import os
from . import loading
from . import reading_images
from . import image_normalization

# TODO: [zp489] Integrate into regular loading code


def loader_feeder(data_list, q_data_to_load, number_of_loaders, parameters, random_seed, permute=True):
    loading.loader_feeder(data_list, q_data_to_load, number_of_loaders, parameters, random_seed, permute=permute)


def loader_feeder_with_given_index(data_list, q_data_to_load, number_of_loaders, parameters, indices):
    loading.loader_feeder_with_given_index(data_list, q_data_to_load, number_of_loaders, parameters, indices)


def flip_image(image, view, horizontal_flip):
    return loading.flip_image(image, view, horizontal_flip, mode='training')


def get_single_image(image, view, parameters, random_number_generator, augmentation,
                     horizontal_flip, auxiliary_image, best_center):
    pid = mp.current_process().pid

    def log_print(s):
        print("[{}:{:%Y-%m-%d %H:%M:%S}] {}".format(pid, dt.datetime.now(), s))
    
    return loading.get_single_image(
        image, view, parameters, random_number_generator, augmentation,
        horizontal_flip=horizontal_flip, auxiliary_image=auxiliary_image, best_center=best_center,
    )


def load_datum_non_image(metadata, parameters):
    return loading.load_datum_non_image(metadata, parameters)


def load_cancer_heatmap_image(cancer_heatmap_prefix_ls, file_name):
    channel_ls = []
    for cancer_heatmap_prefix in cancer_heatmap_prefix_ls:
        channel_ls.append(reading_images.read_image_mat(cancer_heatmap_prefix + file_name))
    result = numpy.stack(channel_ls, axis=2)
    return result

def read_segmentation(seg_file_path, required_shape):
    # TODO: This should go to reading_images
    if os.path.exists(seg_file_path):
        return numpy.minimum(imageio.imread(seg_file_path),1).astype(numpy.float32)
    else:
        placeholder = numpy.empty(required_shape, dtype=numpy.float32)
        placeholder.fill(-1)
        return placeholder
    
def load_cancer_segmentation_image(cancer_heatmap_prefix_ls, file_name, required_shape):
    channel_ls = []
    for cancer_heatmap_prefix in cancer_heatmap_prefix_ls:
        # TODO: Check if correct, the length of the list should have been asserted as 1
        # TODO: use filename_postfixes
        channel_ls.append(read_segmentation(os.path.join(cancer_heatmap_prefix, file_name+'.benign.png'), required_shape))
        channel_ls.append(read_segmentation(os.path.join(cancer_heatmap_prefix, file_name+'.malignant.png'), required_shape))
    result = numpy.stack(channel_ls, axis=2)
    return result

def load_datum(metadata, loader_buffer_image, parameters, random_number_generator, augmentation, randomise_view):

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

    L_CC_image = reading_images.read_image_mat(parameters['data_prefix'] + metadata['L-CC'][L_CC_index] + '.hdf5')
    R_CC_image = reading_images.read_image_mat(parameters['data_prefix'] + metadata['R-CC'][R_CC_index] + '.hdf5')
    L_MLO_image = reading_images.read_image_mat(parameters['data_prefix'] + metadata['L-MLO'][L_MLO_index] + '.hdf5')
    R_MLO_image = reading_images.read_image_mat(parameters['data_prefix'] + metadata['R-MLO'][R_MLO_index] + '.hdf5')

    if ('load_segmentation' in parameters) and parameters['load_segmentation']:
        L_CC_cancer_heatmap = load_cancer_segmentation_image(
            parameters['cancer_heatmap_prefix'], metadata['L-CC'][L_CC_index], L_CC_image.shape)
        R_CC_cancer_heatmap = load_cancer_segmentation_image(
            parameters['cancer_heatmap_prefix'], metadata['R-CC'][R_CC_index], R_CC_image.shape)
        L_MLO_cancer_heatmap = load_cancer_segmentation_image(
            parameters['cancer_heatmap_prefix'], metadata['L-MLO'][L_MLO_index], L_MLO_image.shape)
        R_MLO_cancer_heatmap = load_cancer_segmentation_image(
            parameters['cancer_heatmap_prefix'], metadata['R-MLO'][R_MLO_index], R_MLO_image.shape)
    else:
        L_CC_cancer_heatmap = load_cancer_heatmap_image(
            parameters['cancer_heatmap_prefix'], metadata['L-CC'][L_CC_index] + '.hdf5')
        R_CC_cancer_heatmap = load_cancer_heatmap_image(
            parameters['cancer_heatmap_prefix'], metadata['R-CC'][R_CC_index] + '.hdf5')
        L_MLO_cancer_heatmap = load_cancer_heatmap_image(
            parameters['cancer_heatmap_prefix'], metadata['L-MLO'][L_MLO_index] + '.hdf5')
        R_MLO_cancer_heatmap = load_cancer_heatmap_image(
            parameters['cancer_heatmap_prefix'], metadata['R-MLO'][R_MLO_index] + '.hdf5')

    # some images are flipped horizontally
    if 'horizontal_flip' in metadata:
        if metadata['horizontal_flip'] == 'YES':
            horizontal_flip = 'YES'
        elif metadata['horizontal_flip'] == 'NO':
            horizontal_flip = 'NO'
        else:
            raise KeyError(metadata['horizontal_flip'])
    else:
        horizontal_flip = 'NO'

    # This is a very sensitive part of the code. L_CC_image, R_CC_image, L_MLO_image and R_MLO_image are 16-bit integers at this point.
    # If only cropping is going to be used below, there is no need to cast them explicitly to float32 here.
    # However, if rotation or resizing will be used, the type has to be changed here explicitly.

    if (augmentation == True) and ((parameters['max_crop_size_noise'] > 0) or (parameters['max_rotation_noise'] > 0.0)):
        L_CC_image = L_CC_image.astype(numpy.float32)
        R_CC_image = R_CC_image.astype(numpy.float32)
        L_MLO_image = L_MLO_image.astype(numpy.float32)
        R_MLO_image = R_MLO_image.astype(numpy.float32)

    # the second item the function returns is the cancer heatmap image cropped identically to the original image
    L_CC_image, L_CC_cancer_heatmap = get_single_image(L_CC_image, 'L-CC', parameters, random_number_generator, augmentation, horizontal_flip, L_CC_cancer_heatmap)
    R_CC_image, R_CC_cancer_heatmap = get_single_image(R_CC_image, 'R-CC', parameters, random_number_generator, augmentation, horizontal_flip, R_CC_cancer_heatmap)
    L_MLO_image, L_MLO_cancer_heatmap = get_single_image(L_MLO_image, 'L-MLO', parameters, random_number_generator, augmentation, horizontal_flip, L_MLO_cancer_heatmap)
    R_MLO_image, R_MLO_cancer_heatmap = get_single_image(R_MLO_image, 'R-MLO', parameters, random_number_generator, augmentation, horizontal_flip, R_MLO_cancer_heatmap)

    # Casting from uint16 to float32 is happening here if it didn't happen above yet.
    loader_buffer_image[:, :, 0] = L_CC_image
    loader_buffer_image[:, :, 1] = R_CC_image
    loader_buffer_image[:, :, 2] = L_MLO_image
    loader_buffer_image[:, :, 3] = R_MLO_image

    # setting a default normalisation if normalisation not specified
    if 'normalisation_type' not in parameters:
        parameters['normalization_mode'] = 'standard:single'

    image_normalization.resolve_normalization([
        loader_buffer_image[:, :, 0],
        loader_buffer_image[:, :, 1],
        loader_buffer_image[:, :, 2],
        loader_buffer_image[:, :, 3],
    ], normalization_mode=parameters['normalization_mode'])

    cancer_heatmap = numpy.stack([L_CC_cancer_heatmap, R_CC_cancer_heatmap, L_MLO_cancer_heatmap, R_MLO_cancer_heatmap], axis = 2)

    datum = (non_image_features, loader_buffer_image, cancer_heatmap, label)

    return datum


def load_datum_multiple_input_size(metadata, reader_buffer, cancer_heatmap_reader_buffer,
                                   cc_loader_buffer, mlo_loader_buffer,
                                   cc_cancer_heatmap_loader_buffer, mlo_cancer_heatmap_loader_buffer,
                                   parameters, random_number_generator, augmentation, randomise_view):

    # load non-image features and the label
    non_image_features, label = load_datum_non_image(metadata, parameters)

    pid = mp.current_process().pid

    def log_print(s):
        print("[{}:{:%Y-%m-%d %H:%M:%S}] {}".format(pid, dt.datetime.now(), s))


    # load images
    L_CC_index = 0
    R_CC_index = 0
    L_MLO_index = 0
    R_MLO_index = 0

    if randomise_view == True:
        L_CC_index = random_number_generator.randint(low=0, high=len(metadata['L-CC']))
        R_CC_index = random_number_generator.randint(low=0, high=len(metadata['R-CC']))
        L_MLO_index = random_number_generator.randint(low=0, high=len(metadata['L-MLO']))
        R_MLO_index = random_number_generator.randint(low=0, high=len(metadata['R-MLO']))

    assert parameters['input_format'] in ['binary', 'hdf5'], 'Wrong input format'
    assert parameters['augmentation_center'] in ('best_center', 'center_of_mass', None), 'Wrong augmentation_center format'

    L_CC_buffer, R_CC_buffer, L_MLO_buffer, R_MLO_buffer = reader_buffer
    L_CC_cancer_heatmap_buffer, R_CC_cancer_heatmap_buffer, L_MLO_cancer_heatmap_buffer, R_MLO_cancer_heatmap_buffer = \
        cancer_heatmap_reader_buffer

    if parameters['input_format'] == 'binary':
        if parameters['load_segmentation']:
            raise NotImplementedError("loading segmentation in binary format not implemented")

        L_CC = reading_images.read_image_binary_buffer(
            parameters['data_prefix'] + metadata['L-CC'][L_CC_index] + '.bin', L_CC_buffer)
        R_CC = reading_images.read_image_binary_buffer(
            parameters['data_prefix'] + metadata['R-CC'][R_CC_index] + '.bin', R_CC_buffer)
        L_MLO = reading_images.read_image_binary_buffer(
            parameters['data_prefix'] + metadata['L-MLO'][L_MLO_index] + '.bin', L_MLO_buffer)
        R_MLO = reading_images.read_image_binary_buffer(
        parameters['data_prefix'] + metadata['R-MLO'][R_MLO_index] + '.bin', R_MLO_buffer)
        L_CC_cancer_heatmap = reading_images.read_image_binary_buffer(
            parameters['cancer_data_prefix'] + metadata['L-CC'][L_CC_index] + '.bin', L_CC_cancer_heatmap_buffer)
        R_CC_cancer_heatmap = reading_images.read_image_binary_buffer(
            parameters['cancer_data_prefix'] + metadata['R-CC'][R_CC_index] + '.bin', R_CC_cancer_heatmap_buffer)
        L_MLO_cancer_heatmap = reading_images.read_image_binary_buffer(
            parameters['cancer_data_prefix'] + metadata['L-MLO'][L_MLO_index] + '.bin', L_MLO_cancer_heatmap_buffer)
        R_MLO_cancer_heatmap = reading_images.read_image_binary_buffer(
            parameters['cancer_data_prefix'] + metadata['R-MLO'][R_MLO_index] + '.bin', R_MLO_cancer_heatmap_buffer)

        if parameters['verbose_printing']:
            log_print('Image loaded.')

    elif parameters['input_format'] == 'hdf5':
        L_CC = reading_images.read_image_mat(parameters['data_prefix'] + metadata['L-CC'][L_CC_index] + '.hdf5')
        R_CC = reading_images.read_image_mat(parameters['data_prefix'] + metadata['R-CC'][R_CC_index] + '.hdf5')
        L_MLO = reading_images.read_image_mat(parameters['data_prefix'] + metadata['L-MLO'][L_MLO_index] + '.hdf5')
        R_MLO = reading_images.read_image_mat(parameters['data_prefix'] + metadata['R-MLO'][R_MLO_index] + '.hdf5')

        if parameters['verbose_printing']:
            log_print('Image loaded.')

        if ('load_segmentation' in parameters) and parameters['load_segmentation']:
            L_CC_cancer_heatmap = load_cancer_segmentation_image(
                parameters['cancer_heatmap_prefix'], metadata['L-CC'][L_CC_index], L_CC.shape)
            R_CC_cancer_heatmap = load_cancer_segmentation_image(
                parameters['cancer_heatmap_prefix'], metadata['R-CC'][R_CC_index], R_CC.shape)
            L_MLO_cancer_heatmap = load_cancer_segmentation_image(
                parameters['cancer_heatmap_prefix'], metadata['L-MLO'][L_MLO_index], L_MLO.shape)
            R_MLO_cancer_heatmap = load_cancer_segmentation_image(
                parameters['cancer_heatmap_prefix'], metadata['R-MLO'][R_MLO_index], R_MLO.shape)
        else:
            
            L_CC_cancer_heatmap = load_cancer_heatmap_image(
                parameters['cancer_heatmap_prefix'], metadata['L-CC'][L_CC_index] + '.hdf5')
            R_CC_cancer_heatmap = load_cancer_heatmap_image(
                parameters['cancer_heatmap_prefix'], metadata['R-CC'][R_CC_index] + '.hdf5')
            L_MLO_cancer_heatmap = load_cancer_heatmap_image(
                parameters['cancer_heatmap_prefix'], metadata['L-MLO'][L_MLO_index] + '.hdf5')
            R_MLO_cancer_heatmap = load_cancer_heatmap_image(
                parameters['cancer_heatmap_prefix'], metadata['R-MLO'][R_MLO_index] + '.hdf5')

        if parameters['verbose_printing']:
            log_print('heatmap loaded.')
            
    else:
        raise KeyError("parameters['input_format']")

    # some images are flipped horizontally
    if 'horizontal_flip' in metadata:
        if metadata['horizontal_flip'] == 'YES':
            horizontal_flip = 'YES'
        elif metadata['horizontal_flip'] == 'NO':
            horizontal_flip = 'NO'
        else:
            raise KeyError(metadata['horizontal_flip'])
    else:
        horizontal_flip = 'NO'

    # This is a very sensitive part of the code. L_CC, R_CC, L_MLO and R_MLO are 16-bit integers at this point.
    # If only cropping is going to be used below, there is no need to cast them explicitly to float32 here.
    # However, if rotation or resizing will be used, the type has to be changed here explicitly.

    if (augmentation == True) and ((parameters['max_crop_size_noise'] > 0) or (parameters['max_rotation_noise'] > 0.0)):
        L_CC = L_CC.astype(numpy.float32)
        R_CC = R_CC.astype(numpy.float32)
        L_MLO = L_MLO.astype(numpy.float32)
        R_MLO = R_MLO.astype(numpy.float32)
        L_CC_cancer_heatmap = L_CC_cancer_heatmap.astype(numpy.float32)
        R_CC_cancer_heatmap = R_CC_cancer_heatmap.astype(numpy.float32)
        L_MLO_cancer_heatmap = L_MLO_cancer_heatmap.astype(numpy.float32)
        R_MLO_cancer_heatmap = R_MLO_cancer_heatmap.astype(numpy.float32)


    # Use pre-calculated center points if specified
    L_CC, L_CC_cancer_heatmap = get_single_image(
        L_CC, 'L-CC', parameters, random_number_generator, augmentation, horizontal_flip, L_CC_cancer_heatmap,
        best_center=None if parameters['augmentation_center'] is None else metadata[parameters['augmentation_center']]['L-CC'][L_CC_index],
    )
    R_CC, R_CC_cancer_heatmap = get_single_image(
        R_CC, 'R-CC', parameters, random_number_generator, augmentation, horizontal_flip, R_CC_cancer_heatmap,
        best_center=None if parameters['augmentation_center'] is None else metadata[parameters['augmentation_center']]['R-CC'][R_CC_index],
    )
    L_MLO, L_MLO_cancer_heatmap = get_single_image(
        L_MLO, 'L-MLO', parameters, random_number_generator, augmentation, horizontal_flip, L_MLO_cancer_heatmap,
        best_center=None if parameters['augmentation_center'] is None else metadata[parameters['augmentation_center']]['L-MLO'][L_MLO_index],
    )
    R_MLO, R_MLO_cancer_heatmap = get_single_image(
        R_MLO, 'R-MLO', parameters, random_number_generator, augmentation, horizontal_flip, R_MLO_cancer_heatmap,
        best_center=None if parameters['augmentation_center'] is None else metadata[parameters['augmentation_center']]['R-MLO'][R_MLO_index],
    )

    if parameters['verbose_printing']:
        log_print('Image processed.')


    # Casting from uint16 to float32 is happening here if it didn't happen above yet.
    cc_loader_buffer[:, :, 0] = L_CC
    cc_loader_buffer[:, :, 1] = R_CC
    mlo_loader_buffer[:, :, 0] = L_MLO
    mlo_loader_buffer[:, :, 1] = R_MLO
    cc_cancer_heatmap_loader_buffer[:, :, 0, :] = L_CC_cancer_heatmap
    cc_cancer_heatmap_loader_buffer[:, :, 1, :] = R_CC_cancer_heatmap
    mlo_cancer_heatmap_loader_buffer[:, :, 0, :] = L_MLO_cancer_heatmap
    mlo_cancer_heatmap_loader_buffer[:, :, 1, :] = R_MLO_cancer_heatmap

    # setting a default normalisation if normalisation not specified
    if 'normalisation_type' not in parameters:
        parameters['normalization_mode'] = 'standard:single'

    image_normalization.resolve_normalization([
        cc_loader_buffer[:, :, 0],
        cc_loader_buffer[:, :, 1]
    ], normalization_mode=parameters['normalization_mode'])
    image_normalization.resolve_normalization([
        mlo_loader_buffer[:, :, 0],
        mlo_loader_buffer[:, :, 1]
    ], normalization_mode=parameters['normalization_mode'])
    
    if parameters['verbose_printing']:
        log_print('Image normalisation.')

    datum = (
        non_image_features,
        cc_loader_buffer, mlo_loader_buffer,
        cc_cancer_heatmap_loader_buffer, mlo_cancer_heatmap_loader_buffer,
        label,
    )

    return datum


def loader_multiple_input_size(shared_array_list, shared_array_cancer_heatmap,
                               q_data_to_load, q_data_loaded, q_empty_indices, parameters, random_seed,
                               augmentation, randomise_view):

    random_number_generator = numpy.random.RandomState(random_seed)

    cc_loader_buffer = numpy.zeros(
        shape=(parameters['input_size']['CC'][0], parameters['input_size']['CC'][1], 2),
        dtype=numpy.float32,
    )  # 2 types of views
    mlo_loader_buffer = numpy.zeros(
        shape=(parameters['input_size']['MLO'][0], parameters['input_size']['MLO'][1], 2),
        dtype=numpy.float32,
    )  # 2 types of views
    cc_cancer_heatmap_loader_buffer = numpy.zeros(
        shape=(parameters['input_size']['CC'][0], parameters['input_size']['CC'][1], 2,
               parameters["cancer_heatmap_channels"]),
        dtype=numpy.float32,
    )  # 2 types of views
    mlo_cancer_heatmap_loader_buffer = numpy.zeros(
        shape=(parameters['input_size']['MLO'][0], parameters['input_size']['MLO'][1], 2,
               parameters["cancer_heatmap_channels"]),
        dtype=numpy.float32,
    )  # 2 types of views

    L_CC_buffer = numpy.zeros(shape=5000 * 5000, dtype=numpy.uint16)
    R_CC_buffer = numpy.zeros(shape=5000 * 5000, dtype=numpy.uint16)
    L_MLO_buffer = numpy.zeros(shape=5000 * 5000, dtype=numpy.uint16)
    R_MLO_buffer = numpy.zeros(shape=5000 * 5000, dtype=numpy.uint16)
    L_CC_cancer_heatmap_buffer = numpy.zeros(shape=5000 * 5000, dtype=numpy.uint16)
    R_CC_cancer_heatmap_buffer = numpy.zeros(shape=5000 * 5000, dtype=numpy.uint16)
    L_MLO_cancer_heatmap_buffer = numpy.zeros(shape=5000 * 5000, dtype=numpy.uint16)
    R_MLO_cancer_heatmap_buffer = numpy.zeros(shape=5000 * 5000, dtype=numpy.uint16)

    reader_buffer = (L_CC_buffer, R_CC_buffer, L_MLO_buffer, R_MLO_buffer)
    cancer_heatmap_reader_buffer = (L_CC_cancer_heatmap_buffer, R_CC_cancer_heatmap_buffer,
                                    L_MLO_cancer_heatmap_buffer, R_MLO_cancer_heatmap_buffer)

    pid = mp.current_process().pid

    def log_print(s):
        print("[{}:{:%Y-%m-%d %H:%M:%S}] {}".format(pid, dt.datetime.now(), s))

    while True:
        item = q_data_to_load.get()
        if parameters['verbose_printing']:
            log_print('Meta data taken out of the queue.')
        
        
        if item is None:
            if parameters['verbose_printing']:
                log_print('Loader is finishing.')
            break

        data_index, metadata = item
        if parameters['verbose_printing']: log_print('item length = %d, data_index=%d'%(len(item), data_index))

        try:
            loaded_datum = load_datum_multiple_input_size(
                metadata=metadata,
                reader_buffer=reader_buffer,
                cancer_heatmap_reader_buffer=cancer_heatmap_reader_buffer,
                cc_loader_buffer=cc_loader_buffer, mlo_loader_buffer=mlo_loader_buffer,
                cc_cancer_heatmap_loader_buffer=cc_cancer_heatmap_loader_buffer,
                mlo_cancer_heatmap_loader_buffer=mlo_cancer_heatmap_loader_buffer,
                parameters=parameters, random_number_generator=random_number_generator,
                augmentation=augmentation, randomise_view=randomise_view,
            )
            
            if parameters['verbose_printing']: log_print('loaded_datum length = %d, label=%d'%(len(loaded_datum), loaded_datum[-1]))  

            shared_array_index = q_empty_indices.get()
            if parameters['verbose_printing']:
                log_print('Datum will be put in the shared array at index: {}'.format(shared_array_index))

            non_image_features, cc_images, mlo_images, cc_cancer_heatmap_images, mlo_cancer_heatmap_images, label = \
                loaded_datum  # images and loader_buffer are the same object

            # put this datum in the shared memory
            # B, H, W, C
            shared_array_list[0][shared_array_index, :, :, 0] = cc_images[:, :, 0]  # L_CC
            shared_array_list[1][shared_array_index, :, :, 0] = cc_images[:, :, 1]  # R_CC
            shared_array_list[2][shared_array_index, :, :, 0] = mlo_images[:, :, 0]  # L_MLO
            shared_array_list[3][shared_array_index, :, :, 0] = mlo_images[:, :, 1]  # R_MLO
            # B, C, H, W
            shared_array_cancer_heatmap[0][shared_array_index, :, :, :] = \
                numpy.moveaxis(cc_cancer_heatmap_images[:, :, 0, :], 2, 0)  # L_CC
            shared_array_cancer_heatmap[1][shared_array_index, :, :, :] =  \
                numpy.moveaxis(cc_cancer_heatmap_images[:, :, 1, :], 2, 0)
            shared_array_cancer_heatmap[2][shared_array_index, :, :, :] = \
                numpy.moveaxis(mlo_cancer_heatmap_images[:, :, 0, :], 2, 0)  # L_MLO
            shared_array_cancer_heatmap[3][shared_array_index, :, :, :] = \
                numpy.moveaxis(mlo_cancer_heatmap_images[:, :, 1, :], 2, 0)  # R_MLO

            loaded_datum_minus_images = (non_image_features, label)

            if parameters['verbose_printing']:
                log_print('Loaded datum.')
            q_data_loaded.put((data_index, shared_array_index, loaded_datum_minus_images))
            if parameters['verbose_printing']:
                log_print('Loaded datum put in the queue.')
        except (FileNotFoundError, KeyError, AssertionError, MemoryError):
            
            if parameters['verbose_printing']:
                log_print('Loading datum unsuccessful {}'.format(sys.exc_info()[0]))
            q_data_loaded.put((data_index, -1, None))
            if parameters['verbose_printing']:
                log_print('None put in the queue.')
            raise
        except Exception as err:
            if parameters['verbose_printing']:
                log_print('Error ' + str(type(err)))
        # except:
        #     log_print('other error in loader_multiple_input_size')   

        if parameters['verbose_printing']:
            log_print('There are currently {} data waiting in the queue.'.format(q_data_loaded.qsize()))


def loader(shared_array_images, shared_array_cancer_heatmap, q_data_to_load, q_data_loaded, q_empty_indices, parameters, random_seed, augmentation, randomise_view):

    random_number_generator = numpy.random.RandomState(random_seed)

    loader_buffer_image = numpy.zeros(shape = (parameters['input_size'][0], parameters['input_size'][1], 4), dtype = numpy.float32)

    while True:
        item = q_data_to_load.get()
        if parameters['verbose_printing']: print('Meta data taken out of the queue.')

        if item is None:
            if parameters['verbose_printing']: print('Loader is finishing.')
            break

        data_index, metadata = item

        try:
            loaded_datum = load_datum(metadata, loader_buffer_image, parameters, random_number_generator, augmentation, randomise_view)

            shared_array_index = q_empty_indices.get()
            if parameters['verbose_printing']: print('Datum will be put in the shared array at index: ' + str(shared_array_index))

            non_image_features, images, cancer_heatmaps, label = loaded_datum # images and loader_buffer are the same object

            # put this datum in the shared memory
            shared_array_images[shared_array_index, :, :, 0] = images[:, :, 0] # L_CC
            shared_array_images[shared_array_index, :, :, 1] = images[:, :, 1] # R_CC
            shared_array_images[shared_array_index, :, :, 2] = images[:, :, 2] # L_MLO
            shared_array_images[shared_array_index, :, :, 3] = images[:, :, 3] # R_MLO

            # put this datum in the shared memory
            shared_array_cancer_heatmap[shared_array_index, :, :, 0] = cancer_heatmaps[:, :, 0] # L_CC
            shared_array_cancer_heatmap[shared_array_index, :, :, 1] = cancer_heatmaps[:, :, 1] # R_CC
            shared_array_cancer_heatmap[shared_array_index, :, :, 2] = cancer_heatmaps[:, :, 2] # L_MLO
            shared_array_cancer_heatmap[shared_array_index, :, :, 3] = cancer_heatmaps[:, :, 3] # R_MLO

            loaded_datum_minus_images = (non_image_features, label)

            if parameters['verbose_printing']: print('Loaded datum.')
            q_data_loaded.put((data_index, shared_array_index, loaded_datum_minus_images))
            if parameters['verbose_printing']: print('Loaded datum put in the queue.')

        except (OSError, FileNotFoundError, KeyError, AssertionError, MemoryError):
            if parameters['verbose_printing']: print('Loading datum unsuccessful ' + str(sys.exc_info()[0]))
            q_data_loaded.put((data_index, -1, None))
            if parameters['verbose_printing']: print(str(None) + ' put in the queue.')

        if parameters['verbose_printing']: print('There are currently ' + str(q_data_loaded.qsize()) + ' data waiting in the queue.')
