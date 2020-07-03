import numpy as np
import numpy
from . import loading_cancer_heatmap
import multiprocessing as mp
import sys
import datetime as dt

def loader_feeder(data_list, q_data_to_load, number_of_loaders, parameters, random_seed, permute=True):

    random_number_generator = np.random.RandomState(random_seed)
    number_of_examples = len(data_list)

    if permute:
        indices = random_number_generator.permutation(number_of_examples)
    else:
        indices = range(0, number_of_examples)

    loader_feeder_with_given_index_multiple_samples(data_list, q_data_to_load, number_of_loaders, parameters, indices)

def loader_feeder_with_given_index_multiple_samples(data_list, q_data_to_load, number_of_loaders, parameters, indices):
    counter = 0
    datum = []
    for index in indices:
        metadata = data_list[index] 
        
        if parameters['verbose_printing']: print('Counter=', counter, '<=', parameters['minibatch_size'])
        
        if counter<parameters['minibatch_size']:
            datum.append((index, metadata))
            counter+=1
        else:
            
            if parameters['verbose_printing']: print('Producing meta data for index: ' + ', '.join([str(index) for index, _ in datum]) )
            
            q_data_to_load.put(datum)
        
            if parameters['verbose_printing']: print('Meta data for index ' + ', '.join([str(index) for index, _ in datum]) + ' put in the queue')
            counter = 0
            datum = []
            datum.append((index, metadata))
            counter+=1
    if len(datum)>0:
        if parameters['verbose_printing']: print('Producing meta data for index: ' + ', '.join([str(index) for index, _ in datum]) )
                
        q_data_to_load.put(datum)
        
        if parameters['verbose_printing']: print('Meta data for index ' + ', '.join([str(index) for index, _ in datum]) + ' put in the queue')
            


    for i in range(0, number_of_loaders):
        if parameters['verbose_printing']: print('All indices processed. Producing ending datum: ' + str(None))
        q_data_to_load.put(None)
        if parameters['verbose_printing']: print('Datum ' + str(None) + ' put in the queue')


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
        itemum = q_data_to_load.get()
        
        
        if itemum is None:
            if parameters['verbose_printing']:
                log_print('Loader is finishing.')
            break

        if parameters['verbose_printing']: print('Loading meta data for index: ' + ', '.join([str(index) for index, _ in itemum]) ) 
        if parameters['verbose_printing']: print('Meta data taken out of the queue.')
        
        data_indexum, shared_array_indexum, loaded_datum_minus_imagesum = [], [], []
        for data_index, metadata in itemum:
            if parameters['verbose_printing']: print('Processing meta data for index: ', data_index )
            if parameters['verbose_printing']: log_print('item length = %d, data_index=%d'%(len(itemum), data_index))

            try:
                loaded_datum = loading_cancer_heatmap.load_datum_multiple_input_size(
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

                if parameters['verbose_printing']: log_print('Loaded datum.')

                data_indexum.append(data_index)
                shared_array_indexum.append(shared_array_index) 
                loaded_datum_minus_imagesum.append(loaded_datum_minus_images)

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

        q_data_loaded.put((data_indexum, shared_array_indexum, loaded_datum_minus_imagesum))
        if parameters['verbose_printing']: log_print('Loaded datum put in the queue.')

