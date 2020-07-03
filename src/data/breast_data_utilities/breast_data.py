import numpy as np
import multiprocessing
import ctypes
import math
import json
import itertools
import queue
import datetime as dt
import time
import os
from scipy import misc
import sys
import gin
import cv2

from .breast_data_utils import unpickle_from_file, get_random_seed
from . import loading 
from . import loading_cancer_heatmap
from . import breast_data_utils
from . import config 
from . import loading_multisamples

MODELNAMES = ["AllViewModel", 
            "BlendedModel", 
            "TiedViewModel", 
            "MidFusedModel", 
            "ViewPoolingModel", 
            "GeneralViewPoolingModel", 
            "MeanViewPoolingModel",
            "ViewMergeModel",
            "DeepFusionModel",
            "DeepViewFusionModel",
            "EnsembleModel",
            "ModDropModel"
            ]

class Modes:
    BIRADS = "birads"
    CANCER = "cancer"
    CANCER_WITH_UNKNOWNS = "cancer_with_unknowns"

    MULTICLASS_CANCER = "multiclass_cancer"
    MULTICLASS_CANCER_WITH_UNKNOWNS = "multiclass_cancer_with_unknowns"

    MULTICLASS_CANCER_SIDES = "multiclass_cancer_sides"
    MULTICLASS_CANCER_WITH_UNKNOWNS_SIDES = "multiclass_cancer_with_unknowns_sides"

    MULTICLASS_CANCER_SIDES_COMBINED = "multiclass_cancer_sides_combined"
    MULTICLASS_CANCER_WITH_UNKNOWNS_SIDES_COMBINED = "multiclass_cancer_with_unknowns_sides_combined"

    _all = [
        BIRADS, CANCER, CANCER_WITH_UNKNOWNS,
        MULTICLASS_CANCER, MULTICLASS_CANCER_WITH_UNKNOWNS,
        MULTICLASS_CANCER_SIDES, MULTICLASS_CANCER_WITH_UNKNOWNS_SIDES,
        MULTICLASS_CANCER_SIDES_COMBINED, MULTICLASS_CANCER_WITH_UNKNOWNS_SIDES_COMBINED,
    ]

    @classmethod
    def contains(cls, elem):
        return elem in cls._all


def give_segmentation(seg_path, seg_metadata, parameters, generator, file_name_postfix):
    """
    Function that loads and preprocesses a segmentation map.
    @param seg_path: the path to .all.png folder
    @param seg_metadata: should be an element loaded from file_data_list.pkl
    @param parameters:
    @param generator:
    """
    output = []
    for view in ["L-CC", 'R-CC', 'L-MLO', 'R-MLO']:
        # locates folder of the segmentation
        seg_folder = seg_metadata[view][0]
        seg_file_path = os.path.join(seg_path, seg_folder) + file_name_postfix
        # load the segmentation if it exists
        if os.path.exists(seg_file_path):
            raw_seg = misc.imread(seg_file_path)
            # clean the image
            best_center = seg_metadata["best_center"][view][0]
            # convert it to binary (0 or 1)
            original_img = np.minimum(raw_seg, 1)
            # crop the image to fit the input_size in parameters
            clean_seg, _ = loading.get_single_image(original_img, view, parameters,
                                                    random_number_generator=generator,
                                                    augmentation=False, horizontal_flip="NO",
                                                    best_center=best_center)
            output.append(clean_seg)
        else:
            output.append(None)
    return output

def get_image_of_single_view(view, raw_data_batch):
    if view == 'cc':
        L_view_batch, R_view_batch, _, _ = raw_data_batch[0]
    elif view == 'mlo':
        _, _, L_view_batch, R_view_batch = raw_data_batch[0]
    else:
        raise ValueError('Invalid view (cc/mlo)')
    
    return ((np.append(L_view_batch, R_view_batch, axis = 0), None), raw_data_batch[1])

def get_image_of_both_view(raw_data_batch):
    L_CC_view_batch, R_CC_view_batch, L_MLO_view_batch, R_MLO_view_batch = raw_data_batch[0]
    CC_view_batch = np.append(L_CC_view_batch, R_CC_view_batch, axis = 0)
    MLO_view_batch = np.append(L_MLO_view_batch, R_MLO_view_batch, axis = 0)

    return ((CC_view_batch, MLO_view_batch), raw_data_batch[1])

class data:

    def __init__(self, logger, parameters, verbose=True, loaded_data_list=None):

        self.logger = logger
        self.parameters = parameters
        self.verbose = verbose
        assert 'augmentation_center' in self.parameters, "augmentation_center is not set"

        # creating the shared array to be able to load a few examples into memory at the same time
        self.shared_array_list = []
        for i in range(2):
            cc_shared_array_base = multiprocessing.Array(ctypes.c_float, self.parameters['data_buffer_size'] * self.parameters['input_size']['CC'][0] * self.parameters['input_size']['CC'][1])
            cc_shared_array_ctype = np.ctypeslib.as_array(cc_shared_array_base.get_obj())
            self.shared_array_list.append(cc_shared_array_ctype.reshape(self.parameters['data_buffer_size'], self.parameters['input_size']['CC'][0], self.parameters['input_size']['CC'][1], 1))
        for i in range(2):
            mlo_shared_array_base = multiprocessing.Array(ctypes.c_float, self.parameters['data_buffer_size'] * self.parameters['input_size']['MLO'][0] * self.parameters['input_size']['MLO'][1])
            mlo_shared_array_ctype = np.ctypeslib.as_array(mlo_shared_array_base.get_obj())
            self.shared_array_list.append(mlo_shared_array_ctype.reshape(self.parameters['data_buffer_size'], self.parameters['input_size']['MLO'][0], self.parameters['input_size']['MLO'][1], 1))

        if loaded_data_list is None:
            self._log("Unpickling... ", no_enter=True)
            start = time.time()
            self.logger.flush()
            pickle_data_list = unpickle_from_file(self.parameters['file_data_list'])
            self._log("Done [{:.1f}]".format(time.time()-start))
            self.logger.flush()
        else:
            pickle_data_list = loaded_data_list
        self.data_list_training, self.data_list_validation, self.data_list_test = pickle_data_list[:3]
        if len(pickle_data_list) == 4:
            self.cropping_metadata = pickle_data_list[3]
            if (self.parameters['augmentation_center'] == 'best_center') and \
               ('check_input_size' in self.parameters and self.parameters['check_input_size']):
                assert(self.cropping_metadata['dim_map']['L-CC'] == self.parameters['input_size']['CC'])
                assert(self.cropping_metadata['dim_map']['R-CC'] == self.parameters['input_size']['CC'])
                assert(self.cropping_metadata['dim_map']['L-MLO'] == self.parameters['input_size']['MLO'])
                assert(self.cropping_metadata['dim_map']['R-MLO'] == self.parameters['input_size']['MLO'])

        # using only a fraction of the training set
        number_of_all_training_examples = len(self.data_list_training)
        number_of_remaining_training_examples = math.ceil(number_of_all_training_examples * parameters['training_fraction'])
        self.data_list_training = self.data_list_training[0 : number_of_remaining_training_examples]

        if "validation_fraction" in parameters:
            number_of_all_validation_examples = len(self.data_list_validation)
            number_of_remaining_validation_examples = math.ceil(
                number_of_all_validation_examples * parameters['validation_fraction'])
            self.data_list_validation = self.data_list_validation[0: number_of_remaining_validation_examples]

        if 'data_number_divisibility' in parameters:
            training_number = len(self.data_list_training) - (len(self.data_list_training) % parameters['data_number_divisibility'])
            self.data_list_training = self.data_list_training[0 : training_number]

            validation_number = len(self.data_list_validation) - (len(self.data_list_validation) % parameters['data_number_divisibility'])
            self.data_list_validation = self.data_list_validation[0 : validation_number]

            test_number = len(self.data_list_test) - (len(self.data_list_test) % parameters['data_number_divisibility'])
            self.data_list_test = self.data_list_test[0 : test_number]

        '''
        if 'best_center_cropping' in self.parameters and self.parameters['best_center_cropping']:
            best_center_tuple = pickling.unpickle_from_file(self.parameters['file_best_center_list'])
            self.best_center_list_training, self.best_center_list_validation, self.best_center_list_test = best_center_tuple[0], best_center_tuple[1], best_center_tuple[2]
        else:
            self.best_center_list_training, self.best_center_list_validation, self.best_center_list_test = None, None, None
        '''

        self.current_training_index = 0
        self.current_validation_index = 0
        self.current_test_index = 0
        self.current_observation_index = 0

        self.data_prefix = self.parameters['data_prefix']
        self.list_of_features = self.parameters['non_image_features']
        if "train_sampling_mode" in self.parameters:
            self.train_sampler = DataSampler(
                sampling_mode=self.parameters["train_sampling_mode"],
                sampling_mode_config=self.parameters["train_sampling_mode_config"],
                logger=logger,
            )
        else:
            self.train_sampler = DataSampler.normal(logger)

        self.q_data_to_load = None
        self.q_data_loaded = None
        self.q_empty_indices = None

        self.L_CC_view_batch_buffer = np.zeros(shape = (parameters['minibatch_size'], self.parameters['input_size']['CC'][0], self.parameters['input_size']['CC'][1], 1), dtype = np.float32)
        self.R_CC_view_batch_buffer = np.zeros(shape = (parameters['minibatch_size'], self.parameters['input_size']['CC'][0], self.parameters['input_size']['CC'][1], 1), dtype = np.float32)
        self.L_MLO_view_batch_buffer = np.zeros(shape = (parameters['minibatch_size'], self.parameters['input_size']['MLO'][0], self.parameters['input_size']['MLO'][1], 1), dtype = np.float32)
        self.R_MLO_view_batch_buffer = np.zeros(shape = (parameters['minibatch_size'], self.parameters['input_size']['MLO'][0], self.parameters['input_size']['MLO'][1], 1), dtype = np.float32)

        self.all_threads = []
        self.number_of_loaders = self.parameters['number_of_loaders']

        self._log("Created an object holding a data set.")
        self._log("Using " + str(number_of_remaining_training_examples) + " training examples.")

        self._train_labels = None
        self._validation_labels = None
        self._test_labels = None

        self.training_data_length = None
        self.validation_data_length = None
        self.test_data_length = None

        # To-do [zp489]: Make data class general
        self._train_labels_cancer = None
        self._validation_labels_cancer = None
        self._test_labels_cancer = None

    def _log(self, *args, **kwargs):
        if self.verbose:
            self.logger.log(*args, **kwargs)

    @staticmethod
    def map_label(label):
        if (label == 0) or (label == 4) or (label == 5):
            return 0
        elif (label == 1):
            return 1
        elif (label == 2) or (label == 3):
            return 2

    def training_size(self):
        return len(self.data_list_training)

    def _get_labels(self, data_list, mode):
        number_of_examples = len(data_list)
        if mode in (Modes.BIRADS, Modes.CANCER, Modes.CANCER_WITH_UNKNOWNS):
            if mode == Modes.BIRADS:
                key = "label"
                number_of_classes = config.number_of_classes
            elif mode == Modes.CANCER:
                key = "cancer_label"
                number_of_classes = config.number_of_cancer_classes
            elif mode == Modes.CANCER_WITH_UNKNOWNS:
                key = "cancer_label"
                number_of_classes = config.number_of_cancer_classes_with_unknown
            else:
                raise KeyError(mode)

            labels = np.zeros(shape=(number_of_examples, number_of_classes), dtype=np.int32)
            for i in range(0, number_of_examples):
                labels[i, self.map_label(data_list[i][key])] = 1
        elif mode in (Modes.MULTICLASS_CANCER, Modes.MULTICLASS_CANCER_WITH_UNKNOWNS):
            number_of_classes = 2
            number_of_label_sets = 2
            labels = np.zeros(shape=(number_of_examples, number_of_label_sets, number_of_classes), dtype=np.int32)
            for i in range(0, number_of_examples):
                labels[i, 0, data_list[i]["cancer_label"]["benign"]] = 1
                labels[i, 1, data_list[i]["cancer_label"]["malignant"]] = 1
        elif mode in (Modes.MULTICLASS_CANCER_SIDES, Modes.MULTICLASS_CANCER_WITH_UNKNOWNS_SIDES):
            number_of_label_sets = 4
            number_of_classes = 2
            labels = np.zeros(shape=(number_of_examples, number_of_label_sets, number_of_classes), dtype=np.int32)
            for i in range(0, number_of_examples):
                labels[i, 0, data_list[i]["cancer_label"]["left_benign"]] = 1
                labels[i, 1, data_list[i]["cancer_label"]["right_benign"]] = 1
                labels[i, 2, data_list[i]["cancer_label"]["left_malignant"]] = 1
                labels[i, 3, data_list[i]["cancer_label"]["right_malignant"]] = 1
        elif mode in (Modes.MULTICLASS_CANCER_SIDES_COMBINED, Modes.MULTICLASS_CANCER_WITH_UNKNOWNS_SIDES_COMBINED):
            number_of_label_sets = 2
            number_of_classes = 2
            labels = np.zeros(
                shape=(number_of_examples, number_of_label_sets, number_of_classes),
                dtype=np.int32,
            )
            for i in range(0, number_of_examples):
                left_positive = (
                    data_list[i]['cancer_label']["left_benign"]
                    or data_list[i]['cancer_label']["left_malignant"]
                )
                right_positive = (
                    data_list[i]['cancer_label']["right_benign"]
                    or data_list[i]['cancer_label']["right_malignant"]
                )
                labels[i, 0, left_positive] = 1
                labels[i, 1, right_positive] = 1
        else:
            raise KeyError(mode)
        return labels

    def get_train_labels(self):
        if self._train_labels is None:
            self._train_labels = self._get_labels(self.data_list_training, mode=Modes.BIRADS)
        return self._train_labels

    def get_validation_labels(self):
        if self._validation_labels is None:
            self._validation_labels = self._get_labels(self.data_list_validation, mode=Modes.BIRADS)
        return self._validation_labels

    def get_test_labels(self):
        if self._test_labels is None:
            self._test_labels = self._get_labels(self.data_list_test, mode=Modes.BIRADS)
        return self._test_labels

    def get_train_labels_cancer(self, mode):
        assert Modes.contains(mode)
        if self._train_labels_cancer is None:
            self._train_labels_cancer = self._get_labels(self.data_list_training, mode=mode)
        return self._train_labels_cancer

    def get_validation_labels_cancer(self, mode):
        assert Modes.contains(mode)
        if self._validation_labels_cancer is None:
            self._validation_labels_cancer = self._get_labels(self.data_list_validation, mode=mode)
        return self._validation_labels_cancer

    def get_test_labels_cancer(self, mode):
        assert Modes.contains(mode)
        if self._test_labels_cancer is None:
            self._test_labels_cancer = self._get_labels(self.data_list_test, mode=mode)
        return self._test_labels_cancer

    def get_mrns_and_accession_numbers(self, indices, part_of_data):

        mrns_and_accession_numbers = []

        assert part_of_data in ['training', 'validation', 'test'], 'Wrong part of data'

        if part_of_data == 'training':
            data = self.data_list_training
        elif part_of_data == 'validation':
            data = self.data_list_validation
        elif part_of_data == 'test':
            data = self.data_list_test
        else:
            raise KeyError(part_of_data)

        for index in indices:
            datum = data[index]
            mrn = datum['patient_ID']
            accession_number = datum['accession_number']

            mrns_and_accession_numbers.append((index, mrn, accession_number))

        return mrns_and_accession_numbers

    def get_non_image_data(self, indices, part_of_data):

        non_image_data = dict()

        assert part_of_data in ['training', 'validation', 'test'], 'Wrong part of data'

        if part_of_data == 'training':
            data = self.data_list_training
        elif part_of_data == 'validation':
            data = self.data_list_validation
        elif part_of_data == 'test':
            data = self.data_list_test

        for index in indices:
            non_image_data[index] = data[index]

        return non_image_data

    def get_segmentation_minibatch(self, indices, part_of_data):
        """
        Method that loads the segmentation if available.
        Assumption: no augmentation happens here.
        :param indices: list of integer
        :param part_of_data: phase in ['training', 'validation', 'test']
        :return: list of 4*N elements in order of lcc->rcc->lmlo->rmlo, if no segmentation None will be returned
        """
        # make sure the assumption that no stochasticity is true
        assert "segmentation_dir" in self.parameters, "Requires segmentation_dir in parameters"
        assert part_of_data in ["validation", "visualization", "test"], "Only support visualization, validation, and test"
        assert not self.parameters["stochasticity"][part_of_data], "Stochasticity at {0} should be false when using get_segmentation_minibatch()".format(part_of_data)
        # decide which meta_data_list to look at
        if part_of_data == 'training':
            dataset = self.data_list_training
        elif part_of_data == 'validation':
            dataset = self.data_list_validation
        elif part_of_data == 'test':
            dataset = self.data_list_test
        elif part_of_data == 'visualization':
            dataset = self.data_list_visualisation
        else:
            raise KeyError(part_of_data)
        # TODO: generator is trivial since we assumed no augmentation
        generator = np.random.RandomState(0)
        # collect and re-organize the segmentations
        all_lcc = []
        all_rcc = []
        all_lmlo = []
        all_rmlo = []
        for idx in indices:
            # all segmentation
            lcc, rcc, lmlo, rmlo = give_segmentation(self.parameters["segmentation_dir"], dataset[idx],
                              self.parameters, generator, ".all.png")
            # benign only
            lcc_b, rcc_b, lmlo_b, rmlo_b = give_segmentation(self.parameters["segmentation_dir"], dataset[idx],
                                                     self.parameters, generator, ".benign.png")
            # maligant only
            lcc_m, rcc_m, lmlo_m, rmlo_m = give_segmentation(self.parameters["segmentation_dir"], dataset[idx],
                                                             self.parameters, generator, ".malignant.png")
            all_lcc.append([lcc, lcc_b, lcc_m])
            all_rcc.append([rcc, rcc_b, rcc_m])
            all_lmlo.append([lmlo, lmlo_b, lmlo_m])
            all_rmlo.append([rmlo, rmlo_b, rmlo_m])
        return all_lcc+all_rcc+all_lmlo+all_rmlo

    def get_cancer_label_minibatch(self, indices, part_of_data, mode=Modes.CANCER):

        if part_of_data == 'training':
            dataset = self.data_list_training
        elif part_of_data == 'validation':
            dataset = self.data_list_validation
        elif part_of_data == 'test':
            dataset = self.data_list_test
        else:
            raise KeyError(part_of_data)

        if mode in (Modes.CANCER, Modes.CANCER_WITH_UNKNOWNS):
            if mode == Modes.CANCER:
                number_of_classes = config.number_of_cancer_classes
            elif mode == Modes.CANCER_WITH_UNKNOWNS:
                number_of_classes = config.number_of_cancer_classes_with_unknown
            else:
                raise KeyError(mode)
            cancer_label_batch = np.zeros(
                shape=(len(indices), number_of_classes),
                dtype=np.int32,
            )
            for i, index in enumerate(indices):
                cancer_label_batch[i, dataset[index]['cancer_label']] = 1
        elif mode in (Modes.MULTICLASS_CANCER, Modes.MULTICLASS_CANCER_WITH_UNKNOWNS):
            number_of_label_sets = 2
            number_of_classes = 2
            cancer_label_batch = np.zeros(
                shape=(len(indices), number_of_label_sets, number_of_classes),
                dtype=np.int32,
            )
            for i, index in enumerate(indices):
                cancer_label_batch[i, 0, dataset[index]['cancer_label']["benign"]] = 1
                cancer_label_batch[i, 1, dataset[index]['cancer_label']["malignant"]] = 1
        elif mode in (Modes.MULTICLASS_CANCER_SIDES, Modes.MULTICLASS_CANCER_WITH_UNKNOWNS_SIDES):
            number_of_label_sets = 4
            number_of_classes = 2
            cancer_label_batch = np.zeros(
                shape=(len(indices), number_of_label_sets, number_of_classes),
                dtype=np.int32,
            )
            for i, index in enumerate(indices):
                cancer_label_batch[i, 0, dataset[index]['cancer_label']["left_benign"]] = 1
                cancer_label_batch[i, 1, dataset[index]['cancer_label']["right_benign"]] = 1
                cancer_label_batch[i, 2, dataset[index]['cancer_label']["left_malignant"]] = 1
                cancer_label_batch[i, 3, dataset[index]['cancer_label']["right_malignant"]] = 1
        elif mode in (Modes.MULTICLASS_CANCER_SIDES_COMBINED, Modes.MULTICLASS_CANCER_WITH_UNKNOWNS_SIDES_COMBINED):
            number_of_label_sets = 2
            number_of_classes = 2
            cancer_label_batch = np.zeros(
                shape=(len(indices), number_of_label_sets, number_of_classes),
                dtype=np.int32,
            )
            for i, index in enumerate(indices):
                left_positive = (
                    dataset[index]['cancer_label']["left_benign"]
                    or dataset[index]['cancer_label']["left_malignant"]
                )
                right_positive = (
                    dataset[index]['cancer_label']["right_benign"]
                    or dataset[index]['cancer_label']["right_malignant"]
                )
                cancer_label_batch[i, 0, left_positive] = 1
                cancer_label_batch[i, 1, right_positive] = 1
        else:
            raise KeyError(mode)

        return cancer_label_batch

    def start_epoch(self, data_list, phase, labels, permute, augmentation, randomise_view, random_seed):

        self.q_data_to_load = multiprocessing.Queue(maxsize=self.parameters['metadata_buffer_size'])
        self.q_data_loaded = multiprocessing.Queue(maxsize=self.parameters['metadata_buffer_size'])
        self.q_empty_indices = multiprocessing.Queue(maxsize=self.parameters['data_buffer_size'])

        for i in range(0, self.parameters['data_buffer_size']):
            self.q_empty_indices.put(i)

        self.all_threads = []

        if phase == "training" and self.train_sampler.sampling_mode != "normal":
            assert permute
            oversampled_indices = self.train_sampler.sample_indices(labels, random_seed=random_seed)
            feeder_thread = multiprocessing.Process(
                target=loading.loader_feeder_with_given_index,
                args=[data_list, self.q_data_to_load, self.number_of_loaders, self.parameters, oversampled_indices]
            )
            data_length = len(oversampled_indices)
            self.logger.log('Assigned self.data_observation_list_indices to oversampled_indices.') 
            self.data_observation_list_indices = oversampled_indices
            number_of_loaders = self.number_of_loaders
        else:
            loader_feeder_random_seed = random_seed
            number_of_loaders = self.number_of_loaders if phase!='training_observation' else 1
            feeder_thread = multiprocessing.Process(
                target=loading.loader_feeder,
                args=[data_list, self.q_data_to_load, number_of_loaders, self.parameters, loader_feeder_random_seed, permute]
            )
            data_length = len(data_list)
        feeder_thread.start()
        self.all_threads.append(feeder_thread)

        for i in range(0, number_of_loaders):

            loader_random_seed = random_seed + i + 1

            if self.parameters['features'] == 'only image':
                loader_thread = multiprocessing.Process(
                    target=loading.loader_multiple_input_size,
                    args=[
                        self.shared_array_list, self.q_data_to_load, self.q_data_loaded, self.q_empty_indices,
                        self.parameters, loader_random_seed, augmentation, randomise_view,
                    ])
            elif self.parameters['features'] == 'only non-image':
                loader_thread = multiprocessing.Process(
                    target=loading.non_image_loader,
                    args=[self.q_data_to_load, self.q_data_loaded, self.parameters],
                )
            else:
                raise KeyError(self.parameters['features'])

            loader_thread.start()
            self.all_threads.append(loader_thread)

        return {
            "data_length": data_length,
        }

    def start_training_epoch(self, random_seed, mode=Modes.BIRADS):
        labels_func = {
            Modes.BIRADS: self.get_train_labels,
            Modes.CANCER: lambda: self.get_train_labels_cancer(mode),
            Modes.CANCER_WITH_UNKNOWNS: lambda: self.get_train_labels_cancer(mode),
            Modes.MULTICLASS_CANCER: lambda: self.get_train_labels_cancer(mode),
            Modes.MULTICLASS_CANCER_WITH_UNKNOWNS: lambda: self.get_train_labels_cancer(mode),
            Modes.MULTICLASS_CANCER_SIDES: lambda: self.get_train_labels_cancer(mode),
            Modes.MULTICLASS_CANCER_WITH_UNKNOWNS_SIDES: lambda: self.get_train_labels_cancer(mode),
            Modes.MULTICLASS_CANCER_SIDES_COMBINED: lambda: self.get_train_labels_cancer(mode),
            Modes.MULTICLASS_CANCER_WITH_UNKNOWNS_SIDES_COMBINED: lambda: self.get_train_labels_cancer(mode),
        }
        self.current_training_index = 0
        metadata = self.start_epoch(
            data_list=self.data_list_training,
            phase="training",
            labels=labels_func[mode](), 
            permute=True, #keep data always same when IOL
            augmentation=self.parameters['stochasticity']['training'],
            randomise_view=self.parameters['stochasticity']['training'],
            random_seed=random_seed,
        )
        self.training_data_length = metadata["data_length"]

    def start_validation_epoch(self, random_seed, mode=Modes.BIRADS):
        labels_func = {
            Modes.BIRADS: self.get_validation_labels,
            Modes.CANCER: lambda: self.get_validation_labels_cancer(mode),
            Modes.CANCER_WITH_UNKNOWNS: lambda: self.get_validation_labels_cancer(mode),
            Modes.MULTICLASS_CANCER: lambda: self.get_validation_labels_cancer(mode),
            Modes.MULTICLASS_CANCER_WITH_UNKNOWNS: lambda: self.get_validation_labels_cancer(mode),
            Modes.MULTICLASS_CANCER_SIDES: lambda: self.get_validation_labels_cancer(mode),
            Modes.MULTICLASS_CANCER_WITH_UNKNOWNS_SIDES: lambda: self.get_validation_labels_cancer(mode),
            Modes.MULTICLASS_CANCER_SIDES_COMBINED: lambda: self.get_validation_labels_cancer(mode),
            Modes.MULTICLASS_CANCER_WITH_UNKNOWNS_SIDES_COMBINED: lambda: self.get_validation_labels_cancer(mode),
        }
        self.current_validation_index = 0
        metadata = self.start_epoch(
            data_list=self.data_list_validation,
            phase="validation",
            labels=labels_func[mode](),
            permute=False,
            augmentation=self.parameters['stochasticity']['validation'],
            randomise_view=self.parameters['stochasticity']['validation'],
            random_seed=random_seed,
        )
        self.validation_data_length = metadata["data_length"]

    def start_test_epoch(self, random_seed, mode=Modes.BIRADS):
        labels_func = {
            Modes.BIRADS: self.get_test_labels,
            Modes.CANCER: lambda: self.get_test_labels_cancer(mode),
            Modes.CANCER_WITH_UNKNOWNS: lambda: self.get_test_labels_cancer(mode),
            Modes.MULTICLASS_CANCER: lambda: self.get_test_labels_cancer(mode),
            Modes.MULTICLASS_CANCER_WITH_UNKNOWNS: lambda: self.get_test_labels_cancer(mode),
            Modes.MULTICLASS_CANCER_SIDES: lambda: self.get_test_labels_cancer(mode),
            Modes.MULTICLASS_CANCER_WITH_UNKNOWNS_SIDES: lambda: self.get_test_labels_cancer(mode),
            Modes.MULTICLASS_CANCER_SIDES_COMBINED: lambda: self.get_test_labels_cancer(mode),
            Modes.MULTICLASS_CANCER_WITH_UNKNOWNS_SIDES_COMBINED: lambda: self.get_test_labels_cancer(mode),
        }
        self.current_test_index = 0
        metadata = self.start_epoch(
            data_list=self.data_list_test,
            phase="test",
            labels=labels_func[mode](),
            permute=False,
            augmentation=self.parameters['stochasticity']['test'],
            randomise_view=self.parameters['stochasticity']['test'],
            random_seed=random_seed,
        )
        self.test_data_length = metadata["data_length"]

    def finish_epoch(self):
        for thread in self.all_threads:
            thread.join()

    def training_epoch_done(self):
        return self.current_training_index == self.training_data_length

    def validation_epoch_done(self):
        return self.current_validation_index == self.validation_data_length

    def test_epoch_done(self):
        return self.current_test_index == self.test_data_length

    # def process_data_batch(self, raw_data_batch):
    #     if self.parameters['features'] == 'only image':
    #         return raw_data_batch
    #     elif self.parameters['features'] == 'only non-image':
    #         return self.process_data_non_image_batch(raw_data_batch)

    def give_minibatch(self, current_index, data_length, size_of_the_minibatch):

        # getting the items from the queue

        minibatch = []
        indices = []
        read_counter = 0

        while (read_counter < size_of_the_minibatch) and (current_index < data_length):
            data_index, shared_array_index, datum_minus_images = self.q_data_loaded.get()

            if datum_minus_images is not None:
                non_image_features, label = datum_minus_images

                minibatch.append((non_image_features, shared_array_index, label))
                indices.append(data_index)

                self._log(
                    'Loaded data of an index: ' + str(data_index) + ' number: ' + str(read_counter + 1)
                    + ' in this minibatch. In this epoch seen '
                    + str(current_index + 1) + ' / ' + str(data_length) + '.'
                )
            else:
                self._log(
                    'Loading data of an index: ' + str(data_index) + ' number: ' + str(read_counter + 1)
                    + ' in this minibatch unsuccessful. In this epoch seen '
                    + str(current_index + 1) + ' / ' + str(data_length) + '.'
                )

            read_counter = read_counter + 1
            current_index = current_index + 1

        # copying data from the shared memory and releasing the space in the shared memory

        number_of_correctly_loaded_examples = len(minibatch)

        # creating buffers of appropriate sizes by using previously allocated memory
        L_CC_view_batch = self.L_CC_view_batch_buffer[0 : number_of_correctly_loaded_examples, :, :, :]
        R_CC_view_batch = self.R_CC_view_batch_buffer[0 : number_of_correctly_loaded_examples, :, :, :]
        L_MLO_view_batch = self.L_MLO_view_batch_buffer[0 : number_of_correctly_loaded_examples, :, :, :]
        R_MLO_view_batch = self.R_MLO_view_batch_buffer[0 : number_of_correctly_loaded_examples, :, :, :]

        label_batch = np.zeros(shape=(number_of_correctly_loaded_examples, config.number_of_classes), dtype=np.int32)

        write_counter = 0

        for datum in minibatch:
            _, shared_array_index, label = datum
            label = self.map_label(label)

            # read from the shared memory
            L_CC_view_batch[write_counter, :, :, 0] = self.shared_array_list[0][shared_array_index, :, :, 0]
            R_CC_view_batch[write_counter, :, :, 0] = self.shared_array_list[1][shared_array_index, :, :, 0]
            L_MLO_view_batch[write_counter, :, :, 0] = self.shared_array_list[2][shared_array_index, :, :, 0]
            R_MLO_view_batch[write_counter, :, :, 0] = self.shared_array_list[3][shared_array_index, :, :, 0]

            label_batch[write_counter, label] = 1
            write_counter = write_counter + 1

            self.q_empty_indices.put(shared_array_index)
            self._log('Releasing the index in the shared array: ' + str(shared_array_index), print_log = False)

        images_batch = (L_CC_view_batch, R_CC_view_batch, L_MLO_view_batch, R_MLO_view_batch)

        return current_index, indices, (images_batch, label_batch)

    def give_non_image_minibatch(self, current_index, data_length, size_of_the_minibatch):

        minibatch = []
        indices = []
        counter = 0

        while (counter < size_of_the_minibatch) and (current_index < data_length):
            data_index, datum_minus_images = self.q_data_loaded.get()
            if datum_minus_images is not None:
                minibatch.append(datum_minus_images)
                indices.append(data_index)

                #self._log('Loaded data of an index: ' + str(data_index)  + ' number: ' + str(counter + 1) + ' in this minibatch. In this epoch seen ' + str(current_index + 1) + ' / ' + str(data_length) + '.')
            else:
                self._log('Loading data of an index: ' + str(data_index)  + ' number: ' + str(counter + 1) + ' in this minibatch unsuccessful. In this epoch seen ' + str(current_index + 1) + ' / ' + str(data_length) + '.')

            counter = counter + 1
            current_index = current_index + 1

        return current_index, indices, minibatch

    def process_data_non_image_batch(self, raw_data_batch):

        size_of_the_minibatch = len(raw_data_batch)
        number_of_features = len(self.list_of_features)
        counter = 0

        non_image_batch = np.zeros(shape = (size_of_the_minibatch, number_of_features), dtype = np.float32)
        label_batch = np.zeros(shape = (size_of_the_minibatch, config.number_of_classes), dtype = np.int32)

        for datum in raw_data_batch:
            non_image_features, label = datum
            label = self.map_label(label)

            non_image_batch[counter, :] = non_image_features
            label_batch[counter, label] = 1

            counter = counter + 1

        return (non_image_batch, label_batch)

    def give_training_minibatch(self):

        while not self.training_epoch_done():
            if self.parameters['features'] == 'only image':
                give_minibatch_func = self.give_minibatch
            elif self.parameters['features'] == 'only non-image':
                give_minibatch_func = self.give_non_image_minibatch
            else:
                raise KeyError(self.parameters['features'])

            self.current_training_index, indices, minibatch = give_minibatch_func(
                current_index=self.current_training_index,
                data_length=self.training_data_length,
                size_of_the_minibatch=self.parameters['minibatch_size'],
            )
            yield indices, minibatch
        raise StopIteration

    def give_validation_minibatch(self):

        while not self.validation_epoch_done():
            if self.parameters['features'] == 'only image':
                give_minibatch_func = self.give_minibatch
            elif self.parameters['features'] == 'only non-image':
                give_minibatch_func = self.give_non_image_minibatch
            else:
                raise KeyError(self.parameters['features'])

            self.current_validation_index, indices, minibatch = give_minibatch_func(
                current_index=self.current_validation_index,
                data_length=self.validation_data_length,
                size_of_the_minibatch=self.parameters['minibatch_size'],
            )
            yield indices, minibatch
        raise StopIteration

    def give_test_minibatch(self):

        while not self.test_epoch_done():
            if self.parameters['features'] == 'only image':
                give_minibatch_func = self.give_minibatch
            elif self.parameters['features'] == 'only non-image':
                give_minibatch_func = self.give_non_image_minibatch
            else:
                raise KeyError(self.parameters['features'])

            self.current_test_index, indices, minibatch = give_minibatch_func(
                current_index=self.current_test_index,
                data_length=self.test_data_length,
                size_of_the_minibatch=self.parameters['minibatch_size'],
            )
            yield indices, minibatch
        raise StopIteration

    def start_observation_epoch(self, image_phase = 'training_observation', random_seed = 0):
        self.current_observation_index = 0
        #always using self.data_observation_list_indices which is equal to oversampled_indices when starting training epoch

        self.data_list_observation = [self.data_list_training[i] for i in self.data_observation_list_indices]

        self.start_epoch(self.data_list_observation, phase=image_phase, labels=None,
                         permute = False, augmentation = False, randomise_view = False, random_seed = random_seed)
    
    def observation_epoch_done(self):
        return self.current_observation_index == len(self.data_list_observation)

    def give_observation_minibatch(self, minibatch_size):

        while not self.observation_epoch_done():
            
            self.current_observation_index, indices, minibatch = self.give_minibatch(
                current_index=self.current_observation_index,
                data_length=len(self.data_list_observation),
                size_of_the_minibatch=minibatch_size,#self.parameters['minibatch_size'],
                multiple=True
            )
            yield indices, minibatch

        raise StopIteration

    def process_data_batch(self, raw_data_batch):
        
        if 'SingleViewModel' in self.model:
            return get_image_of_single_view(self.view, raw_data_batch)
        elif self.model in MODELNAMES:
            return get_image_of_both_view(raw_data_batch)
        elif self.parameters['features'] == 'only non-image':
            return process_data_non_image_batch(raw_data_batch)
        else:
            print('Model type is not recorded', self.model)

    def get_cancer_label_minibatch_by_breast(self, indices, part_of_data, ):
                                             #mode=breast_data.Modes.MULTICLASS_CANCER_WITH_UNKNOWNS_SIDES):
        if part_of_data == 'training':
            dataset = self.data_list_training
        elif part_of_data == 'validation':
            dataset = self.data_list_validation
        elif part_of_data == 'test':
            dataset = self.data_list_test
        elif part_of_data == 'observation':
            dataset = self.data_list_observation
        else:
            raise KeyError(part_of_data)

        number_of_label_sets = 2
        number_of_classes = 2
        cancer_label_batch_by_breast = np.zeros(
            shape=(len(indices)*2, number_of_label_sets, number_of_classes),
            dtype=np.int32,
        )

        for i, index in enumerate(indices):
            cancer_label_batch_by_breast[i, 0, dataset[index]['cancer_label']["left_benign"]] = 1
            cancer_label_batch_by_breast[i, 1, dataset[index]['cancer_label']["left_malignant"]] = 1
            cancer_label_batch_by_breast[i+len(indices), 0, dataset[index]['cancer_label']["right_benign"]] = 1
            cancer_label_batch_by_breast[i+len(indices), 1, dataset[index]['cancer_label']["right_malignant"]] = 1

        return cancer_label_batch_by_breast

    def label_exam_transform_to_label_breast(self, labels):
        '''
        Args: labels - shape (number of exams, 4, 2), which is the label array returned when mode is Modes.MULTICLASS_CANCER_SIDES
        Returns: labels_breasts - shape (2*number of exams, 2, 2), 
                 from 0:number of exams are label for left breast
                 from number of exams:  are label for right breast
        '''
        labels_breasts = np.zeros((labels.shape[0]*2, 2, 2))

        labels_breasts[:labels.shape[0], 0] = labels[:, 0] #left benign
        labels_breasts[:labels.shape[0], 1] = labels[:, 2] #left malignant

        labels_breasts[labels.shape[0]:, 0] = labels[:, 1] #right benign
        labels_breasts[labels.shape[0]:, 1] = labels[:, 3] #right malignant

        return labels_breasts

    def get_train_labels_cancer_by_breast(self, mode):
        return self.label_exam_transform_to_label_breast(self.get_train_labels_cancer(mode))

    def get_validation_labels_cancer_by_breast(self, mode):
        return self.label_exam_transform_to_label_breast(self.get_validation_labels_cancer(mode))

    def get_test_labels_cancer_by_breast(self, mode):
        return self.label_exam_transform_to_label_breast(self.get_test_labels_cancer(mode))


class data_with_segmentations(data):
    """
    Segmentations will be [B,C,H,W] instead of the usual [B,H,W,C], because it's faster to copy data over this way
    """

    def __init__(self, logger, parameters, verbose=True, loaded_data_list=None):
        super().__init__(logger, parameters, verbose, loaded_data_list)
        self.loader_input_channels = self.parameters['loader_input_channels'] if 'loader_input_channels' in self.parameters else self.parameters['input_channels']
        self.shared_cancer_heatmap_array_list = []
        for i in range(2):
            cc_shared_array_base_cancer_heatmap = multiprocessing.Array(
                ctypes.c_float,
                self.parameters['data_buffer_size']
                * self.parameters["cancer_heatmap_channels"]
                * self.parameters['input_size']['CC'][0]
                * self.parameters['input_size']['CC'][1],
            )
            cc_shared_array_ctype_cancer_heatmap = np.ctypeslib.as_array(cc_shared_array_base_cancer_heatmap.get_obj())
            self.shared_cancer_heatmap_array_list.append(cc_shared_array_ctype_cancer_heatmap.reshape(
                self.parameters['data_buffer_size'],
                self.parameters["cancer_heatmap_channels"],
                self.parameters['input_size']['CC'][0],
                self.parameters['input_size']['CC'][1],
            ))
        for i in range(2):
            mlo_shared_array_base_cancer_heatmap = multiprocessing.Array(
                ctypes.c_float,
                self.parameters['data_buffer_size']
                * self.parameters["cancer_heatmap_channels"]
                * self.parameters['input_size']['MLO'][0]
                * self.parameters['input_size']['MLO'][1],
            )
            mlo_shared_array_ctype_cancer_heatmap = np.ctypeslib.as_array(mlo_shared_array_base_cancer_heatmap.get_obj())
            self.shared_cancer_heatmap_array_list.append(mlo_shared_array_ctype_cancer_heatmap.reshape(
                self.parameters['data_buffer_size'],
                self.parameters["cancer_heatmap_channels"],
                self.parameters['input_size']['MLO'][0],
                self.parameters['input_size']['MLO'][1],
            ))

        # Overwrite buffers with more channels
        # Generally, input_channels = 1 + cancer_heatmap_channels
        # Hack-ish
        self.L_CC_view_batch_buffer = np.zeros(
            shape=(
                parameters['minibatch_size'],
                self.loader_input_channels,
                self.parameters['input_size']['CC'][0],
                self.parameters['input_size']['CC'][1],
            ),
            dtype=np.float32,
        )
        self.R_CC_view_batch_buffer = np.zeros(
            shape=(
                parameters['minibatch_size'],
                self.loader_input_channels,
                self.parameters['input_size']['CC'][0],
                self.parameters['input_size']['CC'][1],
            ),
            dtype=np.float32,
        )
        self.L_MLO_view_batch_buffer = np.zeros(
            shape=(
                parameters['minibatch_size'],
                self.loader_input_channels,
                self.parameters['input_size']['MLO'][0],
                self.parameters['input_size']['MLO'][1],
            ),
            dtype=np.float32,
        )
        self.R_MLO_view_batch_buffer = np.zeros(
            shape=(
                parameters['minibatch_size'],
                self.loader_input_channels,
                self.parameters['input_size']['MLO'][0],
                self.parameters['input_size']['MLO'][1],
            ),
            dtype=np.float32,
        )

    def start_epoch(self, data_list, phase, labels, permute, augmentation, randomise_view, random_seed):
        # TODO: [zp489] Refactor so we don't copy the whole start_epoch

        self.q_data_to_load = multiprocessing.Queue(maxsize=self.parameters['metadata_buffer_size'])
        self.q_data_loaded = multiprocessing.Queue(maxsize=self.parameters['metadata_buffer_size'])
        self.q_empty_indices = multiprocessing.Queue(maxsize=self.parameters['data_buffer_size'])

        for i in range(0, self.parameters['data_buffer_size']):
            self.q_empty_indices.put(i)

        self.all_threads = []

        if phase == "training" and self.train_sampler.sampling_mode != "normal":
            assert permute 
            oversampled_indices = self.train_sampler.sample_indices(labels, random_seed=random_seed)
            self.logger.log('oversampled_indices %d/%d'%(len(oversampled_indices), len(data_list)))
            feeder_thread = multiprocessing.Process(
                target=loading_cancer_heatmap.loader_feeder_with_given_index,
                args=[data_list, self.q_data_to_load, self.number_of_loaders, self.parameters, oversampled_indices]
            )
            data_length = len(oversampled_indices)
            self.logger.log('Assigned self.data_observation_list_indices to oversampled_indices.') 
            self.data_observation_list_indices = oversampled_indices

        elif phase == 'training_observation':

            loader_feeder_random_seed = random_seed
            feeder_thread = multiprocessing.Process(
                target=loading_multisamples.loader_feeder,
                args=[data_list, self.q_data_to_load, self.number_of_loaders, self.parameters, loader_feeder_random_seed, permute]
            )
            data_length = len(data_list)

        else:
            loader_feeder_random_seed = random_seed
            feeder_thread = multiprocessing.Process(
                target=loading_cancer_heatmap.loader_feeder,
                args=[data_list, self.q_data_to_load, self.number_of_loaders, self.parameters, loader_feeder_random_seed, permute]
            )
            data_length = len(data_list)
            if phase == 'training':
                self.data_observation_list_indices = range(len(data_list))

        feeder_thread.start()
        
        self.all_threads.append(feeder_thread)

        for i in range(0, self.number_of_loaders):

            loader_random_seed = random_seed + i + 1

            if self.parameters['features'] == 'only image':
                if phase == 'training_observation':
                    loader_thread = multiprocessing.Process(
                        target=loading_multisamples.loader_multiple_input_size,
                        args=[self.shared_array_list, self.shared_cancer_heatmap_array_list,
                              self.q_data_to_load, self.q_data_loaded, self.q_empty_indices,
                              self.parameters, loader_random_seed, augmentation, randomise_view],
                    )

                else:
                    loader_thread = multiprocessing.Process(
                        target=loading_cancer_heatmap.loader_multiple_input_size,
                        args=[self.shared_array_list, self.shared_cancer_heatmap_array_list,
                              self.q_data_to_load, self.q_data_loaded, self.q_empty_indices,
                              self.parameters, loader_random_seed, augmentation, randomise_view],
                    )
            elif self.parameters['features'] == 'only non-image':
                loader_thread = multiprocessing.Process(
                    target=loading.non_image_loader,
                    args=[self.q_data_to_load, self.q_data_loaded, self.parameters],
                )
            else:
                raise KeyError(self.parameters['features'])

            loader_thread.start()
            self.all_threads.append(loader_thread)

        return {
            "data_length": data_length,
        }

    def give_minibatch(self, current_index, data_length, size_of_the_minibatch, multiple=False):

        # getting the items from the queue

        minibatch = []
        indices = []
        read_counter = 0
            
        if not multiple:
            while (read_counter < size_of_the_minibatch) and (current_index < data_length):
                start = dt.datetime.now()

                try:
                    data_index, shared_array_index, datum_minus_images = self.q_data_loaded.get(timeout=300)
                    
                except (TimeoutError, queue.Empty, KeyboardInterrupt) as err:
                    end = dt.datetime.now()
                    print("ERROR", type(err))
                    print("ERROR", start, end)
                    print("ERROR", "Queue size", self.q_data_loaded.qsize())
                    print("ERROR", "Queue size", self.q_data_to_load.qsize())
                    raise TimeoutError()

                if datum_minus_images is not None:
                    non_image_features, label = datum_minus_images

                    minibatch.append((non_image_features, shared_array_index, label))
                    indices.append(data_index)

                    self._log(
                        'Loaded data of an index: ' + str(data_index) + ' number: ' + str(read_counter + 1)
                        + ' in this minibatch. In this epoch seen '
                        + str(current_index + 1) + ' / ' + str(data_length) + '.'
                    )
                else:
                    self._log(
                        'Loading data of an index: ' + str(data_index) + ' number: ' + str(read_counter + 1)
                        + ' in this minibatch unsuccessful. In this epoch seen '
                        + str(current_index + 1) + ' / ' + str(data_length) + '.'
                    )

                read_counter = read_counter + 1
                current_index = current_index + 1
        else:
            try:
                data_indexum, shared_array_indexum, datum_minus_imagesum = self.q_data_loaded.get(timeout=300)
            except (TimeoutError, queue.Empty, KeyboardInterrupt) as err:
                    end = dt.datetime.now()
                    print("ERROR", type(err))
                    print("ERROR", start, end)
                    print("ERROR", "Queue size", self.q_data_loaded.qsize())
                    print("ERROR", "Queue size", self.q_data_to_load.qsize())
                    raise TimeoutError()
            #self._log('Data got {}'.format(data_index))
            for ind_sample in range(len(data_indexum)):

                non_image_features, label = datum_minus_imagesum[ind_sample]

                minibatch.append((non_image_features, shared_array_indexum[ind_sample], label))
                indices.append(data_indexum[ind_sample])

                self._log(
                    'Loaded data of an index: ' + str(data_indexum[ind_sample]) + ' number: ' + str(read_counter + 1)
                    + ' in this minibatch. In this epoch seen '
                    + str(current_index + 1) + ' / ' + str(data_length) + '.'
                )

                read_counter = read_counter + 1
                current_index = current_index + 1

        # copying data from the shared memory and releasing the space in the shared memory

        number_of_correctly_loaded_examples = len(minibatch)

        # creating buffers of appropriate sizes by using previously allocated memory
        # Includes both regular view and cancer_heatmap channels
        L_CC_view_batch = self.L_CC_view_batch_buffer[0: number_of_correctly_loaded_examples, :, :, :]
        R_CC_view_batch = self.R_CC_view_batch_buffer[0: number_of_correctly_loaded_examples, :, :, :]
        L_MLO_view_batch = self.L_MLO_view_batch_buffer[0: number_of_correctly_loaded_examples, :, :, :]
        R_MLO_view_batch = self.R_MLO_view_batch_buffer[0: number_of_correctly_loaded_examples, :, :, :]

        label_batch = np.zeros(shape=(number_of_correctly_loaded_examples, config.number_of_classes), dtype=np.int32)

        write_counter = 0

        for datum in minibatch:
            _, shared_array_index, label = datum
            label = self.map_label(label)

            # read from the shared memory
            # B, C, H, W <- [H, W]
            L_CC_view_batch[write_counter, 0] = self.shared_array_list[0][shared_array_index, :, :, 0]
            R_CC_view_batch[write_counter, 0] = self.shared_array_list[1][shared_array_index, :, :, 0]
            L_MLO_view_batch[write_counter, 0] = self.shared_array_list[2][shared_array_index, :, :, 0]
            R_MLO_view_batch[write_counter, 0] = self.shared_array_list[3][shared_array_index, :, :, 0]

            # B, C, H, W <- [C, H, W]
            L_CC_view_batch[write_counter, 1:] = self.shared_cancer_heatmap_array_list[0][shared_array_index]
            R_CC_view_batch[write_counter, 1:] = self.shared_cancer_heatmap_array_list[1][shared_array_index]
            L_MLO_view_batch[write_counter, 1:] = self.shared_cancer_heatmap_array_list[2][shared_array_index]
            R_MLO_view_batch[write_counter, 1:] = self.shared_cancer_heatmap_array_list[3][shared_array_index]

            label_batch[write_counter, label] = 1
            write_counter = write_counter + 1

            self.q_empty_indices.put(shared_array_index)
            self._log('Releasing the index in the shared array: ' + str(shared_array_index), print_log=True)

        if ('return_auxiliary_separately' in self.parameters) and self.parameters['return_auxiliary_separately']:
            # TODO: add safety conditions
            images_batch = (
                np.expand_dims(L_CC_view_batch[:,0], -1),
                np.expand_dims(R_CC_view_batch[:,0], -1),
                np.expand_dims(L_MLO_view_batch[:,0], -1),
                np.expand_dims(R_MLO_view_batch[:,0], -1),
            )
            first_seg_batch = (
                np.expand_dims(L_CC_view_batch[:,1], -1),
                np.expand_dims(R_CC_view_batch[:,1], -1),
                np.expand_dims(L_MLO_view_batch[:,1], -1),
                np.expand_dims(R_MLO_view_batch[:,1], -1),
            )
            second_seg_batch = (
                np.expand_dims(L_CC_view_batch[:,2], -1),
                np.expand_dims(R_CC_view_batch[:,2], -1),
                np.expand_dims(L_MLO_view_batch[:,2], -1),
                np.expand_dims(R_MLO_view_batch[:,2], -1),
            )
            return current_index, indices, ((images_batch, first_seg_batch, second_seg_batch), label_batch)
        else:
            images_batch = (
                np.moveaxis(L_CC_view_batch, 1, 3),
                np.moveaxis(R_CC_view_batch, 1, 3),
                np.moveaxis(L_MLO_view_batch, 1, 3),
                np.moveaxis(R_MLO_view_batch, 1, 3),
            )
        
            return current_index, indices, (images_batch, label_batch)

class DataSampler:
    def __init__(self, sampling_mode, sampling_mode_config, logger):
        self.sampling_mode = sampling_mode
        self.sampling_mode_config = sampling_mode_config
        self.logger = logger

        assert self.sampling_mode in ["normal", "max", "min", "fixed", "fixed_json"], self.sampling_mode

    def get_sampling_template(self, labels):

        if len(labels.shape) == 2:
            num_obs, num_classes = labels.shape[0], labels.shape[-1]
            num_per_class = labels.sum(0)
        elif len(labels.shape) == 3:
            num_obs, num_classes, num_per_class = None, None, None
        else:
            raise RuntimeError()

        if self.sampling_mode == "normal":
            assert len(labels.shape == 2)
            assert self.sampling_mode_config is None
            result = [
                {"n": num_per_class[class_i], "replace": False}
                for class_i in range(num_classes)
            ]
        elif self.sampling_mode in ("max", "min"):
            assert self.sampling_mode_config is None
            if self.sampling_mode == "max":
                match_class = np.argmax(num_per_class)
            else:
                match_class = np.argmin(num_per_class)
            match_class_num = num_per_class[match_class]
            result = [{"n": match_class_num, "replace": True}] * num_per_class
            result[match_class]["replace"] = False
        elif self.sampling_mode == "fixed":
            sampled_per_class = [int(raw_i) for raw_i in self.sampling_mode_config.split("/")]
            assert len(sampled_per_class) == num_classes
            result = []
            for class_i in range(num_classes):
                if sampled_per_class[class_i] == -1:
                    result.append({
                        "n": num_per_class[class_i],
                        "replace": False,
                    })
                else:
                    result.append({
                        "n": sampled_per_class[class_i],
                        "replace": sampled_per_class[class_i] > num_per_class[class_i],
                    })
        elif self.sampling_mode == "fixed_json":
            assert len(labels.shape) == 3  # only supporting multiclass for now
            num_obs, num_label_sets, num_classes = labels.shape
            all_labels = self._all_labels(num_label_sets, num_classes)
            all_labels_str = ["".join(map(str, _)) for _ in all_labels]
            label_values = labels.argmax(-1)
            sampled_per_class_dict = json.loads(self.sampling_mode_config)
            result = []
            for class_i in np.arange(num_label_sets ** num_classes):
                label = all_labels[class_i]
                label_str = all_labels_str[class_i]
                label_count = (label_values == np.array(label)).all(axis=-1).sum()
                if label_str in sampled_per_class_dict:
                    sub_result = {
                        "n": sampled_per_class_dict[label_str],
                        "replace": sampled_per_class_dict[label_str] > label_count,
                    }
                    self.logger.log("Setting class [{}] from {} to {}".format(
                        label_str, label_count, sampled_per_class_dict[label_str],
                    ))
                else:
                    sub_result = {
                        "n": label_count,
                        "replace": False,
                    }
                    self.logger.log("Leaving class [{}] as {}".format(
                        label_str, label_count,
                    ))
                result.append(sub_result)
                
        else:
            raise KeyError(self.sampling_mode)
        self.logger.log(str(result))
        return result

    def sample_indices(self, labels, random_seed):
        """Oversamples smaller classes such that every class has a number of observations equal to the top class

        :param labels: Array (num_obs X num_classes)
        :param random_seed: random seed
        :return: np array of indices
        """
        if len(labels.shape) == 2:
            num_obs, num_classes = labels.shape

            sampling_template = self.get_sampling_template(labels)
            rng = np.random.RandomState(random_seed)

            int_range = np.arange(num_obs)
            resampled_indices_per_class = {}
            for class_i in range(num_classes):
                indices = list(int_range[labels[:, class_i].astype(bool)])
                new_indices = rng.choice(
                    indices,
                    size=sampling_template[class_i]["n"],
                    replace=sampling_template[class_i]["replace"]
                )
                resampled_indices_per_class[class_i] = new_indices
            full_new_indices = np.concatenate(list(resampled_indices_per_class.values()))

            rng.shuffle(full_new_indices)
            return full_new_indices
        elif len(labels.shape) == 3:
            num_obs, num_label_sets, num_classes = labels.shape
            all_labels = self._all_labels(num_label_sets, num_classes)
            label_values = labels.argmax(-1)

            sampling_template = self.get_sampling_template(labels)
            rng = np.random.RandomState(random_seed)

            int_range = np.arange(num_obs)
            resampled_indices_per_class = {}
            for class_i in np.arange(num_label_sets ** num_classes):
                if sampling_template[class_i]["n"] == 0:
                    continue
                indices = int_range[(label_values == all_labels[class_i]).all(axis=-1)]
                new_indices = rng.choice(
                    indices,
                    size=sampling_template[class_i]["n"],
                    replace=sampling_template[class_i]["replace"]
                )
                resampled_indices_per_class[class_i] = new_indices
            full_new_indices = np.concatenate(list(resampled_indices_per_class.values()))

            rng.shuffle(full_new_indices)
            return full_new_indices


    def sample_indices_with_removal(self, labels, indices_to_remove, random_seed):
        """Oversamples smaller classes such that every class has a number of observations equal to the top class

        :param labels: Array (num_obs X num_classes)
        :param random_seed: random seed
        :return: np array of indices
        """

        assert len(labels.shape) == 3
        num_obs, num_label_sets, num_classes = labels.shape
        all_labels = self._all_labels(num_label_sets, num_classes)
        label_values = labels.argmax(-1)

        sampling_template = self.get_sampling_template(labels)
        rng = np.random.RandomState(random_seed)

        int_range = np.arange(num_obs)
        resampled_indices_per_class = {}
        for class_i in np.arange(num_label_sets ** num_classes):
            if sampling_template[class_i]["n"] == 0:
                continue
            indices = int_range[(label_values == all_labels[class_i]).all(axis=-1)]
            if len(set(indices) - set(indices_to_remove))>sampling_template[class_i]["n"]:
                indices = list(set(indices) - set(indices_to_remove))
            new_indices = rng.choice(
                indices,
                size=sampling_template[class_i]["n"],
                replace=sampling_template[class_i]["replace"]
            )
            resampled_indices_per_class[class_i] = new_indices
        full_new_indices = np.concatenate(list(resampled_indices_per_class.values()))

        rng.shuffle(full_new_indices)
        return full_new_indices

    @classmethod
    def _all_labels(cls, num_label_sets, num_classes):
        return list(itertools.product(range(num_classes), repeat=num_label_sets))

    @classmethod
    def normal(cls, logger):
        return cls(
            sampling_mode="normal",
            sampling_mode_config=None,
            logger=logger,
        )

@gin.configurable
class data_single_view(data):
    def __init__(self, logger, 
                       view, 
                       model, 
                       stochasticity,
                       minibatch_size,
                       input_channels,
                       number_of_loaders,
                       data_buffer_size,
                       data_prefix,
                       file_data_list,
                       metadata_buffer_size,
                       cancer_heatmap_channels,
                       
                       training_fraction,
                       validation_fraction,
                       training_epochs_per_validation_epoch,
                       
                       random_seed,
                       cancer_heatmap_prefix,
                       train_sampling_mode="normal",
                       train_sampling_mode_config=None,
                       input_size =config.input_size_dict,
                       resize_shape=None,
                       non_image_features = [],
                       input_format = 'hdf5',
                       image_transformation_library = "cv2",
                       max_crop_noise=(100, 100),
                       max_crop_size_noise=100,
                       max_rotation_noise=0.0,
                       features = 'only image',
                       augmentation_center='best_center',
                       verbose=True, 
                       verbose_printing=True,
                       loaded_data_list=None):

        parameters = dict(stochasticity = stochasticity,
                          minibatch_size = minibatch_size,
                          input_channels = input_channels,
                          number_of_loaders = number_of_loaders,
                          data_buffer_size = data_buffer_size,
                          data_prefix = data_prefix, 
                          file_data_list = file_data_list,
                          cancer_heatmap_channels=cancer_heatmap_channels,
                          metadata_buffer_size = metadata_buffer_size,
                          training_fraction = training_fraction,
                          validation_fraction = validation_fraction,
                          training_epochs_per_validation_epoch = training_epochs_per_validation_epoch, 
                          input_size = input_size,
                          resize_shape = resize_shape,
                          random_seed = get_random_seed(random_seed),
                          train_sampling_mode = train_sampling_mode,
                          train_sampling_mode_config = train_sampling_mode_config,
                          cancer_heatmap_prefix = cancer_heatmap_prefix,
                          non_image_features = non_image_features,
                          input_format = input_format,
                          image_transformation_library = image_transformation_library,
                          max_crop_noise=max_crop_noise,
                          max_crop_size_noise=max_crop_size_noise,
                          max_rotation_noise=max_rotation_noise,
                          features = features,
                          augmentation_center=augmentation_center,
                          verbose=verbose, 
                          verbose_printing=verbose_printing,
                          loaded_data_list=loaded_data_list,

        )
        cv2.setNumThreads(0)
        self.view = view
        self.model = model
        self.seed_shifter = breast_data_utils.SeedShifter.from_parameters(parameters)
        super().__init__(logger, parameters, verbose, loaded_data_list)


@gin.configurable
class data_single_view_with_segmentation(data_with_segmentations):
    def __init__(self, logger, 
                       view, 
                       model, 
                       stochasticity,
                       minibatch_size,
                       input_channels,
                       number_of_loaders,
                       data_buffer_size,
                       data_prefix,
                       file_data_list,
                       metadata_buffer_size,
                       
                       training_fraction,
                       validation_fraction,
                       training_epochs_per_validation_epoch,
                       cancer_heatmap_channels,
                       
                       random_seed,
                       cancer_heatmap_prefix,
                       train_sampling_mode="normal",
                       train_sampling_mode_config=None,
                       input_size =config.input_size_dict,
                       resize_shape=None,
                       non_image_features = [],
                       input_format = 'hdf5',
                       image_transformation_library = "cv2",
                       max_crop_noise=(100, 100),
                       max_crop_size_noise=100,
                       max_rotation_noise=0.0,
                       features = 'only image',
                       augmentation_center='best_center',
                       verbose=True, 
                       verbose_printing=True,
                       loaded_data_list=None):

        parameters = dict(stochasticity = stochasticity,
                          minibatch_size = minibatch_size,
                          input_channels = input_channels,
                          number_of_loaders = number_of_loaders,
                          data_buffer_size = data_buffer_size,
                          data_prefix = data_prefix, 
                          file_data_list = file_data_list,
                          metadata_buffer_size = metadata_buffer_size,
                          training_fraction = training_fraction,
                          validation_fraction = validation_fraction,
                          cancer_heatmap_channels=cancer_heatmap_channels,
                          training_epochs_per_validation_epoch = training_epochs_per_validation_epoch, 
                          input_size = input_size,
                          resize_shape = resize_shape,
                          random_seed = random_seed,
                          train_sampling_mode = train_sampling_mode,
                          train_sampling_mode_config = train_sampling_mode_config,
                          cancer_heatmap_prefix = cancer_heatmap_prefix,
                          non_image_features = non_image_features,
                          input_format = input_format,
                          image_transformation_library = image_transformation_library,
                          max_crop_noise=max_crop_noise,
                          max_crop_size_noise=max_crop_size_noise,
                          max_rotation_noise=max_rotation_noise,
                          features = features,
                          augmentation_center=augmentation_center,
                          verbose=verbose, 
                          verbose_printing=verbose_printing,
                          loaded_data_list=loaded_data_list,

        )
        cv2.setNumThreads(0)
        super().__init__(logger, parameters, verbose, loaded_data_list)
        self.view = view
        self.model = model
        self.seed_shifter = breast_data_utils.SeedShifter.from_parameters(parameters)

@gin.configurable
class data_observation_loader(data_with_segmentations):
    
    def __init__(self, view, model, sampling_mode, logger, parameters, loaded_data_list, verbose=True):
        '''
         loaded_data_list: should be the training data list
        '''
        cv2.setNumThreads(0)
        self.view = view
        self.model = model
        self.sampling_mode = sampling_mode
        self.logger = logger
        self.parameters = parameters
        self.verbose = verbose
        assert 'augmentation_center' in self.parameters, "augmentation_center is not set"

        self.data_list_training = loaded_data_list

        # using only a fraction of the training set
        number_of_all_training_examples = len(self.data_list_training)
        self._train_labels_cancer = None
        self.sampling_fixed = False

        self.current_observation_index = 0

        self.q_data_to_load = None
        self.q_data_loaded = None
        self.q_empty_indices = None

        self.all_threads = []
        self.number_of_loaders = self.parameters['number_of_loaders']

        self._log("Created an object holding a data set.")
        self._log("Using " + str(len(self.data_list_training)) + " training examples.")


        self.loader_input_channels = self.parameters['loader_input_channels'] if 'loader_input_channels' in self.parameters else self.parameters['input_channels']
        # creating the shared array to be able to load a few examples into memory at the same time
        self.shared_array_list = []
        for i in range(2):
            cc_shared_array_base = multiprocessing.Array(ctypes.c_float, self.parameters['data_buffer_size'] * self.parameters['input_size']['CC'][0] * self.parameters['input_size']['CC'][1])
            cc_shared_array_ctype = np.ctypeslib.as_array(cc_shared_array_base.get_obj())
            self.shared_array_list.append(cc_shared_array_ctype.reshape(self.parameters['data_buffer_size'], self.parameters['input_size']['CC'][0], self.parameters['input_size']['CC'][1], 1))
        for i in range(2):
            mlo_shared_array_base = multiprocessing.Array(ctypes.c_float, self.parameters['data_buffer_size'] * self.parameters['input_size']['MLO'][0] * self.parameters['input_size']['MLO'][1])
            mlo_shared_array_ctype = np.ctypeslib.as_array(mlo_shared_array_base.get_obj())
            self.shared_array_list.append(mlo_shared_array_ctype.reshape(self.parameters['data_buffer_size'], self.parameters['input_size']['MLO'][0], self.parameters['input_size']['MLO'][1], 1))
        
        self.shared_cancer_heatmap_array_list = []

        for i in range(2):
            cc_shared_array_base_cancer_heatmap = multiprocessing.Array(
                ctypes.c_float,
                self.parameters['data_buffer_size']
                * self.parameters["cancer_heatmap_channels"]
                * self.parameters['input_size']['CC'][0]
                * self.parameters['input_size']['CC'][1],
            )
            cc_shared_array_ctype_cancer_heatmap = np.ctypeslib.as_array(cc_shared_array_base_cancer_heatmap.get_obj())
            self.shared_cancer_heatmap_array_list.append(cc_shared_array_ctype_cancer_heatmap.reshape(
                self.parameters['data_buffer_size'],
                self.parameters["cancer_heatmap_channels"],
                self.parameters['input_size']['CC'][0],
                self.parameters['input_size']['CC'][1],
            ))

        for i in range(2):
            mlo_shared_array_base_cancer_heatmap = multiprocessing.Array(
                ctypes.c_float,
                self.parameters['data_buffer_size']
                * self.parameters["cancer_heatmap_channels"]
                * self.parameters['input_size']['MLO'][0]
                * self.parameters['input_size']['MLO'][1],
            )
            mlo_shared_array_ctype_cancer_heatmap = np.ctypeslib.as_array(mlo_shared_array_base_cancer_heatmap.get_obj())
            self.shared_cancer_heatmap_array_list.append(mlo_shared_array_ctype_cancer_heatmap.reshape(
                self.parameters['data_buffer_size'],
                self.parameters["cancer_heatmap_channels"],
                self.parameters['input_size']['MLO'][0],
                self.parameters['input_size']['MLO'][1],
            ))

        # Overwrite buffers with more channels
        # Generally, input_channels = 1 + cancer_heatmap_channels
        # Hack-ish
        self.L_CC_view_batch_buffer = np.zeros(
            shape=(
                parameters['minibatch_size'],
                self.loader_input_channels,
                self.parameters['input_size']['CC'][0],
                self.parameters['input_size']['CC'][1],
            ),
            dtype=np.float32,
        )
        self.R_CC_view_batch_buffer = np.zeros(
            shape=(
                parameters['minibatch_size'],
                self.loader_input_channels,
                self.parameters['input_size']['CC'][0],
                self.parameters['input_size']['CC'][1],
            ),
            dtype=np.float32,
        )
        self.L_MLO_view_batch_buffer = np.zeros(
            shape=(
                parameters['minibatch_size'],
                self.loader_input_channels,
                self.parameters['input_size']['MLO'][0],
                self.parameters['input_size']['MLO'][1],
            ),
            dtype=np.float32,
        )
        self.R_MLO_view_batch_buffer = np.zeros(
            shape=(
                parameters['minibatch_size'],
                self.loader_input_channels,
                self.parameters['input_size']['MLO'][0],
                self.parameters['input_size']['MLO'][1],
            ),
            dtype=np.float32,
        )

        if "train_sampling_mode" in self.parameters:
            self.train_sampler = DataSampler(
                sampling_mode=self.parameters["train_sampling_mode"],
                sampling_mode_config=self.parameters["train_sampling_mode_config"],
                logger=logger,
            )
        else:
            self.train_sampler = DataSampler.normal(logger)


    def set_data_observation_list_indices(self, mode, length_to_sample, sample_indices_current_epoch):
        mode_ = 'multiclass_cancer_sides'
        labels_func = {
            Modes.BIRADS: self.get_train_labels,
            Modes.CANCER: lambda: self.get_train_labels_cancer(mode_),
            Modes.CANCER_WITH_UNKNOWNS: lambda: self.get_train_labels_cancer(mode_),
            Modes.MULTICLASS_CANCER: lambda: self.get_train_labels_cancer(mode_),
            Modes.MULTICLASS_CANCER_WITH_UNKNOWNS: lambda: self.get_train_labels_cancer(mode_),
            Modes.MULTICLASS_CANCER_SIDES: lambda: self.get_train_labels_cancer(mode_),
            Modes.MULTICLASS_CANCER_WITH_UNKNOWNS_SIDES: lambda: self.get_train_labels_cancer(mode_),
            Modes.MULTICLASS_CANCER_SIDES_COMBINED: lambda: self.get_train_labels_cancer(mode_),
            Modes.MULTICLASS_CANCER_WITH_UNKNOWNS_SIDES_COMBINED: lambda: self.get_train_labels_cancer(mode_),
        }

        indices_to_obeserve = []
        if mode == 'previous_epoch':
            return sample_indices_current_epoch
        elif mode == 'sample_from_all':

            oversampled_indices = self.train_sampler.sample_indices(labels_func['multiclass_cancer_sides'](), random_seed=10)

            assert length_to_sample == len(oversampled_indices)

            return oversampled_indices

        elif mode == 'sample_from_else':

            oversampled_indices = self.train_sampler.sample_indices_with_removal(labels_func['multiclass_cancer_sides'](), 
                                                                     indices_to_remove=sample_indices_current_epoch,
                                                                     random_seed=10)
            assert length_to_sample == len(oversampled_indices)

            return oversampled_indices

    def start_observation_epoch(self, 
                                sampling_fixed=False,
                                mode = 'previous_epoch',
                                data_observation_list_indices=None, 
                                length_to_sample=None, 
                                image_phase = 'training_observation', random_seed = 0):
        assert mode in ['previous_epoch', 'sample_from_else', 'sample_from_all'] #positive cases the same
        self.current_observation_index = 0
        
        if not sampling_fixed:
            
            data_observation_list_indices = self.set_data_observation_list_indices(mode, length_to_sample, data_observation_list_indices)
        
            self.data_list_observation = [self.data_list_training[i] for i in data_observation_list_indices]

        self.start_epoch(self.data_list_observation, phase=image_phase, labels=None,
                         permute = False, augmentation = False, randomise_view = False, random_seed = random_seed)

    
    def observation_epoch_done(self):
        return self.current_observation_index == len(self.data_list_observation)

    def give_observation_minibatch(self, minibatch_size):

        while not self.observation_epoch_done():
            
            self.current_observation_index, indices, minibatch = self.give_minibatch(
                current_index=self.current_observation_index,
                data_length=len(self.data_list_observation),
                size_of_the_minibatch=minibatch_size,#self.parameters['minibatch_size'],
                multiple=True
            )
            yield indices, minibatch

        raise StopIteration

