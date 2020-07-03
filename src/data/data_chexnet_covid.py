# -*- coding: utf-8 -*-
"""
Simple data getters. 

Unknown and before_first_adverse_event should be removed as option.
"""

import numpy as np
import pandas as pd
import os
import gin
import inspect
import time
import pickle
import logging
logging.getLogger(__name__)
import random 
from collections import defaultdict

import imageio
import skimage.transform
import torch
import torchvision
from torchvision import datasets, transforms
from PIL import Image
from src.utilities.pickling import unpickle_from_file

from torchvision.transforms import functional as F_trans
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from scipy.ndimage import rotate, shift
from enum import Enum


class RepeatChannels(object):

    def __init__(self, channels):
        self.channels = channels
        
    def __call__(self, tensor):
        """
        Args:
            tensor: Tensor to be repeated.

        Returns:
            Tensor: repeated tensor.
        """
        return tensor.repeat(self.channels, 1, 1)


    def __repr__(self):
        return self.__class__.__name__ + '()'

def to_uint8(img):
    img = img.astype('float32')
    img -= img.min()
    img /= img.max()
    img *= 255
    return img.astype('uint8')

def load_image(fpath):
    img = to_uint8(imageio.imread(fpath))
    img = Image.fromarray(img)
    return img.convert('RGB')

def map_multilabel_to_multicls(exam):
    
    vector_label = np.zeros(9)
    if exam['label_24']==1:
        label = 1
    elif exam['label_48']==1:
        label = 2
    elif exam['label_72']==1:
        label = 3
    elif exam['label_96']==1:
        label = 4
    elif exam['label_96']==0:
        label = 0
    elif exam['label_72']==0:
        label = 5
    elif exam['label_48']==0:
        label = 6
    elif exam['label_24']==0:
        label = 7
    else:
        label = 8
    vector_label[label] = 1
    return vector_label

class CenterCrop:
    '''
    Adapted from https://github.com/mlmed/torchxrayvision.
    '''
    def __call__(self, img):
        w, h = img.size
        crop_size = min([w, h]) // 2
        left = w // 2 - crop_size
        right = w // 2 + crop_size
        top = h // 2 - crop_size
        bottom = h // 2 + crop_size
        return img.crop((left, top, right, bottom))
    
class ChestXrayDataset(Dataset):
    def __init__(self, data_dir, image_list_file, include_ratio=1., channels=3, transform=None):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        self.channels = channels
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name= items[0]
                label = items[1:]
                label = [int(i) for i in label]
                image_name = os.path.join(data_dir, image_name)
                image_names.append(image_name)
                labels.append(label)

        num_examples = int(len(image_names) * include_ratio)
        self.image_names = image_names[:num_examples]
        self.labels = labels[:num_examples]
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        image_name = self.image_names[index]
        image = load_image(image_name)
        
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label), torch.LongTensor([index])

    def __len__(self):
        return len(self.image_names)

def load_txt_dataset(transform_trainval, transform_test, data_dir, train_image_list_file, val_image_list_file, train_val_image_list_file, test_image_list_file, test, test_augmentation, k, train_data_ratio):
    if (train_image_list_file is not None) or (val_image_list_file is not None):
        assert train_val_image_list_file is None, 'Please either use separate train/val list or combined train_val list'
        train_dataset = ChestXrayDataset(data_dir=data_dir,
                                         image_list_file=train_image_list_file,
                                         include_ratio=train_data_ratio,
                                         transform=transform_trainval)
        val_dataset = ChestXrayDataset(data_dir=data_dir,
                                       image_list_file=val_image_list_file,
                                       transform=transform_trainval)
    else:
    
        # load the combined train_val dataset, one with random augmentation, one without
        trainval_dataset = ChestXrayDataset(
            data_dir=data_dir,
            image_list_file=train_val_image_list_file,
            transform=transform_trainval
        )

        # split indices into train and test
        indices_list = list(range(len(trainval_dataset)))
        random.shuffle(indices_list)


        validation_num = int(len(indices_list)/k)
        assert validation_num >= 1, 'k is too big: %d'%k

        val_indices = indices_list[:validation_num]
        train_indices = indices_list[validation_num:]

        val_labels = np.array(trainval_dataset.labels)[val_indices]
        reshuffle_counter = 0
        while len(np.unique(val_labels)) == 1:
            assert reshuffle_counter < 500, "Have to reshuffle too many times"
            random.shuffle(indices_list)
            val_indices = indices_list[:validation_num]
            train_indices = indices_list[validation_num:]
            val_labels = np.array(trainval_dataset.labels)[val_indices]
            reshuffle_counter += 1

        # get subset from each dataset
        train_dataset = torch.utils.data.Subset(trainval_dataset, train_indices)
        val_dataset = torch.utils.data.Subset(trainval_dataset, val_indices)
    
    if test:
        transform_test = transform_trainval if test_augmentation else transform_test
        test_dataset = ChestXrayDataset(data_dir=data_dir,
                                        image_list_file=test_image_list_file,
                                        transform=transform_test)
    else:
        test_dataset = None
        
    return train_dataset, val_dataset, test_dataset
    
    
def equal_checking(dict_1, dict_2):
    for k in dict_1.keys():
        if (k in dict_2) and (dict_1[k] == dict_2[k] or (pd.isna(dict_1[k]) and pd.isna(dict_2[k]) )):
            pass
        else:
            return False
    return True


def get_exams(data_list_patients, reject_unknowns, before_first_adverse_event_only, include_post_inh_admission_cases):
    exam_list_for_all_patients = []
    index = 0
    for patient in data_list_patients:
        exams_for_patient = defaultdict(dict)
        for image in patient:
            if 'label' in exams_for_patient[image['accession_number']]:
                assert equal_checking(exams_for_patient[image['accession_number']]['label'], image['label'])
            else:
                exams_for_patient[image['accession_number']]['label'] = image['label']
            exams_for_patient[image['accession_number']]['accession_number'] = image['accession_number']
            exams_for_patient[image['accession_number']]['patient_ID'] = image['patient_ID']
            if 'images' not in exams_for_patient[image['accession_number']]:
                exams_for_patient[image['accession_number']]['images'] = []
            exams_for_patient[image['accession_number']]['images'].append(image['short_file_path'])
            
        exam_list_for_patient = list(exams_for_patient.values())
        if reject_unknowns:
            unknown_indices = []
            for i, exam in enumerate(exam_list_for_patient):
                if exam['label']['unknown'] == 1:
                    unknown_indices.append(i)
            for unknown_index in reversed(unknown_indices):
                exam_list_for_patient.pop(unknown_index)

        if before_first_adverse_event_only:
            not_high_confidence_indices = []
            for i, exam in enumerate(exam_list_for_patient):
                if exam['label']['before_first_adverse_event'] == 0:
                    not_high_confidence_indices.append(i)
            for not_high_confidence_index in reversed(not_high_confidence_indices):
                exam_list_for_patient.pop(not_high_confidence_index)
        try:
            if not include_post_inh_admission_cases:
                not_post_inh_admission_cases_indices = []
                for i, exam in enumerate(exam_list_for_patient):
                    if exam['label']['post_inh_admission'] == 1:
                        not_post_inh_admission_cases_indices.append(i)
                for not_post_inh_admission_cases_index in reversed(not_post_inh_admission_cases_indices):
                    exam_list_for_patient.pop(not_post_inh_admission_cases_index)
        except:
            pass

        exam_list_for_all_patients.append(exam_list_for_patient)
        index+=1
        
    return [x for x in exam_list_for_all_patients if len(x) > 0]


def flatten_exams_list(exam_list):
    flattened = []
    indices_dict = defaultdict(list)
    for i, patient in enumerate(exam_list):
        for j, exam in enumerate(patient):
            exam_index_in_flattened_list = len(flattened)
            flattened.append(exam)
            indices_dict[i].append(exam_index_in_flattened_list)
    return flattened, indices_dict


class PreCovidPickleDataset(Dataset):
    def __init__(self, data_dir, file_data_list, uncertainty_label,
                 data_list_index=0, channels=3, randomly_select_image_from_exam=False, transform=None):
        '''
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        '''
        self.data_dir = data_dir
        self.uncertainty_label = uncertainty_label
        self.channels = channels
        self.randomly_select_image_from_exam = randomly_select_image_from_exam
        self.transform = transform
        data_list = unpickle_from_file(file_data_list)[data_list_index]
        exam_list_per_patient = get_exams(data_list, False, False, False)
        self.flattened_exam_list, self.patient_to_exam_indices_dict = flatten_exams_list(exam_list_per_patient)
        self.labels = []
        for exam in self.flattened_exam_list:
            self.labels.append(self.vectorize_label(exam['label']))
        self.labels = np.array(self.labels)

    def __getitem__(self, index):
        exam = self.flattened_exam_list[index]
        if self.randomly_select_image_from_exam:
            image_index = random.randint(0, len(exam['images']) - 1)
        else:
            image_index = 0

        image_name = os.path.join(self.data_dir, exam['images'][image_index] + '.png')
        image = load_image(image_name)

        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label), torch.LongTensor([index])

    def vectorize_label(self, label_dict):
        '''
        return vectorized_label, which is just a python list of integers
        '''
        vectorized_label = []
        sorted_keys = sorted(list(label_dict.keys()))
        for key in sorted_keys:
            label = label_dict[key]
            if label == 'u':
                label = self.uncertainty_label
            vectorized_label.append(label)
        return vectorized_label

    def __len__(self):
        return len(self.labels)


class ChestXrayPickleDataset(Dataset):
    def __init__(self, data_dir, file_data_list, 
        data_list_index=0, channels=3, randomly_select_image_from_exam=False, 
        reject_unknowns=True, before_first_adverse_event_only=True, include_post_inh_admission_cases=False, labels_in_pkl_loader=[], transform=None, multiclass=False):
        """
        Args:
            data_dir: path to image directory.
            image_list_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        self.channels = channels
        data_list = list(unpickle_from_file(file_data_list)[data_list_index])
        exam_list_per_patient = get_exams(data_list, reject_unknowns, before_first_adverse_event_only, include_post_inh_admission_cases)
        self.flattened_exam_list, self.patient_to_exam_indices_dict = flatten_exams_list(exam_list_per_patient)
        self.data_dir = data_dir
        self.randomly_select_image_from_exam = randomly_select_image_from_exam
        
        if multiclass:
            assert labels_in_pkl_loader == ['label_24', 'label_48', 'label_72','label_96']

        self.labels_in_pkl_loader = labels_in_pkl_loader
        self.labels = []

        for exam in self.flattened_exam_list:
            if multiclass:
                vectorized_label = self.map_to_multicls(exam['label'])
            else:
                vectorized_label = self.vectorize_label(exam['label'])
            self.labels.append(vectorized_label)
            
        self.transform = transform

    def map_to_multicls(self, label_dict):
        return map_multilabel_to_multicls(label_dict)

    def vectorize_label(self, label_dict):
        """
        return vectorized_label, which is just a python list of integers
        """
        vectorized_label = []
        for label in self.labels_in_pkl_loader:
            vectorized_label.append(label_dict[label])
        
        return vectorized_label
    
    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        exam = self.flattened_exam_list[index]
        if self.randomly_select_image_from_exam:
            image_index = random.randint(0, len(exam['images']) - 1)
        else:
            image_index = 0
        
        image_name = os.path.join(self.data_dir, exam['images'][image_index]+'.png') 
        image = load_image(image_name)
        
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label), torch.LongTensor([index])

    def __len__(self):
        return len(self.labels)

def get_exam_indices(patient_indices, patient_to_exam_indices_dict):
    exam_indices = []
    for i in patient_indices:
        exam_indices.extend(patient_to_exam_indices_dict[i])
    return exam_indices
    
def two_unique_values_for_all_labels(labels, multiclass):
    """
    ignore np.nan values
    """
    data_size, label_size = labels.shape
    if not multiclass:
        for i in range(label_size):
            v = np.unique(labels[:,i])
            if len(v[~np.isnan(v)]) == 1:
                return False
        return True
    else:
        print(labels.sum(0))
        if sum(labels.sum(0)>0)==label_size:
            return True
        else:
            return False

def load_precovid_pkl_dataset(uncertainty_label,
                              transform_train,
                              transform_inference,
                              data_dir,
                              combined_pkl_file):
    train_dataset = PreCovidPickleDataset(
        data_dir=data_dir,
        file_data_list=combined_pkl_file,
        uncertainty_label=uncertainty_label,
        data_list_index=0,
        randomly_select_image_from_exam=True,
        transform=transform_train
    )
    val_dataset = PreCovidPickleDataset(
        data_dir=data_dir,
        file_data_list=combined_pkl_file,
        uncertainty_label=uncertainty_label,
        data_list_index=1,
        randomly_select_image_from_exam=False,
        transform=transform_inference
    )
    return train_dataset, val_dataset, None

def load_covid_pkl_dataset(transform_trainval,
                           transform_test,
                           data_dir,
                           combined_pkl_file,
                           test,
                           test_augmentation,
                           k,
                           reject_unknowns,
                           before_first_adverse_event_only,
                           include_post_inh_admission_cases,
                           labels_in_pkl_loader,
                           multiclass,
                           partial_training=None
                           ):

    trainval_dataset = ChestXrayPickleDataset(
        data_dir=data_dir,
        file_data_list=combined_pkl_file,
        data_list_index=0,
        randomly_select_image_from_exam=True, 
        reject_unknowns=reject_unknowns, 
        before_first_adverse_event_only=before_first_adverse_event_only,
        include_post_inh_admission_cases=include_post_inh_admission_cases,
        labels_in_pkl_loader=labels_in_pkl_loader,
        transform=transform_trainval,
        multiclass=multiclass,
    )

    # split indices into train and test
    patient_indices_list = list(range(len(trainval_dataset.patient_to_exam_indices_dict)))
    random.shuffle(patient_indices_list)

    validation_num = int(len(patient_indices_list)/k)
    assert validation_num >= 1, 'k is too big: %d'%k

    val_patient_indices = patient_indices_list[:validation_num]
    train_patient_indices = patient_indices_list[validation_num:]
    
    val_exam_indices = get_exam_indices(val_patient_indices, trainval_dataset.patient_to_exam_indices_dict)
    train_exam_indices = get_exam_indices(train_patient_indices, trainval_dataset.patient_to_exam_indices_dict)

    
    # reshuffle until AUC available for all labels
    val_labels = np.array(trainval_dataset.labels)[val_exam_indices]
    reshuffle_counter = 0
    while not two_unique_values_for_all_labels(val_labels, multiclass):
        assert reshuffle_counter < 500, 'Have to reshuffle too many times.'
        random.shuffle(patient_indices_list)
        val_patient_indices = patient_indices_list[:validation_num]
        train_patient_indices = patient_indices_list[validation_num:]
        val_exam_indices = get_exam_indices(val_patient_indices, trainval_dataset.patient_to_exam_indices_dict)
        train_exam_indices = get_exam_indices(train_patient_indices, trainval_dataset.patient_to_exam_indices_dict)
        val_labels = np.array(trainval_dataset.labels)[val_exam_indices]
        reshuffle_counter += 1

    # get subset from each dataset
    if partial_training is not None:
        train_exam_indices = train_exam_indices[:int(len(train_exam_indices)*partial_training)]
    train_dataset = torch.utils.data.Subset(trainval_dataset, train_exam_indices)
    val_dataset = torch.utils.data.Subset(trainval_dataset, val_exam_indices)
    
    if test:
        transform_test = transform_trainval if test_augmentation else transform_test
        test_dataset = ChestXrayPickleDataset(data_dir=data_dir,
                                              file_data_list=combined_pkl_file,
                                              data_list_index=1,
                                              randomly_select_image_from_exam=test_augmentation,
                                              reject_unknowns=reject_unknowns,
                                              before_first_adverse_event_only=before_first_adverse_event_only,
                                              include_post_inh_admission_cases=False,
                                              labels_in_pkl_loader=labels_in_pkl_loader,
                                              transform=transform_test,
                                              multiclass=multiclass, )
    else:
        test_dataset = None
        
    return train_dataset, val_dataset, test_dataset

@gin.configurable
def get_chexnet_covid(
        data_dir,
        precovid=False,
        uncertainty_label=1,
        combined_pkl_file=None,
        train_val_image_list_file=None,
        train_image_list_file=None,
        val_image_list_file=None, 
        test_image_list_file=None,
        batch_size=128,
        num_workers=10,
        seed=777,
        k=10,
        train_data_ratio=1.0,
        train=True,
        valid=True,
        test=True,
        train_precrop_size=1024,
        test_precrop_size=256,
        image_size=224,
        test_augmentation=False,
        reject_unknowns=True, 
        before_first_adverse_event_only=True,
        labels_in_pkl_loader=[], 
        channels=3,
        multiclass=False,
        partial_training=None,
        include_post_inh_admission_cases=False,
):
    '''
    random_seed_for_validation is now a deprecated argument
    '''
    # Set seed (this is the beginning of the program)
    torch.manual_seed(seed) # This should work for all devices including GPU
    np.random.seed(seed)
    random.seed(seed)

    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    transform_trainval = transforms.Compose([
        CenterCrop(),
        transforms.Resize(train_precrop_size),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.Resize(test_precrop_size),
        transforms.TenCrop(image_size),
        transforms.Lambda
        (lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
        transforms.Lambda
        (lambda crops: torch.stack([normalize(crop) for crop in crops]))
    ])
    if precovid:
        train_dataset, val_dataset, test_dataset = load_precovid_pkl_dataset(
            uncertainty_label=uncertainty_label,
            transform_train=transform_trainval,
            transform_inference=transform_test,
            data_dir=data_dir,
            combined_pkl_file=combined_pkl_file
        )
    else:
        if combined_pkl_file is None:
            train_dataset, val_dataset, test_dataset = load_txt_dataset(
                transform_trainval=transform_trainval,
                transform_test=transform_test,
                data_dir=data_dir,
                train_image_list_file=train_image_list_file,
                val_image_list_file=val_image_list_file,
                train_val_image_list_file=train_val_image_list_file,
                test_image_list_file=test_image_list_file,
                test=test,
                test_augmentation=test_augmentation,
                k=k,
                train_data_ratio=train_data_ratio
            )
        else:
            assert train_val_image_list_file==None, 'Trying to use pkl dataset but train_val txt dataset is provided'
            assert train_image_list_file==None, 'Trying to use pkl dataset but train txt dataset is provided'
            assert val_image_list_file==None, 'Trying to use pkl dataset but val txt dataset is provided'
            assert test_image_list_file==None, 'Trying to use pkl dataset but test txt dataset is provided'

            train_dataset, val_dataset, test_dataset = load_covid_pkl_dataset(
                transform_trainval=transform_trainval,
                transform_test=transform_test,
                data_dir=data_dir,
                combined_pkl_file=combined_pkl_file,
                test=test,
                test_augmentation=test_augmentation,
                k=k,
                reject_unknowns=reject_unknowns,
                before_first_adverse_event_only=before_first_adverse_event_only,
                include_post_inh_admission_cases=include_post_inh_admission_cases,
                labels_in_pkl_loader=labels_in_pkl_loader,
                multiclass=multiclass,
                partial_training=partial_training
            )
    
    
    training_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers, pin_memory=True)
    valid_loader = DataLoader(dataset=val_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)
    if test:
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers, pin_memory=True)

    return_tuple = (
        training_loader if train else None, 
        valid_loader if valid else None, 
        test_loader if test else None, 
        {}
    )
    
    return return_tuple
