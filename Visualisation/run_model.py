# Copyright (C) 2020 Yiqiu Shen, Nan Wu, Jason Phang, Jungkyu Park, Kangning Liu,
# Sudarshini Tyagi, Laura Heacock, S. Gene Kim, Linda Moy, Kyunghyun Cho, Krzysztof J. Geras
#
# This file is part of GMIC.
#
# GMIC is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# GMIC is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with GMIC.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

"""
Script that executes the model pipeline.
"""

import argparse
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
import tqdm
import cv2
import matplotlib.cm as cm
from src.utilities import pickling, tools
from src.models import gmic_globalnet as gmic
from src import modelling
from src.data_loading import loading


def visualize_example(input_img, saliency_maps, 
                      patch_locations, patch_img, patch_attentions,
                      save_dir, parameters):
    """
    Function that visualizes the saliency maps for an example
    """
    # colormap lists
    _, _, h, w = saliency_maps.shape
    _, _, H, W = input_img.shape


    

    # set up colormaps for the 14 chest diseases
    alphas = np.abs(np.linspace(0, 0.95, 259))
    alpha_green = plt.cm.get_cmap('Greens')
    alpha_green._init()
    alpha_green._lut[:, -1] = alphas
    alpha_red = plt.cm.get_cmap('Reds')
    alpha_red._init()
    alpha_red._lut[:, -1] = alphas
    alpha_purple = plt.cm.get_cmap('Purples')
    alpha_purple._init()
    alpha_purple._lut[:, -1] = alphas
    alpha_blue = plt.cm.get_cmap('Blues')
    alpha_blue._init()
    alpha_blue._lut[:, -1] = alphas
    alpha_grey = plt.cm.get_cmap('Greys')
    alpha_grey._init()
    alpha_grey._lut[:, -1] = alphas
    alpha_orange = plt.cm.get_cmap('Oranges')
    alpha_orange._init()
    alpha_orange._lut[:, -1] = alphas
    alpha_pink = plt.cm.get_cmap('pink')
    alpha_pink._init()
    alpha_pink._lut[:, -1] = alphas
    alpha_viridis = plt.cm.get_cmap('viridis')
    alpha_viridis._init()
    alpha_viridis._lut[:, -1] = alphas
    alpha_ocean = plt.cm.get_cmap('ocean')
    alpha_ocean._init()
    alpha_ocean._lut[:, -1] = alphas
    alpha_YlGn = plt.cm.get_cmap('YlGn')
    alpha_YlGn._init()
    alpha_YlGn._lut[:, -1] = alphas
    alpha_terrain = plt.cm.get_cmap('terrain')
    alpha_terrain._init()
    alpha_terrain._lut[:, -1] = alphas
    alpha_RdBu = plt.cm.get_cmap('RdBu')
    alpha_RdBu._init()
    alpha_RdBu._lut[:, -1] = alphas
    alpha_autumn = plt.cm.get_cmap('autumn')
    alpha_autumn._init()
    alpha_autumn._lut[:, -1] = alphas
    alpha_gray = plt.cm.get_cmap('gray')
    alpha_gray._init()
    alpha_gray._lut[:, -1] = alphas




    # create visualization template
    total_num_subplots =  parameters["K"]
    figure = plt.figure(figsize=(60, 10))


   # input image
    subfigure = figure.add_subplot(5, total_num_subplots, 1)
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    subfigure.set_title("input image")
    subfigure.axis('off')


    # patch map
    subfigure = figure.add_subplot(5, total_num_subplots, 2)
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    cm.YlGnBu.set_under('w', alpha=0)
    crop_mask = tools.get_crop_mask(
        patch_locations[0, np.arange(parameters["K"]), :],
        parameters["crop_shape"], (H, W),
        "upper_left")
    subfigure.imshow(crop_mask, alpha=0.7, cmap=cm.YlGnBu, clim=[0.9, 1])
    subfigure.set_title("patch map")
    subfigure.axis('off')
    



    # class activation maps
    subfigure = figure.add_subplot(5, total_num_subplots, 3)
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    resized_cam_atelectasis = cv2.resize(saliency_maps[0,0,:,:], (W, H))
    subfigure.imshow(resized_cam_atelectasis, cmap=alpha_green, clim=[0.0, 1.0])
    subfigure.set_title("SM: atelectasis")
    subfigure.axis('off')

    subfigure = figure.add_subplot(5, total_num_subplots, 4)
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    resized_cam_cardiomegaly = cv2.resize(saliency_maps[0,1,:,:], (W, H))
    subfigure.imshow(resized_cam_cardiomegaly, cmap=alpha_red, clim=[0.0, 1.0])
    subfigure.set_title("SM: cardiomegaly")
    subfigure.axis('off')

    subfigure = figure.add_subplot(5, total_num_subplots, 5)
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    resized_cam_effusion = cv2.resize(saliency_maps[0,2,:,:], (W, H))
    subfigure.imshow(resized_cam_effusion, cmap=alpha_purple, clim=[0.0, 1.0])
    subfigure.set_title("SM: effusion")
    subfigure.axis('off')

    subfigure = figure.add_subplot(5, total_num_subplots, 6)
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    resized_cam_infiltration  = cv2.resize(saliency_maps[0,3,:,:], (W, H))
    subfigure.imshow(resized_cam_infiltration , cmap=alpha_blue, clim=[0.0, 1.0])
    subfigure.set_title("SM: infiltration")
    subfigure.axis('off')

    subfigure = figure.add_subplot(5, total_num_subplots, 7)
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    resized_cam_mass = cv2.resize(saliency_maps[0,4,:,:], (W, H))
    subfigure.imshow(resized_cam_mass, cmap=alpha_grey, clim=[0.0, 1.0])
    subfigure.set_title("SM: mass")
    subfigure.axis('off')

    subfigure = figure.add_subplot(5, total_num_subplots, 8)
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    resized_cam_nodule = cv2.resize(saliency_maps[0,5,:,:], (W, H))
    subfigure.imshow(resized_cam_nodule, cmap=alpha_orange, clim=[0.0, 1.0])
    subfigure.set_title("SM: nodule")
    subfigure.axis('off')

    subfigure = figure.add_subplot(5, total_num_subplots, 9)
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    resized_cam_pneumonia = cv2.resize(saliency_maps[0,6,:,:], (W, H))
    subfigure.imshow(resized_cam_pneumonia, cmap=alpha_pink, clim=[0.0, 1.0])
    subfigure.set_title("SM: pneumonia")
    subfigure.axis('off')

    subfigure = figure.add_subplot(5, total_num_subplots, 10)
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    resized_cam_pneumothorax = cv2.resize(saliency_maps[0,7,:,:], (W, H))
    subfigure.imshow(resized_cam_pneumothorax, cmap=alpha_viridis, clim=[0.0, 1.0])
    subfigure.set_title("SM: pneumothorax")
    subfigure.axis('off')

    subfigure = figure.add_subplot(5, total_num_subplots, 11)
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    resized_cam_consolidation = cv2.resize(saliency_maps[0,8,:,:], (W, H))
    subfigure.imshow(resized_cam_consolidation, cmap=alpha_ocean, clim=[0.0, 1.0])
    subfigure.set_title("SM: consolidation")
    subfigure.axis('off')

    subfigure = figure.add_subplot(5, total_num_subplots, 12)
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    resized_cam_edema = cv2.resize(saliency_maps[0,9,:,:], (W, H))
    subfigure.imshow(resized_cam_edema, cmap=alpha_YlGn, clim=[0.0, 1.0])
    subfigure.set_title("SM: edema")
    subfigure.axis('off')

    subfigure = figure.add_subplot(5, total_num_subplots, 13)
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    resized_cam_emphysema = cv2.resize(saliency_maps[0,10,:,:], (W, H))
    subfigure.imshow(resized_cam_emphysema, cmap=alpha_terrain, clim=[0.0, 1.0])
    subfigure.set_title("SM: emphysema")
    subfigure.axis('off')

    subfigure = figure.add_subplot(5, total_num_subplots, 14)
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    resized_cam_fibrosis = cv2.resize(saliency_maps[0,11,:,:], (W, H))
    subfigure.imshow(resized_cam_fibrosis, cmap=alpha_RdBu, clim=[0.0, 1.0])
    subfigure.set_title("SM: fibrosis")
    subfigure.axis('off')

    subfigure = figure.add_subplot(5, total_num_subplots, 15)
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    resized_cam_pleural_thickening = cv2.resize(saliency_maps[0,12,:,:], (W, H))
    subfigure.imshow(resized_cam_pleural_thickening, cmap=alpha_autumn, clim=[0.0, 1.0])
    subfigure.set_title("SM: pleural_thickening")
    subfigure.axis('off')

    subfigure = figure.add_subplot(5, total_num_subplots, 16)
    subfigure.imshow(input_img[0, 0, :, :], aspect='equal', cmap='gray')
    resized_cam_hernia = cv2.resize(saliency_maps[0,13,:,:], (W, H))
    subfigure.imshow(resized_cam_hernia, cmap=alpha_gray, clim=[0.0, 1.0])
    subfigure.set_title("SM: hernia")
    subfigure.axis('off')
    
    
    
    # crops
    for crop_idx in range(parameters["K"]):
        subfigure = figure.add_subplot(5, total_num_subplots, 17 + crop_idx)
        subfigure.imshow(patch_img[0, crop_idx, :, :], cmap='gray', alpha=0.8, interpolation='nearest',
                         aspect='equal')
        subfigure.axis('off')
        # crops_attn can be None when we only need the left branch + visualization
        subfigure.set_title("$\\alpha_{0} = ${1:.2f}".format(crop_idx, patch_attentions[crop_idx]))
    plt.savefig(save_dir, bbox_inches='tight', format="png", dpi=500)
    plt.close()




def run_model(model, parameters,data_path
):
    """
    Run the model over images in sample_data.
    Save the predictions as csv and visualizations as png.
    """
    exam_list = os.listdir(data_path)
    
    if (parameters["device_type"] == "gpu") and torch.has_cudnn:
        device = torch.device("cuda:{}".format(parameters["gpu_number"]))
    else:
        device = torch.device("cpu")
    model = model.to(device)
    model.eval()

    
    with torch.no_grad():
        # load image
        # the image is already flipped so no need to do it again
       
     for image in tqdm.tqdm(exam_list):
            loaded_image = loading.load_image(
                        image_path=os.path.join(parameters["image_path"], image),
                        
                        horizontal_flip=False,
                    )

            # convert python 2D array into 4D torch tensor in N,C,H,W format
            loaded_image = np.expand_dims(np.expand_dims(loaded_image, 0), 0).copy()
            tensor_batch = torch.Tensor(loaded_image).to(device)
            # forward propagation
            output = model(tensor_batch)
            
            #turn_on_visualization:
            saliency_maps = model.saliency_map.data.cpu().numpy()
            print('>>>>>>>>>>>>',saliency_maps.shape)
            patch_locations = model.patch_locations
            patch_imgs = model.patches
            patch_attentions = model.patch_attns[0, :].data.cpu().numpy()
            
            # create directories
            output_path = "/content/gdrive/My Drive/chestxray/chestnet_results/sample_images_final_0.1_final/"
            os.makedirs(output_path, exist_ok=True)
            os.makedirs(os.path.join(output_path, "visualization"), exist_ok=True)
            short_file_path = image.split('.')[0]
            save_dir = os.path.join(parameters["output_path"], "visualization", "{0}.png".format(short_file_path))
            #print(save_dir)
            visualize_example(loaded_image, saliency_maps, 
                              patch_locations, patch_imgs, patch_attentions,
                              save_dir, parameters)
                
    

def run_single_model(model_path, data_path, parameters):
    """
    Load a single model and run on sample data
    """
    
    # construct model
    #set_seed()
    model = gmic.GMIC(parameters)
    # load parameters
    if parameters["device_type"] == "gpu":
        model.load_state_dict(torch.load(model_path), strict=False)
        
    else:
        model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
    # load metadata
    exam_list = data_path#pickling.unpickle_from_file(data_path)
    
    
    # run the model on the dataset
    run_model(model, parameters, data_path)



    
if __name__ == '__main__':
   


    model_path = "/content/gdrive/My Drive/chestxray/gmic_y_global_1e-4_200_ep_100%_final_changed/model_best_val.pt"

    data_path = "/content/gdrive/My Drive/chestxray/images"
    parameters= {
       "device_type": "cpu",
       "gpu_number": 0,
       "output_path":"/content/gdrive/My Drive/chestxray/chestnet_results/sample_images_final_0.1_final",
       "image_path":"/content/gdrive/My Drive/chestxray/images",
       "cam_size": (16, 16),
       "K": 4,
       "crop_shape": (256, 256),
       "percent_t": 0.1}
      


    run_single_model(model_path, data_path, parameters)

