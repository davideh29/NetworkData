'''Run a simple deep CNN on images.
GPU run command:
	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python running_template.py

'''

import h5py
import tifffile as tiff
from keras.backend.common import _UID_PREFIXES

from cnn_functions import nikon_getfiles, get_image, run_models_on_directory, get_image_sizes, segment_nuclei, segment_cytoplasm, dice_jaccard_indices
from model_zoo import sparse_bn_feature_net_61x61 as nuclear_fn


import os
import numpy as np


"""
Load data
"""
direc_name = '/root/NetworkData/ecoli_net/testing_data/video_1'
data_location = os.path.join(direc_name, 'Raw')
nuclear_location = os.path.join(direc_name, 'Nuclear/61x61')
mask_location = os.path.join(direc_name, 'Masks/61x61')

nuclear_channel_names = ['test_vid']

trained_network_nuclear_directory = "/root/NetworkData/ecoli_net/trained_networks/61x61/"

nuclear_prefix = "2017-12-10_ecoli_david_61x61_bn_feature_net_61x61_"

win_nuclear = 30

image_size_x, image_size_y = get_image_sizes(data_location, nuclear_channel_names)
image_size_x /= 2
image_size_y /= 2

"""
Define model
"""

list_of_nuclear_weights = []
for j in xrange(5):
	nuclear_weights = os.path.join(trained_network_nuclear_directory,  nuclear_prefix + str(j) + ".h5")
	list_of_nuclear_weights += [nuclear_weights]
print "\nNuclear Weights:"
print list_of_nuclear_weights
print ""
"""
Run model on directory
"""

nuclear_predictions = run_models_on_directory(data_location, nuclear_channel_names, nuclear_location, model_fn = nuclear_fn, 
	list_of_weights = list_of_nuclear_weights, image_size_x = image_size_x, image_size_y = image_size_y, 
	win_x = win_nuclear, win_y = win_nuclear, std = False, split = False)

"""
Refine segmentation with active contours
"""

nuclear_masks = segment_nuclei(img = None, color_image = True, load_from_direc = nuclear_location, mask_location = mask_location, area_threshold = 100, solidity_threshold = 0, eccentricity_threshold = 1)


"""
Compute validation metrics (optional)
"""
#val = os.path.join(direc_name, 'Validation')
#imglist_val = nikon_getfiles(direc_val, 'validation_interior')

#val_name = os.path.join(direc_val, imglist_val[0]) 
#val = get_image(val_name)
#val = val[win_cyto:-win_cyto,win_cyto:-win_cyto]
#nuc = nuclear_masks[0,win_cyto:-win_cyto,win_cyto:-win_cyto]
#cyto = cytoplasm_masks[0,win_cyto:-win_cyto,win_cyto:-win_cyto]
#dice_jaccard_indices(cyto, val, nuc)
