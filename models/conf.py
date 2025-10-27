import math

training_patience = 10

training_batch_size = 16

training_default_epochs = 2000

training_default_aug_mult = 1

training_default_aug_percent = 0.0

image_width = 800
image_height = 503
image_depth = 3

row = image_height
col = image_width
ch = image_depth

#when we wish to try training for steering and throttle:
num_outputs = 2

#when steering alone:
#num_outputs = 1

throttle_out_scale = 1.0

