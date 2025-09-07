import tensorflow as tf
try:
    tf.config.experimental.set_memory_growth(tf.config.experimental.list_physical_devices('GPU')[0], True)
    print(tf)
except:
    tf = None

import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))