from cnn import *
from preprocess import *

# Getting the MFCC
sample = wav2mfcc('./data/happy/012c8314_nohash_0.wav')
# We need to reshape it remember?
sample_reshaped = sample.reshape(1, 20, 11, 1)
# Perform forward pass
print(get_labels()[0][
    np.argmax(model.predict(sample_reshaped))
])