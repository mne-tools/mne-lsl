'''
Module providing a decoder for BCI and neurofeedback experiments.
 
It assumes that a classifier has already been trained; Look at the protocol trainer_mi.py to learn how to train the classifier.
Currently, LDA, regularized LDA, Random Forests, and Gradient Boosting Machines are supported as classifier types.
'''

from .decoder import BCIDecoder, BCIDecoderDaemon, get_decoder_info,  check_speed, sample_decoding, log_decoding
from .features import compute_features, feature2chz