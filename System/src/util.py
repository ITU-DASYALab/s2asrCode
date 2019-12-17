
from collections import defaultdict
import numpy as np
import tensorflow as tf
import shutil
import sys
import os

from tensorflow import flags

FLAGS = flags.FLAGS

def load_csv(file, split=","):
    with open(file, "r", encoding="utf-8") as f:
        for line in f:
            yield [x for x in line.strip().split(split)]

def make_dirs(names):
    for name in names:
        if not os.path.exists( name ):
            os.makedirs( name )

def clear_dirs(names,filenames=""):

    for name in names:
        if os.path.exists(name):
            for the_file in os.listdir(name):

                if filenames == "" or filenames in the_file:
                    file_path = os.path.join( name, the_file)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path): 
                            shutil.rmtree(file_path)
                        
                    except Exception as e:
                        print(e)

def load_dictionary(path="./dict/SE.txt", with_space=True):
    dictionary = defaultdict(lambda: len(dictionary))
    if with_space:
        dictionary[' '] # Make space the first letter
    if os.path.exists(path):
        with open(path,"r") as f:
            for line in f:
                line = line.strip()
                if line[0] != "#":
                    dictionary[line]
    
    if FLAGS.include_unknown:
        dictionary['_'] # adding the unknown symbol
        
    dictionary['´']  # adding the spacing symbol between repeating letters
    dictionary[''] # adding the blank symbol

    return dictionary

def find_class_by_name(name, modules):
  """Searches the provided modules for the named class and returns it."""
  modules = [getattr(module, name, None) for module in modules]
  return next(a for a in modules if a)

def de_quantize(feat_vector, max_quantized_value=2.0, min_quantized_value=-2.0):
    ''' De-quantizes strings to float32
    From Youtube-8 code, to optimize the data storage.'''
    assert max_quantized_value > min_quantized_value
    quantized_range = max_quantized_value - min_quantized_value
    scalar = quantized_range / 255.0
    bias = (quantized_range / 512.0) + min_quantized_value
    return feat_vector * scalar + bias

def quantize(features, min_quantized_value=-2.0, max_quantized_value=2.0):
    '''Quantizes float32 `features` into string.
    From Youtube-8 code, to optimize the data storage.'''
    assert features.dtype == 'float32'
    assert len(features.shape) == 1  # 1-D array
    features = np.clip(features, min_quantized_value, max_quantized_value)
    quantize_range = max_quantized_value - min_quantized_value
    features = (features - min_quantized_value) * (255.0 / quantize_range)
    features = [int(round(f)) for f in features]
    return bytes(features)

def makeCustomSummary(value, tag):
    summary = tf.compat.v1.Summary()
    val = summary.value.add()
    val.tag = tag
    val.simple_value = value
    return summary

