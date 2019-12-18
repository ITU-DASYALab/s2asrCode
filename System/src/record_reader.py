
import tensorflow as tf
import glob
import util
import random

from tensorflow import flags

FLAGS = flags.FLAGS

flags.DEFINE_boolean("quantization", False, "If the feature extraction should be quantized")

def resize_axis(tensor, axis, new_size, fill_value=0):

	tensor = tf.convert_to_tensor(tensor)
	shape = tf.unstack(tf.shape(tensor))	
	pad_shape = shape[:]
 
	pad_shape[axis] = tf.maximum(0, new_size - shape[axis])	
	shape[axis] = tf.minimum(shape[axis], new_size)
	shape = tf.stack(shape)	
 
	resized = tf.concat([
	 	tf.slice(tensor, tf.zeros_like(shape), shape),
	  	tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))
		], axis)	
	# Update shape.
	new_shape = tensor.get_shape().as_list()  # A copy is being made.
	new_shape[axis] = new_size
	resized.set_shape(new_shape)
	return resized

class BaseReader(object):
	"""Inherit from this class when implementing new readers."""
  
	def prepare_reader(self, unused_filename_queue):
	  """Create a thread for generating prediction and label tensors."""
	  raise NotImplementedError()


def _decode_float_from_binary(binary_sequence):
	decoded_feature_uint8 = tf.decode_raw(binary_sequence,tf.uint8)
	decoded_feature_float32 = tf.cast(decoded_feature_uint8, tf.float32)
	return decoded_feature_float32

class Reader(BaseReader):

	def __init__(self,
		feature_names=[
			"power_spectrograms",
			"magnitude_spectrograms", 
			"mel_spectrograms", 
			"log_mel_spectrograms", 
			"all_mfcc", 
			"mfccs"
			],
		feature_sizes=[512,512,80,80,80,13],
		output_size=30,
		max_frames=300,
		compression_type="GZIP",
		include_org_signal = False,
  		connect_features= False,
		org_length = False):
	 
		assert len(feature_names) == len(feature_sizes), (
			"length of feature_names (={}) != length of feature_sizes (={})".format(
			len(feature_names), len(feature_sizes)))
	 
		self.feature_names = feature_names
		self.compression_type = compression_type
		self.include_org_signal = include_org_signal
		self.max_frames= max_frames
		self.output_size = output_size
		self.feature_sizes= feature_sizes
		self.connect_features= connect_features
		self.org_length = org_length
  
		self.context_feature_description= {
			'name': tf.io.FixedLenFeature([], tf.string, default_value=''),
			'string': tf.io.FixedLenFeature([], tf.string, default_value=''),
			'label':  tf.io.VarLenFeature( dtype=tf.int64),
			'label_length': tf.io.FixedLenFeature([], tf.int64, default_value=0),
			'label_word':  tf.io.VarLenFeature( dtype=tf.int64),
			'label_word_length': tf.io.FixedLenFeature([], tf.int64, default_value=0),
			'target_string': tf.io.FixedLenFeature([], tf.string, default_value=''),
		}

		#for feature_name in [ x + "_shape" for x in feature_names]:
		#	self.context_feature_description[feature_name] = tf.io.FixedLenFeature([3],tf.int64)
		if FLAGS.quantization:
			self.sequence_feature_description = {
				feature_name : tf.io.FixedLenSequenceFeature([], dtype=tf.string)
					for  feature_name in self.feature_names
			}
		else:
			self.sequence_feature_description = {
				feature_name : tf.io.VarLenFeature( dtype=tf.float32)
					for  feature_name in self.feature_names
			}

		if self.include_org_signal:
			self.context_feature_description["signal_shape"] = tf.io.FixedLenFeature([], tf.int64)
			self.sequence_feature_description["signal"]= tf.io.FixedLenSequenceFeature([], dtype=tf.string)
		
	def prepare_reader(self, input_path, num_parallel=8):

		file_addresses = []
		for x in input_path.split("|"):
			file_addresses.extend(glob.glob(x))
		
		random.Random(FLAGS.seed).shuffle(file_addresses)

		raw_dataset = tf.data.TFRecordDataset(file_addresses, 
			compression_type=self.compression_type, 
			num_parallel_reads=num_parallel)
		#tf.compat.v1.add_to_collection("raw_dataset", raw_dataset)
		return raw_dataset.map(self.parse, num_parallel_calls=num_parallel)

		
	def parse(self,rec):
		context_parsed, sequence_parsed_raw = tf.io.parse_single_sequence_example(rec, 
			context_features = self.context_feature_description,
			sequence_features = self.sequence_feature_description)
		
		sequence_parsed = {}
		if self.include_org_signal:
			decoded_signal = _decode_float_from_binary(sequence_parsed_raw["signal"])
			#decoded_signal_reshaped = tf.reshape(decoded_signal, context_parsed["signal_shape"])
			decoded_signal_de_quantized = util.de_quantize(decoded_signal)
			sequence_parsed["signal"] = decoded_signal_de_quantized

		feature_matrices = [None] * len(self.feature_names)
  
		for idx, feature in enumerate(self.feature_names):
			if FLAGS.quantization:
				decoded_feature = _decode_float_from_binary(sequence_parsed_raw[feature])
			else:
				decoded_feature =  tf.cast(sequence_parsed_raw[feature].values, tf.float32)
			

			decoded_reshaped = tf.reshape(decoded_feature, [-1 , self.feature_sizes[idx]])
			num_frames = tf.minimum(tf.shape(decoded_reshaped)[0], self.max_frames)

			if FLAGS.quantization:
				decoded_reshaped = util.de_quantize(decoded_reshaped)

			if self.org_length:
				decoded_resized = decoded_reshaped
			else:
				decoded_resized = resize_axis(decoded_reshaped, 0, self.max_frames)

			if self.connect_features:
				feature_matrices[idx] = decoded_resized
			else:
				sequence_parsed[feature] = decoded_resized
				sequence_parsed[feature + "_num_frames"] = num_frames
			
		if self.connect_features:
			sequence_parsed = {
   				"model_input": tf.concat(feature_matrices, 1),
				"input_num_frames": num_frames
			}

		context = {
			"name": context_parsed["name"],
			"string": context_parsed["string"],
			"label": tf.cast(context_parsed["label"], tf.int32),
			"label_length": tf.cast(context_parsed["label_length"], tf.int32),
			"label_word": tf.cast(context_parsed["label"], tf.int32),
			"label_word_length": tf.cast(context_parsed["label_length"], tf.int32),
			"target_string": context_parsed["target_string"]
		}

		return  {**sequence_parsed , **context}
		
	