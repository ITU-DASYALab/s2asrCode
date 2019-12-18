import numpy as np
import sys
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import record_reader
import util

import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from tensorflow import flags

FLAGS = flags.FLAGS

if __name__ == "__main__":
	flags.DEFINE_string("output_folder", "./plots/", "Output folder location for the plots made")
	flags.DEFINE_string("output_name", "FE", "the output name")
	flags.DEFINE_string("input_tfrecord", "./FE_data/x00000.tfrecord", "The tfrecord to plot")
	flags.DEFINE_list("feature_names", [
		#"power_spectrograms",
		#"magnitude_spectrograms", 
		#"mel_spectrograms", 
		"log_mel_spectrograms", 
		#"all_mfcc", 
		"mfccs"
		], "The feature names to load from the TFrecord")
	flags.DEFINE_list("feature_sizes", 
        [
			#512,
			#512,
			#80,
			80,
			#80,
			13
		],
                   "the feature sizes to use")
	flags.DEFINE_boolean("include_org_signal", True, 
		"Boolean specifying if the original signal should be stored as well inside the tfrecord")
	flags.DEFINE_integer("dpi", 110, "dots per inch for plotting")
	flags.DEFINE_list("image_size",
		[
			1440,
			2160
		],
		"The size of the image")
	flags.DEFINE_boolean("include_first_dim_in_mfcc",True, "set to true to remove the first dimension of the MFCC.")
	flags.DEFINE_integer("seed", 1, "The seed to shuffle the record loaded (there is only one)")

def plot(lis):
	plt.autoscale(enable=True, axis='x', tight=True)
	plt.plot(lis, 'g')

def plot2d(matrix):
	img = plt.imshow(matrix, 
		aspect = 'auto', 
		origin="lower", 
		#interpolation="none",
		cmap='RdYlGn')
	plt.colorbar(img)

def _decode_float_from_binary(binary_sequence):
	decoded_feature_uint8 = tf.decode_raw(binary_sequence,tf.uint8)
	decoded_feature_float32 = tf.cast(decoded_feature_uint8, tf.float32)
	return decoded_feature_float32

def plot_sequence():

	reader = record_reader.Reader(
		feature_names=FLAGS.feature_names,
		feature_sizes= list(map(int, FLAGS.feature_sizes)),
		compression_type= "GZIP", 
		include_org_signal=FLAGS.include_org_signal ,
		org_length=True)
	


	dataset = reader.prepare_reader(FLAGS.input_tfrecord)

	for x in dataset.take(1):

		sequence_parsed = x



	features_to_plot = ["signal"] + FLAGS.feature_names 
	features_to_plot.reverse()

	image_size = [int(x) for x in FLAGS.image_size]

	plt.figure(figsize=(
		image_size[0]/FLAGS.dpi, 
		image_size[1]/FLAGS.dpi), dpi=FLAGS.dpi)
	
	
	for idx, x in enumerate(features_to_plot):
		ax1 = plt.axes([0.1, 0.1 + ((0.95 / len(features_to_plot)) * idx), 0.85, (0.5 / len(features_to_plot))] , frameon=False)
		ax1.set_title(x)
		max_val = np.amax(sequence_parsed[x].numpy())
		min_val = np.amin(sequence_parsed[x].numpy())
		print(x,max_val,min_val)
		if x == "signal":
			line = sequence_parsed["signal"][0].numpy()[150*160:-100*160]
			#ax1.set_yticks([])
			plot(line)
		else:
			if x == "mfccs" and not FLAGS.include_first_dim_in_mfcc:
				spectogram = sequence_parsed["mfccs"].numpy()[150:-100].transpose()[1::]
			else:
				spectogram = sequence_parsed[x].numpy()[150:-100].transpose()
			plt.yticks([0,len(spectogram)-1], [1, len(spectogram)])
			plot2d(spectogram)

	print(sequence_parsed["string"].numpy().decode("utf-8"))

	plt.title = "FE_F"
	plt.savefig(FLAGS.output_folder + FLAGS.output_name + ".pdf")
	plt.savefig(FLAGS.output_folder + FLAGS.output_name + ".png")
	

	

if __name__ == '__main__':
	util.make_dirs([FLAGS.output_folder])
	plot_sequence()