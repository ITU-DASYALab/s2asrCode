import numpy as np
import os
import multiprocessing as mp
import util
import os
import re
import tqdm
import time

from functools import partial

import tensorflow as tf

from tensorflow import flags

FLAGS = flags.FLAGS

if __name__ == "__main__":
	flags.DEFINE_integer("sample_rate",
		16000, "The rate at which the sound files are sampled")
	flags.DEFINE_integer("frame_size", 
		512, # 32 ms frame size
		"The frame size for the Short-time Fourier transform")
	flags.DEFINE_integer("step_size", 
		160, # 10 ms step size
		"The step size for the Short-time Fourier transform")
	flags.DEFINE_integer("chunk_size", 64, "A size for the number of files inside each TFRecord")
	flags.DEFINE_integer("num_mel_bins", 80, "Number of MEL bins")
	flags.DEFINE_string("input_dataset", "./data/sv-SE/", "The dataset to do the FE from")
	flags.DEFINE_string("clip_path", "./data/sv-SE/clips/", "The path of the clips")
	flags.DEFINE_string("output_folder", "./FE_data/sv-SE", "Output folder location for the TFRecords")
	flags.DEFINE_string("graph_checkpoint", "./graphs/g_FE", "Output folder for tensorflow graph files for visualization in tensorboard")
	flags.DEFINE_bool("parallel_inference", True, "Boolean for doing parallel inference or not enabling faster inference to the TFRecord files")
	flags.DEFINE_integer("parallelism_degree", 16, "The amount of parallelism to do feature extraction")
	flags.DEFINE_list("data_sets", ["dev","test","train"], "The sets to extract")
	flags.DEFINE_bool("verbose", False, "Boolean to specify the amount of printing")
	flags.DEFINE_list("feature_names", [
		#"power_spectrograms",
		#"magnitude_spectrograms", 
		#"mel_spectrograms", 
		"log_mel_spectrograms", 
		#"all_mfcc", 
		"mfccs"
		], "The feature names to store inside the TFrecord")
	flags.DEFINE_bool("include_org_signal", 
		False,
		"Boolean specifying if the original signal should be stored as well inside the TFRecord")
	flags.DEFINE_boolean("wordbased", False, "If wordbased labels should be included in the feature extraction")
	flags.DEFINE_string("dictionary", "SE", "The dictionary file used as integers to letters")
	flags.DEFINE_boolean("quantization", False, "If the feature extraction should be quantized")
	flags.DEFINE_integer("logging_level", 20, 
		"The logging level specifying how much to log to console"
		"Debug = 10, Info = 20, Warn = 30, Error = 40, Fatal = 50")
	flags.DEFINE_boolean("test", False, "test the FE on one file")
	flags.DEFINE_boolean("amplitude_normalization", False, "apply amplitude normalization to the sound signals before feature extraction")
	flags.DEFINE_boolean("help", False, "Boolean to print all flags")
	flags.DEFINE_boolean("include_unknown", True, "a boolean to define if the training should include unknown symbols in the string.")
	flags.DEFINE_boolean("libri_speech", False, "defines if the datset read is LibriSpeech")
	# ALL FLAGS ABOVE HERE !
	tf.compat.v1.logging.set_verbosity(FLAGS.logging_level)

	if FLAGS.help:
		print(FLAGS)
		exit()
	

def prepareTensorGraph(
		frame_length = 1024, 
		frame_step = 512, 
		sample_rate = 16000, 
		channel_count=2, 
		lower_edge_hertz= 80.0, 
		upper_edge_hertz= 8000.0, 
		num_mel_bins = 64,
		log_offset = 1e-6,
		num_mfccs = 13):
	
	'''
	Setup an inference from audio file to different spectral images of the audio.
	The code is inspired by:
	https://www.tensorflow.org/api_guides/python/contrib.signal
	
	'''
	
	with tf.device('/cpu:0'):

		path = tf.compat.v1.placeholder(tf.string, [], name="path")
		summaries = []
		with tf.name_scope("FE"):
			with tf.name_scope("Parse"):
				audio_binary = tf.io.read_file(path)
				if FLAGS.libri_speech:
					signal = tf.contrib.ffmpeg.decode_audio(audio_binary, 
						file_format="wav", samples_per_second=sample_rate, channel_count=channel_count, stream=None)
				else:
					signal = tf.contrib.ffmpeg.decode_audio(audio_binary, 
						file_format="mp3", samples_per_second=sample_rate, channel_count=channel_count, stream=None)

				signal_averaged = tf.reduce_mean(signal, 1)
				signal_batched = tf.reshape(signal_averaged, [1,-1])

				if FLAGS.test:
					summaries.append(tf.compat.v1.summary.audio(
					    "signal_batched",
					    signal_batched,
					    sample_rate,
					    max_outputs=1))
					summaries.append(tf.compat.v1.summary.histogram("signal_batched", signal_batched))

				if FLAGS.amplitude_normalization:
					signal_batched = signal_amplitude_normalization(signal_batched)
					

				if FLAGS.test:
					summaries.append(tf.compat.v1.summary.audio(
					    "signal_batched_amplified",
					    signal_batched,
					    sample_rate,
					    max_outputs=1))
					summaries.append(tf.compat.v1.summary.histogram("signal_batched_norm", signal_batched))


			with tf.name_scope("Short_time_Fourier_transform"):
				# Calculate Short-Time Fourier Transform (STFT) of frames
				stfts = tf.contrib.signal.stft(
					signal_batched, 
					frame_length=frame_length, 
					frame_step=frame_step, 
					fft_length= frame_length,
					pad_end = True)
			with tf.name_scope("Power_Spectogram"):
				power_spectrograms = tf.math.real(stfts * tf.math.conj(stfts))


			with tf.name_scope("Magnitude_Spectogram"):
				# Calculate magnitude spectrograms which is the magnitude of the complex-valued STFT
				magnitude_spectrograms = tf.abs(stfts)

			with tf.name_scope("Mel"):
				# Define number of spectrogram bins
				num_spectrogram_bins = magnitude_spectrograms.shape[-1].value

				# Calculate log-Mel Spectrograms
				linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
					num_mel_bins,
					num_spectrogram_bins,
					sample_rate,
					lower_edge_hertz,
					upper_edge_hertz)
				mel_spectrograms = tf.tensordot(
					magnitude_spectrograms,
					linear_to_mel_weight_matrix,
					1)
				mel_spectrograms.set_shape(
					magnitude_spectrograms.shape[:-1].concatenate(
						linear_to_mel_weight_matrix.shape[-1:]
						)
					)
			with tf.name_scope("Log_Mel"):
				log_mel_spectrograms = tf.math.log(mel_spectrograms + log_offset)
			with tf.name_scope("MFCC"):
				# Define number of MFCCs by keep the first `num_mfccs` MFCCs.
				all_mfcc = tf.contrib.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)
				mfccs = all_mfcc[..., :num_mfccs]

		tf.compat.v1.add_to_collection("path", path)
		tf.compat.v1.add_to_collection("signal", signal)
		tf.compat.v1.add_to_collection("signal_averaged", signal_averaged)
		tf.compat.v1.add_to_collection("signal_batched", signal_batched)
		tf.compat.v1.add_to_collection("power_spectrograms", power_spectrograms)
		tf.compat.v1.add_to_collection("magnitude_spectrograms", magnitude_spectrograms)
		tf.compat.v1.add_to_collection("linear_to_mel_weight_matrix", linear_to_mel_weight_matrix)
		tf.compat.v1.add_to_collection("mel_spectrograms", mel_spectrograms)
		tf.compat.v1.add_to_collection("log_mel_spectrograms", log_mel_spectrograms)
		tf.compat.v1.add_to_collection("all_mfcc", all_mfcc)
		tf.compat.v1.add_to_collection("mfccs", mfccs)

		output_dict = {
			"path": path,
			"signal": signal,
			"signal_averaged": signal_averaged,
			"signal_batched": signal_batched,
			"stfts": stfts,
			"power_spectrograms": power_spectrograms,
			"magnitude_spectrograms": magnitude_spectrograms,
			"linear_to_mel_weight_matrix": linear_to_mel_weight_matrix,
			"mel_spectrograms": mel_spectrograms, 
			"log_mel_spectrograms": log_mel_spectrograms,
			"all_mfcc": all_mfcc,
			"mfccs": mfccs,
		}
		
		if FLAGS.test:
			summary_op = tf.compat.v1.summary.merge(summaries)
			tf.compat.v1.add_to_collection("summary_op", summary_op)
			output_dict["summary_op"] = summary_op

		return output_dict

def signal_amplitude_normalization(signal_batched):
	signal_max = tf.compat.v1.math.reduce_max(signal_batched)
	signal_min = tf.compat.v1.math.reduce_min(signal_batched)
	abs_signal_min = tf.compat.v1.math.abs(signal_min)
	multiplier = 1 / tf.compat.v1.math.reduce_max([signal_max, abs_signal_min])
	signal_batched_amplified = signal_batched * multiplier
	return signal_batched_amplified

def _int64_list_feature(value):
	"""Returns a int64 list feature containing the value list input."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _int64_feature(value):
	"""Returns a int64 list feature containing one value from a value."""
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	"""Returns a bytes_list from a string / byte."""
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _bytes_feature_list(input_list):
	"""Returns a bytes_list from a list"""
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=input_list))

def _float_feature(value):
	"""Returns a float_list from a float / double."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_feature_list(value):
	"""Returns a float_list from a float / double."""
	return tf.train.Feature(float_list=tf.train.FloatList(value=value))


# Run Tensor graph with the different audio clips
def run_session(name, csvData):
	print("Run session")
	tf.compat.v1.logging.set_verbosity(FLAGS.logging_level)
	tensor_dict = prepareTensorGraph(
		frame_length=FLAGS.frame_size, 
		frame_step=FLAGS.step_size, 
		sample_rate=FLAGS.sample_rate,
		num_mel_bins=FLAGS.num_mel_bins)
	print("Load dictionary")
	if FLAGS.wordbased:
		word_to_index = util.load_dictionary(path =  "./dict/" + FLAGS.dictionary + "_word.txt")
	else:
		character_to_index = util.load_dictionary(path = "./dict/"+FLAGS.dictionary+".txt")

	print("writing output")
	output_record_name = name + '{0:05d}.tfrecord'.format(csvData[0])
	writer = tf.io.TFRecordWriter(
		os.path.join(FLAGS.output_folder, output_record_name),
		options=tf.compat.v1.python_io.TFRecordCompressionType().GZIP
		)
	print("Flags test? ")
	if FLAGS.test:

		summary_writer = tf.compat.v1.summary.FileWriter(
			FLAGS.graph_checkpoint , 
			graph=tf.compat.v1.get_default_graph())

	with tf.compat.v1.Session(
		config=tf.compat.v1.ConfigProto( intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
		) as sess:

		print("Starting clip loop")
		failed_counter = 0
		for clip in csvData[1]:
			if len(clip) < 1 or len(clip[0]) < 1 or len(clip[1]) < 1 or len(clip[2]) < 1:
				failed_counter = failed_counter + 1
			else:
				# Get path of the sound file.
				if FLAGS.libri_speech:
					path = clip[1]
				else:
					path = os.path.join(FLAGS.clip_path , clip[1] )
				
				print("Path: ")
				print(path)

				# Check if the sound file exists.
				if not os.path.exists(path):
					print("path: " + path)
					assert(os.path.exists(path))

				#Get the features from the sound file.
				res = sess.run(tensor_dict, feed_dict={tensor_dict["path"]: path})

				if FLAGS.test:
					summary_writer.add_summary(res["summary_op"])

				# Setup the context for the TF Record.
				context = {
					"name": _bytes_feature(clip[1].encode('utf-8')),
					"string": _bytes_feature(clip[2].encode('utf-8')),
				}

				if FLAGS.wordbased:
					label_word , label_length_word, target_sentence = generate_sparse_target_word(clip[2], word_to_index)
					context =  {**{
					"label_word":  _int64_list_feature(label_word),
					"label_length_word": _int64_feature(label_length_word),
					"target_string": _bytes_feature(target_sentence.encode('utf-8')),
					} , **context}
				else:
					print("Clip: ")
					print(clip[1])
					label , label_length, target_sentence = generate_sparse_target(clip[2], character_to_index)
					context =  {**{
					"label": _int64_list_feature(label),
					"label_length": _int64_feature(label_length),
					"target_string": _bytes_feature(target_sentence.encode('utf-8'))
					} , **context}

				# Setup the features for the tf record.
				feature_list = {}

				if FLAGS.include_org_signal:
					feature_list["signal"] =  tf.train.FeatureList(
							feature=[_bytes_feature(util.quantize(x)) for x in res["signal_batched"]])
					context["signal_shape"] = _int64_list_feature(res["signal_batched"][0].shape)

				for name in FLAGS.feature_names:
					if FLAGS.quantization:
						feature_list[name] = tf.train.FeatureList(
							feature=[_bytes_feature(util.quantize(x)) for x in res[name][0]])
					else:
						feature_list[name]=  tf.train.FeatureList(
							feature=[_float_feature_list(res[name].flatten()) ])
					context[name + "_shape"] = _int64_list_feature(res[name].shape)

				# Combine the context and features.
				example = tf.compat.v1.train.SequenceExample(
					context=tf.train.Features(feature=context),
					feature_lists=tf.train.FeatureLists(feature_list=feature_list))
				
				# Write the TFRecord to disk
				writer.write(example.SerializeToString())
		print("Failed Counter: ", failed_counter)

	
def generate_sparse_target(line, character_to_index):
	values = []
	line = line.lower()
	target_sentence = ""
	assert len(line) > 0

	for char_index, char in enumerate(line):
		if char in character_to_index:
			if char_index > 0 and char == line[char_index -1]:

				values.append(character_to_index['´'])
				target_sentence += '´'
			values.append(character_to_index[char])
			target_sentence += char

		elif FLAGS.include_unknown:
			values.append(character_to_index['_'])
			target_sentence += char
	
	string_lengths = np.array([len(values)])

	if not string_lengths > 0:
		print(line)
		print(target_sentence)
		print(character_to_index)
		string_lengths += 1
		values.append(character_to_index[' '])


	values = np.array(values)

	return values, string_lengths, target_sentence

def generate_sparse_target_word(line, word_to_index):
	values = []
	
	split = list(filter(None,re.split(r'[^\w]', line.lower())))
	target_sentence = " ".join(split)
	for word_index, word in enumerate(split):
		if word_index > 0 and word == split[word_index -1]:
			values.append(word_to_index[''])
		if word in word_to_index:
			values.append(word_to_index[word])
		else:
			print("MISSING!: ",word) 
			values.append(word_to_index['_'])


	length = np.array([len(values)])
	values = np.array(values)

	return values, length, target_sentence

def session_starter(name, split):
	print("session_starter")
	if FLAGS.parallel_inference and not FLAGS.test :
		print("parallel")
		pool = mp.Pool(FLAGS.parallelism_degree)
		func = partial(run_session, name)
		for _ in tqdm.tqdm(pool.imap_unordered(func, split), total=len(split)):
			pass
		pool.close()
		pool.join()
	else:
		print("non-parallel")
		for x in split:
			run_session( name, x)


def generate_TFRecords(from_path, name = "x", test= False):
	'''
	1. Read tsv-file with sentences to retrieve audio files
	2. Parse the audio into Spectral format
	3. Write Spectral images to TfRecords
	'''
	# Read TSV-file containing sentences and related file names
	print("Load TSV")
	csv_data = list(util.load_csv(from_path,split="\t"))[1::]

	print("Parse Audio")
	if test:
		csv_data = csv_data[:1]
	csv_split = [(index + 1,csv_data[x:x + FLAGS.chunk_size]) 
		for index, x in enumerate(range(0, len(csv_data), FLAGS.chunk_size))]

	print("Start session")
	session_starter(name, csv_split)



def generate_TFRecords_LibriSpeech():
	
	test_train_dev = {
		"test": "test-clean",
		"train-100": "train-clean-100",
		"train-360": "train-clean-360",
		"train-500": "train-other-500",
		"dev": "dev-clean",
		"other-test": "test-other",
		"other-dev": "dev-other",
	}
	for key in test_train_dev:
		base_path = os.path.join(FLAGS.input_dataset , test_train_dev[key], "LibriSpeech", test_train_dev[key])
		dir_list = os.listdir(base_path)
		data = []
		for x in dir_list:
			sub_path = os.path.join(base_path, x)
			sub_dir_list = os.listdir(sub_path)
			for y in sub_dir_list:
				sub_sub_path = os.path.join(sub_path, y)
				sub_sub_dir_list = os.listdir(sub_sub_path)
				txt_file = list(filter(lambda x: "trans" in x, sub_sub_dir_list))
				assert len(txt_file) == 1
				with open(os.path.join(sub_sub_path,txt_file[0]), "r", encoding="utf-8") as f:
					for line in f:
						line_split = line.lower().strip().split(" ",1)
						file_path = os.path.join(sub_sub_path,line_split[0]+ ".wav")
						if not  os.path.exists(file_path):
							print("path: " + file_path)
							assert os.path.exists(file_path)
						if not len(line_split[1]) > 0:
							print("path: " + file_path)
							assert len(line_split[1]) > 0
						line_split[0] = file_path
						features= ["", file_path, line_split[1]]
						data.append(features)

		print(len(data))
		if FLAGS.test:
			data = data[:64]

		data_split = [(index + 1,data[x:x + FLAGS.chunk_size]) 
			for index, x in enumerate(range(0, len(data), FLAGS.chunk_size))]


		session_starter(key, data_split)
		print("done", key)




if __name__ == '__main__':
	import os
	os.environ["CUDA_VISIBLE_DEVICES"]="-1"  
	
	print("Starting Feature Extraction")
	util.make_dirs([
		FLAGS.output_folder,
		FLAGS.graph_checkpoint])

	util.clear_dirs([FLAGS.graph_checkpoint], "tfevents")
	
	if FLAGS.libri_speech:
		print("Librispeech")
		generate_TFRecords_LibriSpeech()
	else:
		if FLAGS.test:
			generate_TFRecords(os.path.join(FLAGS.input_dataset , "dev.tsv"), test=True)
		else:
			for x in FLAGS.data_sets:
				input_path = os.path.join(FLAGS.input_dataset , x + ".tsv")
				print(input_path)
				generate_TFRecords(input_path, name=x)
	
	print("Done generating TFRecords")