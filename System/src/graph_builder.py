import json
import os
import sys
import time
import numpy as np
# Finally import tensorflow
import tensorflow as tf
import codecs
from tensorflow import flags

# Own files
import models
import record_reader
import util

FLAGS = flags.FLAGS

flags.DEFINE_boolean("word_based", False, "If the output is words, then True, else if the output is characters False")
flags.DEFINE_integer("max_frames", 800, "The maximum number of frames in the records")
flags.DEFINE_integer("min_frames", 300, "The starting number of frames to train on using the curriculum learning strategy")
flags.DEFINE_integer("steps_until_max_frames", 10000, "The step when the curriculum learning use max frames")
flags.DEFINE_boolean("curriculum_learning", True, "start with smaller elements to train on and progressively use longer ones")
flags.DEFINE_bool("l2_batch_normalization", True, "Activate l2 batch normalization")
flags.DEFINE_bool("drop_remainder", False, "drop the unfinished batch of data")
flags.DEFINE_integer("num_gpu", 1, "The number of gpus used by the system")
flags.DEFINE_boolean("print_input_tensors", False, "boolean to print input tensors")
flags.DEFINE_string("input_tfrecord", "FE_data/sv-SE/train*.tfrecord", "The tfrecord(s) to train based on")
flags.DEFINE_boolean("activate_learning_rate_decay", False, "determine whether the learning rate decays during training")
flags.DEFINE_integer("num_parallel_reader", 4, "The amount of files read in parallel")
flags.DEFINE_integer("buffer_size", 4, "The buffer size for the queue")
flags.DEFINE_string("dictionary", "SE", "The dictionary file used as integers to letters")
flags.DEFINE_string("wordcollection", "Libri_word.txt", "The file containing all words in the dataset")
flags.DEFINE_integer("beam_width", 20, "The width of the beam search when finding a sentence")
flags.DEFINE_boolean("include_unknown", True, "a boolean to define if the training should include unknown symbols in the string.")
flags.DEFINE_boolean("custom_beam_search", False, "If we should use the custom wordbased beam search")
flags.DEFINE_integer("seed", 12367512, "a random seed, but it stabilize the results accross executions.")
flags.DEFINE_float("beamsearch_smoothing", 0.0, "The smoothing value for the word beam search")
flags.DEFINE_boolean('automatic_mixed_precision', False, 'whether to allow automatic mixed precision training.')

def combine_gradients(tower_grads):
    filtered_grads = [
            [x for x in grad_list if x[0] is not None] for grad_list in tower_grads
          ]
    final_grads = []
    for i in range(len(filtered_grads[0])):
        grads = [filtered_grads[t][i] for t in range(len(filtered_grads))]
        grad = tf.stack([x[0] for x in grads], 0)
        grad = tf.reduce_sum(grad, 0)
        final_grads.append((
            grad,
            filtered_grads[0][i][1],
        ))

    return final_grads

def clip_gradient_norms(gradients_to_variables, max_norm):

    clipped_grads_and_vars = []
    for grad, var in gradients_to_variables:
        if grad is not None:
            if isinstance(grad, tf.IndexedSlices):
                tmp = tf.clip_by_norm(grad.values, max_norm)
                grad = tf.IndexedSlices(tmp, grad.indices, grad.dense_shape)
            else:
                grad = tf.clip_by_norm(grad, max_norm)
        clipped_grads_and_vars.append((grad, var))

    return clipped_grads_and_vars

def build_graph(model, reader, training=False, optimizer_class=None):
    tf.compat.v1.set_random_seed(FLAGS.seed)
    with tf.device("/cpu:0"):
        global_step = tf.Variable(0, trainable=False, name="global_step")
        tf.compat.v1.add_to_collection("global_step", global_step)

        if training:
            with tf.name_scope("learning_rate"):
                learning_rate = get_learning_rate(global_step)
            
            optimizer = optimizer_class(learning_rate)
            if FLAGS.automatic_mixed_precision:
                tf.compat.v1.logging.info('Enabling automatic mixed precision training.')
                optimizer = tf.compat.v1.train.experimental.enable_mixed_precision_graph_rewrite(optimizer)
        else:
            optimizer = None

        with tf.name_scope("dataset"):
            input_values = get_dataset(reader, drop_remainder=FLAGS.drop_remainder, training=False)
            data_dict = prepare_data(input_values)


    res_dict = build_gpu_towers(training, model, data_dict, optimizer)


    with tf.device("/cpu:0"):
        with tf.name_scope("Loss"):

            label_loss = tf.reduce_mean(tf.stack(res_dict["tower_label_losses"]))
            tf.compat.v1.add_to_collection("loss", label_loss)

            if training:
                if FLAGS.regularization_penalty != 0:
                    reg_loss = tf.reduce_mean(tf.stack(res_dict["tower_reg_losses"]))
                merged_gradients = combine_gradients(res_dict["tower_gradients"])
                if FLAGS.clip_gradient_norm > 0:
                    with tf.name_scope("clip_grads"):
                        merged_gradients = clip_gradient_norms(merged_gradients,
                                                                 FLAGS.clip_gradient_norm)

        with tf.name_scope("predictions"):
            all_predictions = tf.concat(res_dict["tower_predictions"], 0, name="predictions")
            tf.compat.v1.add_to_collection("predictions", all_predictions)

        if training:
            train_op = optimizer.apply_gradients(
                merged_gradients, global_step=global_step)
            tf.compat.v1.add_to_collection("train_op", train_op)


        if training:
            with tf.name_scope("train_summaries"):
                tf.compat.v1.summary.scalar("loss", label_loss)
                tf.compat.v1.summary.scalar("cutoff_frame_count", 
                    tf.compat.v1.get_collection("cutoff_frame_count")[0])

                if FLAGS.regularization_penalty > 0:
                    tf.compat.v1.summary.scalar("reg_loss", reg_loss)
        
        

        summary_op = tf.compat.v1.summary.merge_all()
        tf.compat.v1.add_to_collection("summary_op", summary_op)


# Build graph used during evaluation
def build_evaluation_graph(model,reader):
    build_graph(model,reader,training=False, optimizer_class=None)


# Build graph used during training
def build_training_graph(reader,model,optimizer_class=tf.compat.v1.train.AdamOptimizer):
    build_graph(model,reader,training=True, optimizer_class=optimizer_class)
    
def get_learning_rate(global_step):
    if FLAGS.activate_learning_rate_decay:
        learning_rate = tf.compat.v1.train.exponential_decay(
            FLAGS.base_learning_rate,
            global_step * FLAGS.batch_size,
            FLAGS.learning_rate_decay_examples,
            FLAGS.learning_rate_decay,
            staircase=True)
        tf.compat.v1.summary.scalar("learning_rate", learning_rate)
    else:
        learning_rate = FLAGS.base_learning_rate
    return learning_rate

# Get dataset using the given reader
def get_dataset(reader, drop_remainder, training):

    global_step = tf.compat.v1.get_collection("global_step")[0]

    if FLAGS.curriculum_learning:
        with tf.compat.v1.variable_scope("curriculum_learning"):
            a = (FLAGS.max_frames - FLAGS.min_frames) / FLAGS.steps_until_max_frames
            cutoff_frame_count =  tf.compat.v1.math.minimum(
                tf.cast(a * tf.cast(global_step, tf.float32),tf.int32) + FLAGS.min_frames,  
                FLAGS.max_frames)
    else:
        cutoff_frame_count = tf.constant(FLAGS.max_frames)

    FLAGS.min_frames

    tf.compat.v1.add_to_collection("cutoff_frame_count", cutoff_frame_count)

    dataset = reader.prepare_reader(FLAGS.input_tfrecord, FLAGS.num_parallel_reader)
    if FLAGS.curriculum_learning:
        dataset = dataset.filter(
            lambda x: 
                x["input_num_frames"] < cutoff_frame_count)
    else:
        dataset = dataset.filter(
            lambda x: 
                x["input_num_frames"] < FLAGS.max_frames)
    dataset = dataset.filter(
        lambda x: 
            tf.cast((x["input_num_frames"]- FLAGS.conv_width) / FLAGS.stride, tf.int32) + 1  >= x["label_length"] )

    
    dataset = dataset.repeat(FLAGS.num_epochs)
    if training:
        dataset = dataset.shuffle(FLAGS.buffer_size * FLAGS.batch_size)
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder)	

    dataset = dataset.prefetch(buffer_size=FLAGS.buffer_size)

    input_values = get_input_values(dataset).get_next()

    tf.compat.v1.add_to_collection("num_frames", input_values['input_num_frames'])
    tf.compat.v1.add_to_collection("label", input_values['label'])
    tf.compat.v1.add_to_collection("label_characters", input_values['target_string'])


    if FLAGS.print_input_tensors:
        [tf.compat.v1.logging.info(str(x) +  str(input_values[x])) for x in input_values]

    return input_values

def get_input_values(dataset):
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
    tf.compat.v1.add_to_collection("iterator", iterator)
    return iterator

def prepare_data(input_values):

    model_input_raw = input_values['model_input']
    input_num_frames = input_values['input_num_frames']
    output_num_frames = tf.cast((input_num_frames - FLAGS.conv_width) / FLAGS.stride, tf.int32) + 1 
    tf.compat.v1.add_to_collection("output_num_frames", output_num_frames)
    if FLAGS.word_based:
        target_strings = input_values['label_word']
        sequence_length = input_values['label_word_length']        
    else:
        target_strings = input_values['label']
        sequence_length = input_values['label_length']
    

    num_towers = 1 if FLAGS.num_gpu == 0 else FLAGS.num_gpu
    
    with tf.name_scope("Prepare_data"):
        feature_dim = len(model_input_raw.get_shape()) - 1
        if FLAGS.l2_batch_normalization:
            model_input = tf.nn.l2_normalize(model_input_raw, feature_dim)
        else:
            model_input = model_input_raw

        tower_model_input = tf.split(model_input, num_towers)
        tower_output_num_frames = tf.split(output_num_frames, num_towers)
        tower_target_strings_sparse = tf.sparse.split(sp_input = target_strings, num_split=num_towers, axis=0)
        tower_sequence_length = tf.split(sequence_length, num_towers)

        tf.compat.v1.add_to_collection("tower_model_input", tower_model_input)
        tf.compat.v1.add_to_collection("tower_output_num_frames", tower_output_num_frames)
        tf.compat.v1.add_to_collection("tower_target_strings_sparse", tower_target_strings_sparse)
        tf.compat.v1.add_to_collection("tower_sequence_length", tower_sequence_length)
        
        if FLAGS.word_based:
            dictionary = util.load_dictionary("dict/" + FLAGS.dictionary + "_word.txt")
        else:
            dictionary = util.load_dictionary("dict/" + FLAGS.dictionary + ".txt")
        
        i_t_c = tf.constant(sorted(dictionary, key=dictionary.get, reverse=False))
        letter_table = tf.contrib.lookup.index_to_string_table_from_tensor(i_t_c, default_value="?")
        tf.compat.v1.add_to_collection("letter_table", letter_table)

        output_size = len(dictionary)
        tf.compat.v1.add_to_collection("output_size", output_size)

        tf.compat.v1.logging.debug("Output size: %d" % output_size)
        tf.compat.v1.logging.debug("Dictionary: %s" % str(dictionary))

        data_dict = {
            "tower_model_input": tower_model_input,
            "tower_target_strings_sparse": tower_target_strings_sparse,
            "tower_sequence_length" : tower_sequence_length,
            "tower_output_num_frames": tower_output_num_frames
        }

        return data_dict

def build_gpu_towers(
        regularization_loss,
        model, 
        data_dict,
        optimizer=None):
    tower_gradients = []
    tower_label_losses = []
    tower_reg_losses = []
    tower_final_losses = []
    tower_predictions = []
    tower_sentences = []
    loss_values = []

    if FLAGS.num_gpu > 0:
        tf.compat.v1.logging.info("Using the following number of GPUs to train: " + str(FLAGS.num_gpu))
        num_towers = FLAGS.num_gpu
        device_string = "/gpu:%d"
    else:
        tf.compat.v1.logging.info("No GPUs found. Training on CPU.")
        num_towers = 1
        device_string = "/cpu:%d"


    os.environ["CUDA_VISIBLE_DEVICES"]= ",".join([str(x) for x in list(range(0,FLAGS.num_gpu))])

    for i in range(num_towers):
        with tf.device(device_string % i):
            with tf.compat.v1.variable_scope("tower", reuse=True if i > 0 else None):
            
                
                model_res = model.create_model(
                        data_dict["tower_model_input"][i],
                        output_size=tf.compat.v1.get_collection("output_size")[0],
                        training= True
                    )

                predictions = model_res["predictions"]

                tower_predictions.append(predictions)
                
                #https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss_v2
                loss_value = tf.compat.v1.nn.ctc_loss_v2(
                    labels=data_dict["tower_target_strings_sparse"][i], 
                    logits=predictions, 
                    label_length=data_dict["tower_sequence_length"][i],
                    logit_length=data_dict["tower_output_num_frames"][i],
                    blank_index= -1,
                    logits_time_major=False,
                    name="ctc_loss")


                if regularization_loss:
                    if "regularization_loss" in model_res.keys():
                        reg_loss = model_res["regularization_loss"]
                    else:
                        reg_loss = tf.constant(0.0)

                    reg_losses = tf.compat.v1.losses.get_regularization_losses()
                    if reg_losses:
                            reg_loss += tf.add_n(reg_losses)


                    tower_reg_losses.append(reg_loss)

                # a dependency to the train_op.
                # Adds update_ops (e.g., moving average updates in batch normalization) as
                # https://github.com/google/youtube-8m/blob/master/train.py
                update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
                if update_ops:
                  with tf.control_dependencies(update_ops):
                    barrier = tf.no_op(name="gradient_barrier")
                    with tf.control_dependencies([barrier]):
                      loss_value = tf.identity(loss_value)

                tower_label_losses.append(loss_value)

                predictions_softmax = tf.nn.softmax(predictions)
  
                
                predictions_time_major = tf.transpose(predictions_softmax, [1,0,2])

                tf.compat.v1.add_to_collection("predictions_time_major", predictions_time_major)
                assert(FLAGS.beam_width > 0)

                if FLAGS.custom_beam_search:
                    with tf.device("/cpu:0"):
                        word_beam_search_module = tf.compat.v1.load_op_library('../../CTCWordBeamSearch/cpp/proj/TFWordBeamSearch.so')

                        dictionary_chars = " abcdefghijklmnopqrstuvwxyz'´"
                        corpus = codecs.open("dict/" + FLAGS.wordcollection, 'r', 'utf8').read()
                        wordChars= "abcdefghijklmnopqrstuvwxyz'´"

                        output_num_frames = predictions_softmax.shape[1]
                        character_num = predictions_softmax.shape[2]
                        if FLAGS.num_gpu > 0:
                            tower_batch_size= int(FLAGS.batch_size / FLAGS.num_gpu)
                        else:
                            tower_batch_size= FLAGS.batch_size
                        flattened = tf.reshape(predictions_softmax, [-1])
                        zero_padding = tf.zeros([tower_batch_size * output_num_frames  * character_num] - tf.shape(flattened), dtype=flattened.dtype)
                        padded = tf.concat([flattened, zero_padding], 0)
                        reshaped = tf.reshape(padded, [tower_batch_size, output_num_frames, character_num])
                        reshaped = tf.transpose(reshaped, [1,0,2])
                        #reshaped = tf.nn.softmax(reshaped)
                        tf.compat.v1.add_to_collection("output_padded", reshaped)

                        output_num_frames_un_padded = data_dict["tower_output_num_frames"][i]
                        output_num_frames_un_padded_flattened = tf.reshape(output_num_frames_un_padded, [-1])
                        output_padding = tf.zeros([tower_batch_size] - tf.shape(output_num_frames_un_padded_flattened), dtype=output_num_frames_un_padded.dtype)
                        output_num_frames_padded = tf.concat([output_num_frames_un_padded_flattened, output_padding], 0)
                        output_num_frames = tf.reshape(output_num_frames_padded, [tower_batch_size ])

                        splits = tf.split(reshaped, num_or_size_splits=tower_batch_size, axis=1 )
                        tf.compat.v1.add_to_collection("output_num_frames_padded", output_num_frames)

                        for ix, x in enumerate(splits):
                            x = tf.slice(x, [0, 0, 0],[output_num_frames[ix] , 1, 30])
                            beam = word_beam_search_module.word_beam_search(
                                x, 
                                FLAGS.beam_width, 
                                'Words', 
                                FLAGS.beamsearch_smoothing, 
                                corpus.encode('utf8'), 
                                dictionary_chars.encode('utf8'), 
                                wordChars.encode('utf8'))

                            tf.compat.v1.add_to_collection("beam_output_values", beam)
                            beam_output_letters =  tf.compat.v1.get_collection("letter_table")[0].lookup(tf.cast(beam, dtype=tf.int64))
                            seperator = ""

                            if FLAGS.word_based:
                                seperator = " "

                            sentence = tf.strings.reduce_join(beam_output_letters, -1, separator=seperator, keep_dims=True)
                            tower_sentences.append(sentence)

                else:
                    if FLAGS.beam_width == 1:
                        beam = tf.compat.v1.nn.ctc_greedy_decoder(
                            predictions_time_major,
                            sequence_length=data_dict["tower_output_num_frames"][i],
                            merge_repeated=True
                        )
                    else:
                        beam = tf.compat.v1.nn.ctc_beam_search_decoder_v2(
                            predictions_time_major, 
                            sequence_length=data_dict["tower_output_num_frames"][i],
                            top_paths=1,
                            beam_width=FLAGS.beam_width)

                    beam_output, _ = beam

                    beam_output_values = tf.sparse.to_dense(beam_output[0])
                    tf.compat.v1.add_to_collection("beam_output_values", beam_output_values)
                    #beam_output_letters = tf.contrib.lookup.index_to_string(beam_output_values, mapping=i_t_c, default_value="UNKNOWN")
                    beam_output_letters =  tf.compat.v1.get_collection("letter_table")[0].lookup(beam_output_values)

                    seperator = ""
                    if FLAGS.word_based:
                        seperator = " "

                    sentence = tf.strings.reduce_join(beam_output_letters, -1, separator=seperator, keep_dims=True)
                    tower_sentences.append(sentence)

                if optimizer:
                    tower_final_losses = FLAGS.regularization_penalty * reg_loss + tf.reduce_mean(tf.concat(loss_value,0))
                    grads = optimizer.compute_gradients(tower_final_losses)
                    tower_gradients.append(grads)
    

    tf.compat.v1.add_to_collection("sentence", tf.concat(tower_sentences,0))
                
    res_dict = {
        "tower_gradients": tower_gradients,
        "tower_label_losses": tower_label_losses,
        "tower_reg_losses": tower_reg_losses,
        "tower_predictions": tower_predictions,
        "tower_sentences": tower_sentences
    }
    return res_dict


