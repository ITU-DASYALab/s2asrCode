

import json
import os
import sys
import time
import numpy as np
# Finally import tensorflow
import tensorflow as tf
from tensorflow import flags

# Own files
import models
import record_reader
import util
import graph_builder

FLAGS = flags.FLAGS

if __name__ == "__main__":
    flags.DEFINE_string("training_directory", "models/default/", 
        "Output folder location for the plots made")
    flags.DEFINE_list("feature_names", 
        [
            "mfccs", 
            "log_mel_spectrograms"
        ], "The feature names to load from the TFrecord")
    flags.DEFINE_list("feature_sizes", 
        [
            "13",
            "80"
        ],
                   "the feature sizes to use")
    flags.DEFINE_boolean("include_org_signal", 
        False, 
        "Boolean specifying if the original signal should be stored as well inside the tfrecord")
    flags.DEFINE_string("model", "StreamSpeechM33",
                     "the name of the model in use")
    flags.DEFINE_integer("batch_size", 32, "The batch size")
    flags.DEFINE_boolean("linear_scaling_rule", False,
        "Scaling rule specifying that leaning rate should be multiplied with batch size")
    flags.DEFINE_float("base_learning_rate", 1.00, "Learning rate at the start of training")
    flags.DEFINE_string("optimizer","AdadeltaOptimizer", 
        "The optimization algorithm in use"
        "AdamOptimizer, AdadeltaOptimizer, etc")
    flags.DEFINE_float("learning_rate_decay_examples", 1000000, "The number of elements processed before the learning rate decrease")
    flags.DEFINE_float("learning_rate_decay", 0.95, "The amount the learningrate is decreased by")
    flags.DEFINE_float("clip_gradient_norm", 0.0, "The norm to clip gradients to")
    flags.DEFINE_float("regularization_penalty", 1.0, "how much weight the regularization loss have ")
    flags.DEFINE_integer("num_epochs", None, "The number of epocs of the dataset")
    flags.DEFINE_boolean("log_device_placement", False, "determine if you want to store device placement")
    flags.DEFINE_integer("max_steps", None, "the maximum number of steps")
    flags.DEFINE_integer("eval_steps", 50, "The number of steps before evaluations")
    flags.DEFINE_float("dropout", 0.5, "The dropout value between layers")
    flags.DEFINE_integer("logging_level", 20, 
        "The logging level specifying how much to log to console"
        "Debug = 10, Info = 20, Warn = 30, Error = 40, Fatal = 50")
    flags.DEFINE_float("max_gpu_memory_use", 0.99, "The maximum percentage memory allowed to be used by this process")
    flags.DEFINE_boolean("new_model", True,
        "Defines if the old model should be overwritten")
    flags.DEFINE_boolean("help", False, "Boolean to print all flags")
    flags.DEFINE_integer("save_model_interval", 250, "The number of iterations before the model is saved")
    flags.DEFINE_float("keep_checkpoint_every_n_hours", 1.0," The number of hours between each checkpoint presisting")
    flags.DEFINE_integer("elements_to_print_in_evalutaions", 3, "The number of elements to print while having a evaluation ")

    # ALL FLAGS ABOVE HERE !
    tf.compat.v1.logging.set_verbosity(FLAGS.logging_level)
    if FLAGS.help:
        print(FLAGS)
        exit()

    tf.compat.v1.enable_resource_variables()
    


def build_model(model, reader):
    
    # https://arxiv.org/pdf/1706.02677.pdf
    if FLAGS.linear_scaling_rule:
        learning_rate = FLAGS.base_learning_rate * FLAGS.batch_size
    else:
        learning_rate = FLAGS.base_learning_rate
 
    optimizer_class = util.find_class_by_name(FLAGS.optimizer, [tf.compat.v1.train])

    graph_builder.build_training_graph(
        reader, model,
        optimizer_class=optimizer_class)

    return tf.compat.v1.train.Saver(
            var_list=tf.compat.v1.global_variables(),
            max_to_keep=2, 
            keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours)

def get_meta_filename( start_new_model, train_dir):
    if start_new_model:
        tf.compat.v1.logging.info("Flag 'new_model' is set. Building a new model.")
        return None, None

    latest_checkpoint = tf.train.latest_checkpoint(train_dir)
    if not latest_checkpoint:
          tf.compat.v1.logging.info("No checkpoint file found. Building a new model.")
          return latest_checkpoint, None

    meta_filename = latest_checkpoint + ".meta"
    if not  os.path.exists(meta_filename):
        tf.compat.v1.logging.info("No meta graph file found. Building a new model.")
        return latest_checkpoint, None
    else:
        return latest_checkpoint, meta_filename

def check_flags():

    model_flags_dict = {
        "model": FLAGS.model,
        "feature_sizes": list(map(int, FLAGS.feature_sizes)),
        "feature_names": FLAGS.feature_names,
        "conv_width": FLAGS.conv_width,
        "conv_output": FLAGS.conv_output,
        "activation": FLAGS.activation,
        "add_batch_norm": FLAGS.add_batch_norm,
        "dropout": FLAGS.dropout,
        "LSTM_size": FLAGS.LSTM_size,
        "LSTM_Layer_count": FLAGS.LSTM_Layer_count,
        "stride": FLAGS.stride,
        "include_org_signal": FLAGS.include_org_signal
    }
    
    flags_json_path = os.path.join(FLAGS.training_directory, "model_flags.json")
    print(flags_json_path)
    if os.path.exists(flags_json_path):
        with open(flags_json_path, "r") as read_file:
            existing_flags = json.load(read_file)

        if existing_flags != model_flags_dict:
            tf.compat.v1.logging.error(
                "Model flags do not match existing file %s. Please "
                "delete the file, change --training_directory, or pass flag "
                "--start_new_model", flags_json_path)
            tf.compat.v1.logging.error("Ran model with flags: %s", str(model_flags_dict))
            tf.compat.v1.logging.error("Previously ran with flags: %s", str(existing_flags))
            exit(1)
    else:
        # Write the file.
        with open(flags_json_path, "w") as write_file:
            json.dump(model_flags_dict, write_file)

def print_output_sentences(sentence_val, label_val):
    count = 0
    for y,z in zip(sentence_val, label_val):
        count += 1
        #tf.compat.v1.logging.info( x)
        
        tf.compat.v1.logging.info(
            "\n   out: '" +  y[0].decode("utf-8") + "'" +
            "\ntarget: '" +  z.decode("utf-8") + "'")
        if count >= FLAGS.elements_to_print_in_evalutaions:
            break


def run( ):
    
    if FLAGS.new_model:
        util.clear_dirs([FLAGS.training_directory])

    check_flags()

    model = util.find_class_by_name(FLAGS.model,[models])()
    reader = record_reader.Reader(
        feature_names=FLAGS.feature_names,
        feature_sizes=list(map(int, FLAGS.feature_sizes)),
        max_frames=FLAGS.max_frames,
        include_org_signal=FLAGS.include_org_signal,
        connect_features=True)

    latest_checkpoint, meta_filename = get_meta_filename(
        FLAGS.new_model, FLAGS.training_directory)
    
    #with tf.Graph().as_default() as graph:

    saver = build_model(model, reader)


    global_step = 	tf.compat.v1.get_collection("global_step")[0]
    loss = 			tf.compat.v1.get_collection("loss")[0]
    predictions = 	tf.compat.v1.get_collection("predictions")[0]
    sentence = 		tf.compat.v1.get_collection("sentence")[0]
    beam_output_values = tf.compat.v1.get_collection("beam_output_values")[0]
    train_op = 		tf.compat.v1.get_collection("train_op")[0]
    init_op = 		tf.compat.v1.global_variables_initializer()
    summary_op = 	tf.compat.v1.get_collection("summary_op")[0]
    label =         tf.compat.v1.get_collection("label_characters")[0]
    cutoff_frame_count = tf.compat.v1.get_collection("cutoff_frame_count")[0]
    iterator =      tf.compat.v1.get_collection("iterator")[0]

    tf.compat.v1.get_default_graph().clear_collection('iterator')
    tf.compat.v1.get_default_graph().clear_collection('label')
    tf.compat.v1.get_default_graph().clear_collection('beam_output_values')
    tf.compat.v1.get_default_graph().clear_collection('tower_sequence_length')
    tf.compat.v1.get_default_graph().clear_collection('tower_sentences')
    tf.compat.v1.get_default_graph().clear_collection('tower_target_strings_sparse')
    tf.compat.v1.get_default_graph().clear_collection('tower_output_num_frames')
    tf.compat.v1.get_default_graph().clear_collection('tower_model_input')
    tf.compat.v1.get_default_graph().clear_collection('summary_op')

    gpu_options = tf.compat.v1.GPUOptions(
        allow_growth= True,
        per_process_gpu_memory_fraction=FLAGS.max_gpu_memory_use
        )
    config = tf.compat.v1.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement,
        gpu_options= gpu_options
        )
    
    max_steps_reached = False

    with tf.compat.v1.Session(
        config=config, 
        graph = tf.compat.v1.get_default_graph()) as sess:

        # Initialize tabels for the sentence decoding
        tf.compat.v1.tables_initializer().run() 
        # Initialize Model
        sess.run(init_op)
        # Initialize Data pipeline
        sess.run(iterator.initializer)

        if meta_filename:
            saver.restore(sess, latest_checkpoint)
            saver = tf.compat.v1.train.Saver(
                var_list=tf.compat.v1.global_variables(),
                max_to_keep=2, 
                keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours)

        tf.compat.v1.logging.info("Starting session.")

        try:
            log_folder = FLAGS.training_directory
            summary_writer = tf.compat.v1.summary.FileWriter(log_folder, sess.graph)
            tf.compat.v1.logging.info("Entering training loop.")

            while not max_steps_reached:

                batch_start_time = time.time()

                global_step_val = sess.run(global_step)
                if FLAGS.max_steps and FLAGS.max_steps <= global_step_val:
                    max_steps_reached = True

                if global_step_val % FLAGS.eval_steps != 0:
                    _, loss_val = sess.run([train_op, loss])

                    #print("Hello from step %d" % global_step_val)
                    tf.compat.v1.logging.debug("Debug It: %d, loss %.2f" %(global_step_val,loss_val))
                else:
                    _, loss_val, predictions_val  = sess.run(
                        [train_op, loss, predictions])

                    seconds_per_batch = time.time() - batch_start_time
                    examples_per_second = predictions_val.shape[0] / seconds_per_batch


                    # Put eval calculations here:
                    eval_start_time = time.time()

                    # If curriculum learning then reinitialize the iterator
                    if FLAGS.curriculum_learning:
                        cutoff_frame_count_val = sess.run(cutoff_frame_count)
                        if cutoff_frame_count_val < FLAGS.max_frames:
                            sess.run(iterator.initializer)
                        

                    summary_str = sess.run(summary_op)
                    summary_writer.add_summary(summary_str, global_step_val)
                    summary_writer.add_summary( 
                        util.makeCustomSummary(
                            float(examples_per_second), 
                            "train_summaries/examples_per_second" ), 
                        global_step_val)

                    # Print output sentences.
                    sentence_val, label_val = sess.run([sentence,label])
                    print_output_sentences(sentence_val, label_val)
                        
                    
                    eval_time = time.time() - eval_start_time

                    tf.compat.v1.logging.info("Step " + str(global_step_val) + 
                          " | Loss: % 7.2f" % loss_val + 
                          " | Ex/S: % 8.2f" % examples_per_second + 
                          " | Evt: % 3.2f" % eval_time )
                          
                if global_step_val % FLAGS.save_model_interval == 0:
                    saver.save(sess, 
                        os.path.join( FLAGS.training_directory , "model"), 
                        global_step=global_step_val)
                    tf.compat.v1.logging.info("Exporting the model at step %s to %s.",
                        global_step_val, FLAGS.training_directory )
            pass
        except tf.errors.OutOfRangeError:
            if FLAGS.num_epochs == None:
                tf.compat.v1.logging.error("Error, there are no data matching the criteria selected")
                

            else:
                tf.compat.v1.logging.info("Finished Epocs.")
                saver.save(sess, 
                    os.path.join( FLAGS.training_directory , "model"), 
                    global_step=global_step_val)
                tf.compat.v1.logging.info("Exporting the model at step %s to %s.",
                    global_step_val, FLAGS.training_directory )
            pass
        except KeyboardInterrupt:
            tf.compat.v1.logging.info("User terminate Keyboard Interrupt")
            exit()
        except Exception as e:
            print(e)
            tf.compat.v1.logging.info("Execptional error, Saving model just in case, while printing exception.")
            saver.save(sess, 
                os.path.join( FLAGS.training_directory , "model"), 
                global_step=global_step_val)
            tf.compat.v1.logging.info("Exporting the model at step %s to %s.",
                global_step_val, FLAGS.training_directory )

    if FLAGS.num_epochs == None:  
        tf.compat.v1.logging.error("Changing parameters: min_frames " + str(FLAGS.min_frames))
        FLAGS.min_frames = FLAGS.min_frames + 10
        tf.compat.v1.reset_default_graph()
        run()


if __name__ == "__main__":

    util.make_dirs([FLAGS.training_directory])


    run()
