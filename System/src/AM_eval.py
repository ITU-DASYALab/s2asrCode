
import os
import tensorflow as tf
import time
import json
import numpy as np
from tensorflow import flags

import util
import models
import record_reader
import util_eval
import graph_builder


FLAGS = flags.FLAGS

if __name__ == "__main__":
    ## Saving settings
    flags.DEFINE_string("training_directory", "models/default/", 
        "The folder containing the model that needs to be evaluated.")
    flags.DEFINE_string("summary_name","eval",
        "The name of the summary inside the training directory")

    flags.DEFINE_string("model", "DeepSpeech1",
        "the name of the model in use")
    flags.DEFINE_integer("batch_size", 32, "The batch size")
    flags.DEFINE_float("dropout", 0.0, "The dropout value between layers")
    flags.DEFINE_float("max_gpu_memory_use", 0.99, "The maximum percentage memory allowed to be used by this process")
    flags.DEFINE_bool("include_batch_summary", False, "Boolean to define if the batch summary is included in the summary")
    ## Logging settings
    flags.DEFINE_bool("help", False, "Boolean to print all flags")
    flags.DEFINE_integer("num_epochs", 1, "Number of iterations of the dataset, In evaluation it should be 1 for a deterministic model")
    flags.DEFINE_integer("logging_level", 20, 
        "The logging level specifying how much to log to console"
        "Debug = 10, Info = 20, Warn = 30, Error = 40, Fatal = 50")
    flags.DEFINE_boolean("wait_for_checkpoint", True, "Defines if the evaluation should run once or multiple times.")
    flags.DEFINE_boolean("save_sentences", True, "If the outputsentences should be saved to a file.")
    
    
    flags.DEFINE_boolean("real_language_model", False,"Use the language model on the logits result.")
    
    # Language model parameters (Credit Mozilla Deep Speech)
    flags.DEFINE_string("lm_binary_path", "lm/lm.binary", "the path to the language model.")
    flags.DEFINE_string("lm_trie_path", "lm/trie", "the trie file created while making the language model.")
    flags.DEFINE_integer("lm_beam_width", 1024, "The specific beamwidth of the full language model.")
    flags.DEFINE_float('lm_alpha', 0.75, 'the alpha hyperparameter of the CTC decoder. Language Model weight.')
    flags.DEFINE_float('lm_beta', 1.85, 'the beta hyperparameter of the CTC decoder. Word insertion weight.')
    flags.DEFINE_float('cutoff_prob', 1.0, 'only consider characters until this probability mass is reached. 1.0 = disabled.')
    flags.DEFINE_integer('cutoff_top_n', 300, 'only process this number of characters sorted by probability mass for each time step. If bigger than alphabet size, disabled.')

    
    tf.compat.v1.logging.set_verbosity(FLAGS.logging_level)
    if FLAGS.help:
        print(FLAGS)
        exit()

    tf.compat.v1.enable_resource_variables()

def evaluation_loop(
        collection, 
        saver,
        summary_writer,
        evl_metrics,
        latest_checkpoint,
        iterator
    ):
    config = tf.compat.v1.ConfigProto(
            gpu_options=tf.compat.v1.GPUOptions(
                allow_growth=True,
                per_process_gpu_memory_fraction=FLAGS.max_gpu_memory_use
                ), 
            allow_soft_placement=True)

    with tf.compat.v1.Session(
        config=config) as sess:
        #tf.initialize_all_variables().run()
        
        # to generate sentences
        tf.compat.v1.tables_initializer().run() 
        
        if latest_checkpoint:
            #tf.compat.v1.logging.info("Loading checkpoint for eval: %s", latest_checkpoint)
            saver.restore(sess, latest_checkpoint)
            global_step_val = os.path.basename(latest_checkpoint).split("-")[-1]

        
        sess.run(iterator.initializer)
        #tf.compat.v1.logging.info(
        #    "global_step_val = %s. ",global_step_val)
        

        examples_processed = 0
        
        try:
            while True:
                # execute the eval step
                batch_start_time = time.time()
                output_data_dict = sess.run(collection)

                # Get the Results
   
                predictions_val = output_data_dict["predictions"]

                iteration_info_dict = evl_metrics.accumulate( 
                    predictions_val, 
                    output_data_dict["sentence"], 
                    output_data_dict["label"], 
                    output_data_dict["loss"], 
                    output_data_dict["label_characters"],
                    output_data_dict["output_num_frames"])

                seconds_per_batch = time.time() - batch_start_time
                
                if FLAGS.include_batch_summary:
                    # Calculate the examples per second.
                    example_per_second = predictions_val.shape[0] / seconds_per_batch
                    examples_processed += predictions_val.shape[0]
                    iteration_info_dict["examples_per_second"] = example_per_second
                    for x in iteration_info_dict:
                        summary_writer.add_summary( 
                            util.makeCustomSummary(
                                float(iteration_info_dict[x]), 
                                "eval_summaries/" + str(x) ), 
                            global_step_val)

                    summary_writer.flush()

                    tf.compat.v1.logging.info(
                        "%15s" % "batch" + 
                        " |  %6d" % examples_processed +
                        " | loss: % 7.2f" % iteration_info_dict["batch_loss"] +
                        " | WER: % 5.4f" % iteration_info_dict["WER_batch"] + 
                        " | CER: % 5.4f" % iteration_info_dict["CER_batch"] +
                        " | Ex/S: % 8.2f" % example_per_second +
                        "")



        except tf.errors.OutOfRangeError:
            #tf.compat.v1.logging.info("Done with batched inference.")
            #tf.compat.v1.logging.info("Calculating global performance metrics.")
            epoch_info_dict = evl_metrics.get()
            #epoch_info_dict["epoch_id"] = global_step_val
            #summary_writer.add_summary(summary_val, global_step_val)
            for x in epoch_info_dict:

                summary_writer.add_summary( 
                    util.makeCustomSummary(
                        float(epoch_info_dict[x]), 
                        "eval_summaries/" + str(x) ), 
                    global_step_val)

            summary_writer.flush()
            tf.compat.v1.logging.info(
                "%15s"  % ("Total - " + FLAGS.summary_name) + 
                " |  %6d" % examples_processed +
                " | loss: % 7.2f" % epoch_info_dict["avg_loss"] +
                " | WER: % 5.4f" % epoch_info_dict["avg_WER"] + 
                " | CER: % 5.4f" % epoch_info_dict["avg_CER"])
        
        
        evl_metrics.clear()
        return global_step_val


def get_training_flags():
    flags_json_path = os.path.join(FLAGS.training_directory, "model_flags.json")
    if os.path.exists(flags_json_path):
        with open(flags_json_path, "r") as read_file:
            existing_flags = json.load(read_file)
    else:
        tf.compat.v1.logging.error(
            "Model flags file does not exist in path, Are you using the correct path?: %s",
            flags_json_path)
    return existing_flags

def evaluate():
    """Starts main evaluation loop."""
    tf.compat.v1.set_random_seed(0)  # for reproducibility

    flags_dict = get_training_flags()

    feature_names, feature_sizes = (
        flags_dict["feature_names"],
        flags_dict["feature_sizes"])

    model = util.find_class_by_name(flags_dict["model"],[models])()
    reader = record_reader.Reader(
        feature_names=feature_names,
        feature_sizes=feature_sizes,
        max_frames=FLAGS.max_frames,
        include_org_signal=False,
        connect_features=True)

    graph_builder.build_evaluation_graph(model, reader)
    tf.compat.v1.logging.info("built evaluation graph")

    collection = {
        "global_step":      tf.compat.v1.get_collection("global_step")[0],
        "loss":             tf.compat.v1.get_collection("loss")[0],
        "predictions":      tf.compat.v1.get_collection("predictions")[0],
        #"output_num_frames_padded" : tf.compat.v1.get_collection("output_num_frames_padded")[0],
        #"predictions_time_major" : tf.compat.v1.get_collection("predictions_time_major")[0],
        #"output_padded":    tf.compat.v1.get_collection("output_padded")[0],
        #"beam_output_values": tf.compat.v1.get_collection("beam_output_values")[0],
        #"tower_sequence_length": tf.compat.v1.get_collection("tower_sequence_length")[0],
        #"input_batch_raw":  tf.compat.v1.get_collection("model_input_raw")[0],
        "sentence":         tf.compat.v1.get_collection("sentence")[0],
        "output_num_frames":       tf.compat.v1.get_collection("output_num_frames")[0],
        "label":           tf.compat.v1.get_collection("label")[0],
        "label_characters": tf.compat.v1.get_collection("label_characters")[0],
        #"summary_op":       tf.compat.v1.get_collection("summary_op")[0]
    }

    iterator = tf.compat.v1.get_collection("iterator")[0]

    # To remove the errors for saving
    tf.compat.v1.get_default_graph().clear_collection('iterator')
    tf.compat.v1.get_default_graph().clear_collection('label')
    tf.compat.v1.get_default_graph().clear_collection('beam_output_values')
    tf.compat.v1.get_default_graph().clear_collection('tower_sequence_length')
    tf.compat.v1.get_default_graph().clear_collection('tower_target_strings_sparse')
    tf.compat.v1.get_default_graph().clear_collection('tower_output_num_frames')
    tf.compat.v1.get_default_graph().clear_collection('tower_model_input')
    tf.compat.v1.get_default_graph().clear_collection('summary_op')

    saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables())

    summary_writer = tf.compat.v1.summary.FileWriter(
        os.path.join(FLAGS.training_directory, FLAGS.summary_name), 
        graph=tf.compat.v1.get_default_graph())

    evl_metrics = util_eval.EvaluationMetrics()

    checkpoint = tf.train.latest_checkpoint(FLAGS.training_directory)
    checkpoint_old = ""
    waited = 0
    while True:
        if checkpoint is not None and checkpoint_old != checkpoint:
            evaluation_loop(
                collection, 
                saver,
                summary_writer,
                evl_metrics,
                checkpoint,
                iterator)
            checkpoint_old = checkpoint
        else:
            waited += 10
            if waited > 100:
                tf.compat.v1.logging.info("Waiting for new checkpoint")
                waited = 0
            time.sleep(10)

        checkpoint = tf.train.latest_checkpoint(FLAGS.training_directory)
        
        if not FLAGS.wait_for_checkpoint:
            break

    
if __name__ == "__main__":

    evaluate()