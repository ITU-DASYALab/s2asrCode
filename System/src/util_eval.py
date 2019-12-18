
import numpy
import util
import tensorflow as tf

from jiwer import wer
import textdistance

import multiprocessing as mp
import time
import os
import numpy as np

## From Deep Speech
from ds_ctcdecoder import ctc_beam_search_decoder, Scorer
from extra.config import Config, initialize_globals

from tensorflow import flags

FLAGS = flags.FLAGS

from scipy.special import softmax

def ctc_decoder(x):
    scorer = Scorer(
        FLAGS.lm_alpha, 
        FLAGS.lm_beta,
        FLAGS.lm_binary_path, 
        FLAGS.lm_trie_path,
        Config.alphabet)

    values = np.squeeze(x[0][:x[1]])
    #spacer = values[:,28]
    #empty = values[:,29]
    #new_empty = np.reshape(spacer + empty, (x[1],1))
    #values = np.delete(values,29, 1)
    #values = np.delete(values,28, 1)
    #values = np.concatenate((values, new_empty), axis=1)
    #values = softmax(values, axis=0)
    return ctc_beam_search_decoder(
        values, 
        Config.alphabet, 
        FLAGS.lm_beam_width,
        scorer=scorer, 
        cutoff_prob=FLAGS.cutoff_prob,
        cutoff_top_n=FLAGS.cutoff_top_n)[0][1] 

def evaluate(x):

    scorer = Scorer(
        FLAGS.lm_alpha, 
        FLAGS.lm_beta,
        FLAGS.lm_binary_path, 
        FLAGS.lm_trie_path,
        Config.alphabet)

    values = np.squeeze(x[0][:x[1]])
    spacer = values[:,28]
    empty = values[:,29]
    new_empty = np.reshape(spacer + empty, (x[1],1))
    values = np.delete(values,29, 1)
    values = np.delete(values,28, 1)
    values = np.concatenate((values, new_empty), axis=1)
    #values = softmax(values, axis=0)


    if FLAGS.real_language_model:
        output_sentences_utf8 = ctc_beam_search_decoder(
            values, 
            Config.alphabet, 
            FLAGS.lm_beam_width,
            scorer=scorer, 
            cutoff_prob=FLAGS.cutoff_prob,
            cutoff_top_n=FLAGS.cutoff_top_n)[0][1]
    else:
        output_sentences_utf8 =  x[2][0].decode("utf-8").strip().replace('´',"")


    labels_characters_utf8 = x[3].decode("utf-8").lower().replace('´',"")

    WER = wer(labels_characters_utf8, output_sentences_utf8)
    CER = textdistance.levenshtein.normalized_distance(labels_characters_utf8, output_sentences_utf8)

    return (output_sentences_utf8, labels_characters_utf8, WER, CER)

class EvaluationMetrics(object):
    def __init__(self):
        self.file_print_path = os.path.join(FLAGS.training_directory, FLAGS.summary_name, "sentences.txt")
        self.output_sentences = []
        self.target_sentences = []
        self.clear()


        if FLAGS.word_based:
            dictionary = util.load_dictionary("dict/" + FLAGS.dictionary + "_word.txt")
        else:
            dictionary = util.load_dictionary("dict/" + FLAGS.dictionary + ".txt")

        self.dict = dictionary
        initialize_globals()
    
    def clear(self):
        with open(self.file_print_path, 'w+', encoding='utf-8') as file:
            file.write("Output Sentences: \n") 
            for x in zip(self.output_sentences, self.target_sentences):
                file.write(x[0] + "\n") 
                file.write(x[1] + "\n") 
                file.write("--" + "\n") 
        self.num_examples = 0
        self.sum_WER = 0.0
        self.sum_CER = 0.0
        self.sum_loss = 0.0
        self.output_sentences = []
        self.target_sentences = []

    

    def accumulate(self, predictions, output_sentences, labels, loss, labels_characters, output_length):
        #Get Batch Size
        batch_size = predictions.shape[0]

        #Get Loss sum and average
        loss_sum = loss * batch_size

        pool = mp.Pool()
        results = pool.map(evaluate, zip(predictions, output_length, output_sentences, labels_characters))
        pool.close()
        pool.join()

        #Get Output sentences and label sentences
        output_sentences_utf8 = [x[0] for x in results]
        labels_characters_utf8 = [x[1] for x in results]
        WER = [x[2] for x in results]
        CER = [x[3] for x in results]

        self.target_sentences.extend(output_sentences_utf8)
        self.output_sentences.extend(labels_characters_utf8)

        [print("\nl: "+ label + "\no: " +  output_sentences_utf8[idx] + "\n Wer:" + str(WER[idx])) 
            for idx, label in enumerate(labels_characters_utf8)]

        # Get WER sum and average 
        WER_sum = sum(WER)
        WER_average = WER_sum / batch_size
        
        # Get CER sum and average
        CER_sum = sum(CER)
        CER_average = CER_sum / batch_size

        # Add batch sums to accumulated sum
        self.sum_WER += WER_sum
        self.sum_CER += CER_sum
        self.sum_loss += loss_sum
        self.num_examples += batch_size

        # Return batch info dictionary for printing and Tensorboard
        info_dict_step = {
            "batch_loss": loss,
            "WER_batch": WER_average,
            "CER_batch": CER_average
        }
        return info_dict_step

    def get(self):
        avg_WER = self.sum_WER / self.num_examples
        avg_CER = self.sum_CER / self.num_examples
        avg_loss = self.sum_loss / self.num_examples

        info_dict = {
            "avg_WER": avg_WER,
            "avg_CER": avg_CER,
            "avg_loss": avg_loss
        }
        return info_dict



