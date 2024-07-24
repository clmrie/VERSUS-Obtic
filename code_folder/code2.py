

# LIBRARIES
# ==========================================
# Importing libraries
import os
import pandas as pd
import numpy as np
import string
from unidecode import unidecode
import unicodedata
import re
import torch
from tqdm import tqdm



from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tkinter as tk
import textdistance
from sentence_transformers import SentenceTransformer, util
from IPython.display import display, Image
from Levenshtein import distance
from transformers import AutoTokenizer, AutoModel

from difflib import SequenceMatcher
import difflib

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer



import warnings
# Filter out specific warning type
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)



# FUNCTIONS    
# ==========================================
# class to select most similar sentences in dataframe
class SimFiltering:

    def __init__(self, sim_matrix, chunks_1, chunks_2, top_quantile, precision_label, nb_sentences_value):
        self.sim_matrix = sim_matrix
        self.chunks_1 = chunks_1
        self.chunks_2 = chunks_2
        self.top_quantile = top_quantile
        self.precision_label = precision_label
        self.nb_sentences_value = nb_sentences_value

    def return_filt_df(self, method_filt,dict_filt):

        print("method_filt:", method_filt)
        if method_filt == "TOP":
            
            similar_pairs = []
            for i in range(len(self.chunks_1)):
                for j in range(len(self.chunks_2)):
                    similar_pairs.append((self.chunks_1[i], self.chunks_2[j], self.sim_matrix[i, j]))
            
            df = pd.DataFrame(similar_pairs, columns = ['sent1', 'sent2', 'sim_score'])
            df = df.sort_values(by='sim_score', ascending = False)
            
            df = df.drop_duplicates(subset='sent1', keep='first')
            df = df.drop_duplicates(subset='sent2', keep='first')

            if self.precision_label == 'selection_sentences':
                df = df.head(int(self.nb_sentences_value))
            else:
                quantile_value = dict_filt['quantile_value']
                print("quantile_value:", quantile_value)
                quantile_value = df['sim_score'].quantile(quantile_value)
                df = df[df['sim_score'] >= quantile_value]  
                print('quantile_value', quantile_value) 
                df = df.head(10)
            return df
        

        elif method_filt == "THRESHOLD":
            threshold = dict_filt['threshold']

            print('matrix:', self.sim_matrix)

            similar_pairs = []
            for i in range(len(self.chunks_1)):
                for j in range(len(self.chunks_2)):
                    if self.sim_matrix[i, j] > threshold:
                        similar_pairs.append((self.chunks_1[i], self.chunks_2[j], self.sim_matrix[i, j]))
            
            df = pd.DataFrame(similar_pairs, columns = ['sent1', 'sent2', 'sim_score'])
            df = df.drop_duplicates(subset='sent1', keep='first')
            df = df.drop_duplicates(subset='sent2', keep='first')

            df = df.sort_values(by='sim_score', ascending = False)

            return df
    
   




# class to turn text into tokens
class Tokenizer:
    def __init__(self, text1, text2):
        self.text1 = text1
        self.text2 = text2

    def split_and_filter(self, text, seps):
        # Create a regular expression pattern to match any of the separators
        pattern = '|'.join(map(re.escape, seps))
        
        # Split the text using the pattern
        lines = re.split(pattern, text)
        
        # Filter and strip the lines
        result = [line.strip() for line in lines if len(line.strip().split()) > 2]
        
        return result
    
    def generate_word_windows2(self, text, n_grams, n_overlap):
        # Tokenize the text into words
        words = word_tokenize(text)
        # List to store the windows
        windows = []
        # Handle case where n_shift is 0 (non-overlapping windows)
        if n_overlap == 0:
            n_overlap = n_grams  # Make the shift equal to the window size to avoid overlap

        # Generate windows with the specified shift
        for i in range(0, len(words) - n_grams + 1, n_overlap):
            window = words[i:i + n_grams]
            window = TreebankWordDetokenizer().detokenize(window)
            window = re.sub(r"\s+'", "'", re.sub(r"'\s+", "'", window))
            windows.append(window)

        # Handle the case where the last segment might be shorter than n but still needs to be included
        if len(words) % n_grams != 0 and len(words) % n_overlap != 0:
            last_window = words[-n_grams:]
            last_window = TreebankWordDetokenizer().detokenize(last_window)
            last_window = re.sub(r"\s+'", "'", re.sub(r"'\s+", "'", last_window))
            windows.append(last_window)

        return windows
    
    def generate_word_windows(self , text, n_grams, n_overlap):

        # Split the text into words based on whitespace
        words = text.split()

        # List to store the windows
        windows = []

        # Handle case where n_overlap is 0 (non-overlapping windows)
        if n_overlap == 0:
            n_overlap = n_grams  # Make the shift equal to the window size to avoid overlap

        # Generate windows with the specified shift
        for i in range(0, len(words) - n_grams + 1, n_overlap):
            window = words[i:i + n_grams]
            window = ' '.join(window)  # Join the words to form a window
            # Optionally, you can apply any additional processing here
            windows.append(window)

        # Handle the case where the last segment might be shorter than n but still needs to be included
        if len(words) % n_grams != 0 and len(words) % n_overlap != 0:
            last_window = words[-n_grams:]
            last_window = ' '.join(last_window)  # Join the words to form the last window
            # Optionally, you can apply any additional processing here
            windows.append(last_window)

        return windows


    def return_chunks(self, method, dict_method):

        # by default we set 0
        n_overlap = 0

        print("method ", method)

        if method == 'NGRAMS':
            chunks_1 = self.generate_word_windows(self.text1, n_grams = dict_method['n_grams'], n_overlap = n_overlap)
            chunks_2 = self.generate_word_windows(self.text2, n_grams = dict_method['n_grams'], n_overlap = n_overlap)
        elif method == 'SENTENCES':
            chunks_1 = self.split_and_filter(self.text1, seps=dict_method['seps'])
            chunks_2 = self.split_and_filter(self.text2, seps = dict_method['seps'])
        else:
            print("error method return chunks")

        return chunks_1, chunks_2




# class to get dataframe for lexical similarity
class LexicalSim:

    def __init__(self, chunks_1, chunks_2, top_quantile, precision_label, nb_sentences_value):
        self.chunks_1 = chunks_1
        self.chunks_2 = chunks_2
        self.top_quantile = top_quantile
        self.precision_label = precision_label
        self.nb_sentences_value = nb_sentences_value


    # get similarity matrix with Jaccard
    def jaccard_similarity_matrix(self, sentences1, sentences2):
        words1 = [set(sentence.lower().split()) for sentence in sentences1]
        words2 = [set(sentence.lower().split()) for sentence in sentences2]
        
        words1_array = np.array(words1)
        words2_array = np.array(words2)
        
        intersection_matrix = np.array([len(words1_array[i].intersection(words2_array[j])) for i in range(len(words1_array)) for j in range(len(words2_array))]).reshape(len(words1_array), len(words2_array))
        union_matrix = np.array([len(words1_array[i].union(words2_array[j])) for i in range(len(words1_array)) for j in range(len(words2_array))]).reshape(len(words1_array), len(words2_array))
        
        similarity_matrix = intersection_matrix / union_matrix
        
        return similarity_matrix
    
    # get similarity matrix for Levenshtein
    def levenshtein_matrix(self, list1, list2):
        distance_matrix = []
        for sentence1 in list1:
            row = []
            for sentence2 in list2:
                lev_distance = 1/(1+distance(sentence1, sentence2))
                row.append(lev_distance)
            distance_matrix.append(row)
        return distance_matrix
    
    # get similarity matrix for Hamming distance
    def hamming_normalized_distance(self, list1, list2):
        distance_matrix = []
        for sentence1 in list1:
            row = []
            for sentence2 in list2:
                lev_distance = textdistance.hamming.normalized_similarity(sentence1, sentence2)
                row.append(lev_distance)
            distance_matrix.append(row)
        return distance_matrix
    
    # get similarity matrix with Jaro Winkler
    def jaro_winkler_matrix(self, list1, list2):
        similarity_matrix = []
        for sentence1 in list1:
            row = []
            for sentence2 in list2:
                similarity = textdistance.jaro_winkler.similarity(sentence1, sentence2)
                row.append(similarity)
            similarity_matrix.append(row)
        return similarity_matrix

    def return_comp_df(self, method):

        if self.precision_label == 'selection_quantile':
            method_filt = "TOP"
            dict_filt = {'quantile_value': self.top_quantile}
        elif self.precision_label == 'selection_sim_score':
            method_filt = "THRESHOLD"
            dict_filt = {'threshold': self.top_quantile}
        else:
            method_filt = "TOP"
            dict_filt = {'threshold': self.top_quantile}


        if method == "JACCARD":
            sim_matrix_jaccard = np.array(self.jaccard_similarity_matrix(self.chunks_1, self.chunks_2))
            simfilt = SimFiltering(sim_matrix_jaccard, self.chunks_1, self.chunks_2, self.top_quantile, self.precision_label,
                                   self.nb_sentences_value)
            
            df_top_filt_jaccard = simfilt.return_filt_df(method_filt, dict_filt)
            
            return df_top_filt_jaccard
        elif method == "LEVENSHTEIN":
            sim_matrix_lev = np.array(self.levenshtein_matrix(self.chunks_1, self.chunks_2))
            simfilt = SimFiltering(sim_matrix_lev, self.chunks_1, self.chunks_2, self.top_quantile, self.precision_label,
                                   self.nb_sentences_value)

            df_top_filt_lev = simfilt.return_filt_df(method_filt, dict_filt)
            return df_top_filt_lev
        elif method == "HAMMING":
            sim_matrix_hamming = np.array(self.hamming_normalized_distance(self.chunks_1, self.chunks_2))
            simfilt = SimFiltering(sim_matrix_hamming, self.chunks_1, self.chunks_2, self.top_quantile, self.precision_label,
                                   self.nb_sentences_value)

            df_top_filt_hamming = simfilt.return_filt_df(method_filt, dict_filt)

            return df_top_filt_hamming
        elif method == "JARO-WINKLER":
            sim_matrix_jarowinkler = np.array(self.jaro_winkler_matrix(self.chunks_1, self.chunks_2))
            simfilt = SimFiltering(sim_matrix_jarowinkler, self.chunks_1, self.chunks_2, self.top_quantile, self.precision_label,
                                   self.nb_sentences_value)

            df_top_filt_jarowinkler= simfilt.return_filt_df(method_filt, dict_filt)
            return df_top_filt_jarowinkler
        

# class to get similarity dataframe with embeddings
class EmbeddingsSim:
    def __init__(self, chunks_1, chunks_2, embedding_model, top_quantile, precision_label, nb_sentences_value):
        self.chunks_1 = chunks_1
        self.chunks_2 = chunks_2
        self.embedding_model = embedding_model
        self.top_quantile = top_quantile
        self.precision_label = precision_label
        self.nb_sentences_value = nb_sentences_value

    # using tokenizer to convert chunks to embeddings using transformers
    def embed_chunks(self, sentences, model):
        embeddings = model.encode(sentences, show_progress_bar=True)
        return embeddings

    # normalizing matrix
    def normalize_matrix(self, matrix):
        scaler = MinMaxScaler()
        scaled_matrix = scaler.fit_transform(matrix)
        return scaled_matrix

    # get similarity matrix from inner product
    def get_inner_product_matrix(self, embeddings1, embeddings2):
        num_embeddings1 = embeddings1.shape[0]
        num_embeddings2 = embeddings2.shape[0]

        inner_product_matrix = np.zeros((num_embeddings1, num_embeddings2))

        # Calculate inner product for each pair of embeddings using nested loops
        for i in range(num_embeddings1):
            for j in range(num_embeddings2):
                inner_product_matrix[i, j] = np.dot(embeddings1[i], embeddings2[j])
        
        inner_product_matrix = self.normalize_matrix(inner_product_matrix)
        return inner_product_matrix
    
    # converting distance to similarity score
    def dist_to_sim(self, matrix):
        return 1/(1+matrix)
    

    # get embeddings depending on embeddings model
    def get_embeddings(self, embeddings_model):

        embeddings_model = 'stsb-xlm'
        if embeddings_model == "minilm":
            MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        elif embeddings_model == "bert_base":
            MODEL_NAME = "google-bert/bert-base-multilingual-cased"
        elif embeddings_model == "distiluse":
            MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v2"
        elif embeddings_model == 'stsb-xlm':
            MODEL_NAME = 'sentence-transformers/stsb-xlm-r-multilingual'
        else:
            print("error embeddings", embeddings_model)

    
        model = SentenceTransformer(MODEL_NAME)
        print('✨we got the model')

        embeddings1 = self.embed_chunks(self.chunks_1, model)
        embeddings2 = self.embed_chunks(self.chunks_2, model)

        return embeddings1, embeddings2

    # get comparison dataframe depending on given method
    def return_comp_df(self, submethod):

        embeddings1, embeddings2 = self.get_embeddings(self.embedding_model)
        print('✨ we got embeddings\n')

        if self.precision_label == 'selection_quantile':
            method_filt = "TOP"
            dict_filt = {'quantile_value': self.top_quantile}
        elif self.precision_label == 'selection_sim_score':
            method_filt = "THRESHOLD"
            dict_filt = {'threshold': self.top_quantile}
        else:
            method_filt = "TOP"
            dict_filt = {'quantile_value': self.top_quantile}

        
        if submethod == "COSINE":
            sim_matrix_cos = np.array(cosine_similarity(embeddings1, embeddings2))
            print('✨ we got sim matrix\n')
            simfilt = SimFiltering(sim_matrix_cos, self.chunks_1, self.chunks_2, self.top_quantile, self.precision_label,
                                   self.nb_sentences_value)
          
            df_top_filt_cos = simfilt.return_filt_df(method_filt, dict_filt)
            return df_top_filt_cos
        elif submethod == "EUCLIDEAN":
            dist_matrix_euclid = pairwise_distances(embeddings1, embeddings2, metric = 'euclidean')
            sim_matrix_euclid = self.dist_to_sim(dist_matrix_euclid)
            simfilt = SimFiltering(sim_matrix_euclid, self.chunks_1, self.chunks_2, self.top_quantile, self.precision_label,
                                   self.nb_sentences_value)

            df_top_filt_euclid = simfilt.return_filt_df(method_filt, dict_filt)
            return df_top_filt_euclid
        elif submethod == "DOT":
            sim_matrix_dot = self.get_inner_product_matrix(embeddings1, embeddings2)
            simfilt = SimFiltering(sim_matrix_dot, self.chunks_1, self.chunks_2, self.top_quantile, self.precision_label,
                                   self.nb_sentences_value)

            df_top_filt_dot = simfilt.return_filt_df(method_filt, dict_filt)
            return df_top_filt_dot
        else:
            print("error submethod")



# class to get similarity dataframe with hybrid method
class Hybrid:
    def __init__(self, chunks_1, chunks_2, submethods, lex_coef, embedding_model, top_quantile, precision_label, nb_sentences_value) :
        self.chunks_1 = chunks_1
        self.chunks_2 = chunks_2
        self.submethods = submethods
        self.lex_coef = lex_coef
        self.precision_label = precision_label

        self.top_quantile = top_quantile
        self.lexsim = LexicalSim(self.chunks_1, self.chunks_2, top_quantile, precision_label, nb_sentences_value)
        self.embsim = EmbeddingsSim(self.chunks_1, self.chunks_2, embedding_model, top_quantile, precision_label, nb_sentences_value)

        self.embedding_model = embedding_model

        self.nb_sentences_value = nb_sentences_value
    
    # return weighted matrix based on the two other matrices
    def calculate_weighted_matrix(self, lex_coef, lex_matrix, sem_matrix):
        lex_coef = float(lex_coef)
        sem_coef = 1 - float(lex_coef)
        sim_matrix_weighted = lex_coef*lex_matrix + sem_coef*sem_matrix
        return sim_matrix_weighted

    # returns hybrid matrix based on two submethods
    def get_hybrid_sim_matrix(self):

        submethod_lexical = self.submethods[0]
        submethod_semantic = self.submethods[1]

        # calculating two similarity matrices
        if submethod_lexical == "JACCARD":
            sim_matrix_lex = np.array(self.lexsim.jaccard_similarity_matrix(self.chunks_1, self.chunks_2))
        elif submethod_lexical == "LEVENSHTEIN":
            sim_matrix_lex = np.array(self.lexsim.levenshtein_matrix(self.chunks_1, self.chunks_2))
        elif submethod_lexical == "HAMMING":
            sim_matrix_lex = np.array(self.lexsim.hamming_normalized_distance(self.chunks_1, self.chunks_2))
        elif submethod_lexical == "JARO-WINKLER":
            sim_matrix_lex = np.array(self.lexsim.jaro_winkler_matrix(self.chunks_1, self.chunks_2))

        embeddings1, embeddings2 = self.embsim.get_embeddings(self.embedding_model)

        if submethod_semantic == "COSINE":
            sim_matrix_sem = np.array(util.pytorch_cos_sim(embeddings1, embeddings2))
        elif submethod_semantic == "EUCLIDEAN":
            dist_matrix_euclid = pairwise_distances(embeddings1, embeddings2, metric = 'euclidean')
            sim_matrix_sem = self.embsim.dist_to_sim(dist_matrix_euclid)
        elif submethod_semantic == "DOT":
            sim_matrix_sem = self.embsim.get_inner_product_matrix(embeddings1, embeddings2)

        weighted_matrix = self.calculate_weighted_matrix(self.lex_coef, sim_matrix_lex, sim_matrix_sem)
        return weighted_matrix
    

    # get sim matrix and most similar sentences
    def return_comp_df(self):
        weighted_matrix = self.get_hybrid_sim_matrix()
        simfilt = SimFiltering(weighted_matrix, self.chunks_1, self.chunks_2, self.top_quantile, self.precision_label, 
                               self.nb_sentences_value)

        if self.precision_label == 'selection_quantile':
            method_filt = "TOP"
            dict_filt = {'quantile_value': self.top_quantile}
        elif self.precision_label == 'selection_sim_score':
            method_filt = "THRESHOLD"
            dict_filt = {'threshold': self.top_quantile}
        else:
            method_filt = "TOP"
            dict_filt = {'quantile_value': self.top_quantile}
        
        df_top_filt_weighted = simfilt.return_filt_df(method_filt, dict_filt)
        return df_top_filt_weighted
        

# get indices for highlighting purposes in interface
def get_indices(df_comp, input_1, input_2):
    
    # find indices of substring in string
    def find_substring_indices(main_string, substring):
        start_index = main_string.find(substring)
        end_index = start_index + len(substring)
        return start_index, end_index

    nb_lines = df_comp.shape[0]
    
    highlight_ranges_text1 = []
    highlight_ranges_text2 = []
    sent_list_1 = []
    sent_list_2 = []
    
    for i in range(nb_lines):
        sent1 = df_comp['sent1'].values[i]
        sent2 = df_comp['sent2'].values[i]
        
        tuple1 = find_substring_indices(input_1, sent1)
        tuple2 = find_substring_indices(input_2, sent2)
        
        highlight_ranges_text1.append(tuple1)
        highlight_ranges_text2.append(tuple2)
        
        sent_list_1.append(sent1)
        sent_list_2.append(sent2)
        
    return nb_lines, highlight_ranges_text1, highlight_ranges_text2, sent_list_1, sent_list_2


# main function executing python
def call(text1, text2, method,slidingValue,  submethods, embedding_model, top_quantile, precision_label, textProcess, ngramsInput,
         nb_sentences_value):

    # Getting chunks
    tokenizer = Tokenizer(text1, text2)

    method_split = textProcess
    dict_method = {'seps': ['.', '?', '!'],
                   'n_grams': int(ngramsInput)}
    
    chunks_1, chunks_2 = tokenizer.return_chunks(method_split, dict_method)
    
    print('✨ return chunks\n')
    
    if slidingValue != None:
        print(slidingValue)

    # For lexical and embeddings section
    submethod = submethods[0]

    if method == "LEXICAL":
        lexsim = LexicalSim(chunks_1, chunks_2, top_quantile, precision_label, nb_sentences_value)
        df_comp = lexsim.return_comp_df(submethod)
        print(df_comp)

    elif method == "EMBEDDINGS":
        embsim = EmbeddingsSim(chunks_1, chunks_2, embedding_model, top_quantile, precision_label, nb_sentences_value)
        df_comp = embsim.return_comp_df(submethod)
        print(df_comp)

    elif method == "HYBRID":
        hybridsim = Hybrid(chunks_1, chunks_2, submethods, slidingValue, embedding_model, top_quantile, precision_label, nb_sentences_value)
        df_comp = hybridsim.return_comp_df()
    else:
        print("Error method", method)

    return df_comp

    


# adding highlighting in HTML format and breaklines
def replace_sentences_html(text_1, text_2, sent_list_1, sent_list_2, colors_list):
    n = len(sent_list_1)

    text_1_store = text_1
    text_2_store = text_2

    #sent_list_1 = [s.replace("'", '"') for s in sent_list_1]
    print(sent_list_1)


    for i in range(n):
        new_sent_1 = '<span style="background-color: ' + colors_list[i] + ';">' + sent_list_1[i] + '</span>'
        new_sent_2 = '<span style="background-color: ' + colors_list[i] + ';">' + sent_list_2[i] + '</span>'

        text_1_store = text_1_store.replace(sent_list_1[i], new_sent_1)
        text_2_store = text_2_store.replace(sent_list_2[i], new_sent_2)

    text_1_store = text_1_store.replace('\\n', '<br>')
    text_2_store = text_2_store.replace('\\n', '<br>')   

    return text_1_store, text_2_store

