

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

from transformers import AutoTokenizer, AutoModel

from difflib import SequenceMatcher
import difflib

from Levenshtein import distance

import warnings
# Filter out specific warning type
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)



# FUNCTIONS    
# ==========================================
# class to select most similar sentences in dataframe
class SimFiltering:

    def __init__(self, sim_matrix, chunks_1, chunks_2, top_quantile, precision_label):
        self.sim_matrix = sim_matrix
        self.chunks_1 = chunks_1
        self.chunks_2 = chunks_2
        self.top_quantile = top_quantile
        self.precision_label = precision_label

    # keeping sentences with similarity score above threshold and sorting them by sim score
    def get_comp_sentences(self, sim_df, sentences1, sentences2, threshold = 0):
        result  =sim_df.stack()[sim_df.stack() > threshold]
        indices = [(idx[0], idx[1]) for idx in result.index]

        df_comp = pd.DataFrame(columns=['sent1', 'sent2', 'sim_score'])
        for row, col in indices:
            df_comp = df_comp.append([
                {'sent1': sentences1[row],
                'sent2': sentences2[col],
                'sim_score': sim_df.iloc[row][col]}
            ])
        df_comp = df_comp.sort_values(by='sim_score', ascending = False)
        return df_comp


    # keeping sentences with sim score above certain quantile or sim score 
    def get_top_comp(self, df_comp, value, precision_label):
        if precision_label == 'selection_quantile':
            val_top = df_comp['sim_score'].quantile(value)
            df_comp_top = df_comp[df_comp['sim_score'] >= val_top]
        elif precision_label == 'selection_sim_score':
            df_comp_top = df_comp[df_comp['sim_score'] >= value]
        else:
            print('error - precision_label')
        return df_comp_top
    
    # removing duplicates in comparison dataframe
    def filter_df_comp_top(self, df):
        result = df.sort_values(by = ['sent1', 'sim_score'], ascending=[True, False])
        result = result.drop_duplicates(subset='sent1', keep='first')
        
        result = result.sort_values(by = ['sent2', 'sim_score'], ascending=[True, False])
        result = result.drop_duplicates(subset='sent2', keep='first')
        
        result = result.sort_values(by = ['sim_score'], ascending = False)
        return result
    

    # return filtered comparison dataframe with top similarity
    def return_filt_df(self):

        sim_df = pd.DataFrame(self.sim_matrix)
        df_comp = self.get_comp_sentences(sim_df, self.chunks_1, self.chunks_2)
        df_comp_top = self.get_top_comp(df_comp, self.top_quantile, self.precision_label)
        df_top_filt = self.filter_df_comp_top(df_comp_top)

        return df_top_filt



# class to turn text into tokens
class Tokenizer:
    def __init__(self, text1, text2):
        self.text1 = text1
        self.text2 = text2

    # convert text into list of sentences (chunks)
    def break_into_chunks(self, text, split):
        if split == 'SENTENCES':
            sentences = [sentence.strip() for sentence in text.split('.')]
            sentences = [sentence for sentence in sentences if sentence]
            return sentences
        elif split == 'NGRAMS':
            n = 3
            words = text.split()
            chunks = [" ".join(words[i:i+n]) for i in range(0, len(words), n)]
            return chunks
        else:
            print('error split method')


    def return_chunks(self, method = "SENTENCES"):
        chunks_1 = self.break_into_chunks(self.text1, split = method)
        chunks_2 = self.break_into_chunks(self.text2, split = method)

        return chunks_1, chunks_2




# class to get dataframe for lexical similarity
class LexicalSim:

    def __init__(self, chunks_1, chunks_2, top_quantile, precision_label):
        self.chunks_1 = chunks_1
        self.chunks_2 = chunks_2
        self.top_quantile = top_quantile
        self.precision_label = precision_label


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
    
        if method == "JACCARD":
            sim_matrix_jaccard = np.array(self.jaccard_similarity_matrix(self.chunks_1, self.chunks_2))
            simfilt = SimFiltering(sim_matrix_jaccard, self.chunks_1, self.chunks_2, self.top_quantile, self.precision_label)
            df_top_filt_jaccard = simfilt.return_filt_df()
            return df_top_filt_jaccard
        elif method == "LEVENSHTEIN":
            sim_matrix_lev = np.array(self.levenshtein_matrix(self.chunks_1, self.chunks_2))
            simfilt = SimFiltering(sim_matrix_lev, self.chunks_1, self.chunks_2, self.top_quantile, self.precision_label)
            df_top_filt_lev = simfilt.return_filt_df()
            return df_top_filt_lev
        elif method == "HAMMING":
            sim_matrix_hamming = np.array(self.hamming_normalized_distance(self.chunks_1, self.chunks_2))
            simfilt = SimFiltering(sim_matrix_hamming, self.chunks_1, self.chunks_2, self.top_quantile, self.precision_label)
            df_top_filt_hamming = simfilt.return_filt_df()
            return df_top_filt_hamming
        elif method == "JARO-WINKLER":
            sim_matrix_jarowinkler = np.array(self.jaro_winkler_matrix(self.chunks_1, self.chunks_2))
            simfilt = SimFiltering(sim_matrix_jarowinkler, self.chunks_1, self.chunks_2, self.top_quantile, self.precision_label)
            df_top_filt_jarowinkler= simfilt.return_filt_df()
            return df_top_filt_jarowinkler
        



# class to get similarity dataframe with embeddings
class EmbeddingsSim:
    def __init__(self, chunks_1, chunks_2, embedding_model, top_quantile, precision_label):
        self.chunks_1 = chunks_1
        self.chunks_2 = chunks_2
        self.embedding_model = embedding_model
        self.top_quantile = top_quantile
        self.precision_label = precision_label

    # using tokenizer to convert chunks to embeddings using transformers
    def embed_chunks_transformers(self, sentences, tokenizer, model):
        inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
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

        if embeddings_model == "minilm":
            MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
        elif embeddings_model == "bert_base":
            MODEL_NAME = "bert-base-multilingual-cased"
        elif embeddings_model == "distiluse":
            MODEL_NAME = "sentence-transformers/distiluse-base-multilingual-cased-v2"
        else:
            print("error embeddings", embeddings_model)

        
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModel.from_pretrained(MODEL_NAME)

        embeddings1 = self.embed_chunks_transformers(self.chunks_1, tokenizer, model).numpy()
        embeddings2 = self.embed_chunks_transformers(self.chunks_2, tokenizer, model).numpy()

        return embeddings1, embeddings2

    # get comparison dataframe depending on given method
    def return_comp_df(self, submethod):

        embeddings1, embeddings2 = self.get_embeddings(self.embedding_model)
        
        if submethod == "COSINE":
            sim_matrix_cos = np.array(util.pytorch_cos_sim(embeddings1, embeddings2))
            simfilt = SimFiltering(sim_matrix_cos, self.chunks_1, self.chunks_2, self.top_quantile, self.precision_label)
            df_top_filt_cos = simfilt.return_filt_df()
            return df_top_filt_cos
        elif submethod == "EUCLIDEAN":
            dist_matrix_euclid = pairwise_distances(embeddings1, embeddings2, metric = 'euclidean')
            sim_matrix_euclid = self.dist_to_sim(dist_matrix_euclid)
            simfilt = SimFiltering(sim_matrix_euclid, self.chunks_1, self.chunks_2, self.top_quantile, self.precision_label)
            df_top_filt_euclid = simfilt.return_filt_df()
            return df_top_filt_euclid
        elif submethod == "DOT":
            sim_matrix_dot = self.get_inner_product_matrix(embeddings1, embeddings2)
            simfilt = SimFiltering(sim_matrix_dot, self.chunks_1, self.chunks_2, self.top_quantile, self.precision_label)
            df_top_filt_dot = simfilt.return_filt_df()
            return df_top_filt_dot
        else:
            print("error submethod")



# class to get similarity dataframe with hybrid method
class Hybrid:
    def __init__(self, chunks_1, chunks_2, submethods, lex_coef, embedding_model, top_quantile, precision_label):
        self.chunks_1 = chunks_1
        self.chunks_2 = chunks_2
        self.submethods = submethods
        self.lex_coef = lex_coef
        self.precision_label = precision_label

        self.top_quantile = top_quantile
        self.lexsim = LexicalSim(self.chunks_1, self.chunks_2, top_quantile, precision_label)
        self.embsim = EmbeddingsSim(self.chunks_1, self.chunks_2, embedding_model, top_quantile, precision_label)

        self.embedding_model = embedding_model
    
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
        simfilt = SimFiltering(weighted_matrix, self.chunks_1, self.chunks_2, self.top_quantile, self.precision_label)
        df_top_filt_weighted = simfilt.return_filt_df()
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
def call(text1, text2, method,slidingValue,  submethods, embedding_model, top_quantile, precision_label):

    # Getting chunks
    tokenizer = Tokenizer(text1, text2)
    chunks_1, chunks_2 = tokenizer.return_chunks()

    if slidingValue != None:
        print(slidingValue)

    # For lexical and embeddings section
    submethod = submethods[0]

    if method == "LEXICAL":
        lexsim = LexicalSim(chunks_1, chunks_2, top_quantile, precision_label)
        df_comp = lexsim.return_comp_df(submethod)

    elif method == "EMBEDDINGS":
        embsim = EmbeddingsSim(chunks_1, chunks_2, embedding_model, top_quantile, precision_label)
        df_comp = embsim.return_comp_df(submethod)

    elif method == "HYBRID":
        hybridsim = Hybrid(chunks_1, chunks_2, submethods, slidingValue, embedding_model, top_quantile, precision_label)
        df_comp = hybridsim.return_comp_df()
    else:
        print("Error method", method)

    return df_comp

    

# adding highlighting in HTML format and breaklines
def replace_sentences_html(text_1, text_2, sent_list_1, sent_list_2, colors_list):
    n = len(sent_list_1)

    text_1_store = text_1
    text_2_store = text_2

    for i in range(n):
        new_sent_1 = '<span style="background-color: ' + colors_list[i] + ';">' + sent_list_1[i] + '</span>'
        new_sent_2 = '<span style="background-color: ' + colors_list[i] + ';">' + sent_list_2[i] + '</span>'

        text_1_store = text_1_store.replace(sent_list_1[i], new_sent_1)
        text_2_store = text_2_store.replace(sent_list_2[i], new_sent_2)
        
    text_1_store = text_1_store.replace('\n', '<br>')
    text_2_store = text_2_store.replace('\n', '<br>')

    return text_1_store, text_2_store