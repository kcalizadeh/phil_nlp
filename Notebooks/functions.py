import pandas as pd
import requests
import numpy as np
import pickle
import re
import json
import string
import matplotlib.pyplot as plt
import wordcloud
import seaborn as sns

from google.colab import files

import plotly.express as px
from collections import Counter

from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import Phrases
from gensim.models.phrases import Phraser

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import BernoulliNB, BaseEstimator, BaseNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import plot_confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline

import lime
from lime import lime_text
from lime.lime_text import LimeTextExplainer

import pkg_resources

import nltk 
nltk.download('stopwords')
from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.collocations import *
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder

import spacy

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, GRU
from tensorflow.keras.layers import Dropout, Activation, Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Sequential
from tensorflow.keras import initializers, regularizers, constraints, optimizers, layers
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint


from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences



# gets text from a gutenberg URL
def get_guten(url):
    # retrieve the source text
    r = requests.get(url)
    r.encoding = 'utf-8'
    text = r.text
    return text

# gets the text from a txt file
def get_text(path, encoding='utf-8'):
    f = open(path, 'r', encoding=encoding)
    text = f.read()
    f.close()
    return text

def baseline_clean(to_correct, capitals=True, bracketed_fn=False):
  # remove utf8 encoding characters and some punctuations
  result = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff\xad\x0c6§\\\£\Â‘’“”*_<>''""⎫•{}]', ' ', to_correct)
  result = re.sub(r'[\u2014\u2013\u2012-]', ' ', result)

  # replace whitespace characters with actual whitespace
  result = re.sub(r'\s', ' ', result)

  # replace the ﬀ, ﬃ and ﬁ with the appropriate counterparts
  result = re.sub(r'ﬀ', 'ff', result)
  result = re.sub(r'ﬁ', 'fi', result)
  result = re.sub(r'ﬃ', 'ffi', result)

  # remove some recurring common and meaninless words/phrases
  result = re.sub(r'\s*This\s*page\s*intentionally\s*left\s*blank\s*', ' ', result)
  result = re.sub(r'(?i)Aufgabe\s+', ' ', result)
  result = re.sub(r',*\s+cf\.', ' ', result)
  result = re.sub('coroll\.', 'coroll', result)
  result = re.sub('pt\.', 'pt', result)

  # some texts have footnotes conveniently in brackets - this removes them all, 
  # with a safety measure for unpaired brackets, and deletes all brackets afterwards
  if bracketed_fn:
    result = re.sub(r'\[.{0,300}\]|{.{0,300}}|{.{0,300}\]|\[.{0,300}}', ' ', result)
  result = re.sub(r'[\[\]{}]', ' ', result)

  # replace ampersands with 'and'
  result = re.sub(r'&', 'and', result)

  # remove roman numerals, first capitalized ones
  result = re.sub(r'\s((I{2,}V*X*\.*)|(IV\.*)|(IX\.*)|(V\.*)|(V+I*\.*)|(X+L*V*I*]\.*))\s', ' ', result)
  # then lowercase
  result = re.sub(r'\s((i{2,}v*x*\.*)|(iv\.*)|(ix\.*)|(v\.*)|(v+i*\.*)|(x+l*v*i*\.*))\s', ' ', result)

  # remove periods and commas flanked by numbers
  result = re.sub(r'\d\.\d', ' ', result)
  result = re.sub(r'\d,\d', ' ', result)

  # remove the number-letter-number pattern used for many citations
  result = re.sub(r'\d*\w{,2}\d', ' ', result)

  # remove numerical characters
  result = re.sub(r'\d+', ' ', result)

  # remove words of 2+ characters that are entirely capitalized 
  # (these are almost always titles, headings, or speakers in a dialogue)
  # remove capital I's that follow capital words - these almost always roman numerals
  # some texts do use these capitalizations meaningfully, so we make this optional
  if capitals:
    result = re.sub(r'[A-Z]{2,}\s+I', ' ', result)
    result = re.sub(r'[A-Z]{2,}', ' ', result)

  # remove isolated colons and semicolons that result from removal of titles
  result = re.sub(r'\s+:\s*', ' ', result)
  result = re.sub(r'\s+;\s*', ' ', result)

  # remove isolated letters (do it several times because strings of isolated letters do not get captured properly)
  result = re.sub(r'\s[^aAI\.]\s', ' ', result)
  result = re.sub(r'\s[^aAI\.]\s', ' ', result)
  result = re.sub(r'\s[^aAI\.]\s', ' ', result)
  result = re.sub(r'\s[^aAI\.]\s', ' ', result)
  result = re.sub(r'\s[^aAI\.]\s', ' ', result)

  # remove isolated letters at the end of sentences or before commas
  result = re.sub(r'\s[^aI]\.', '.', result)
  result = re.sub(r'\s[^aI],', ',', result)

  # deal with spaces around periods and commas
  result = re.sub(r'\s,\s', ', ', result)
  result = re.sub(r'\s\.\s;', '. ', result)

  # remove empty parantheses
  result = re.sub(r'(\(\s*\.*\s*\))|(\(\s*,*\s*)\)', ' ', result)

  # reduce multiple periods or whitespaces into a single one
  result = re.sub(r'\.+', '.', result)
  result = re.sub(r'\s+', ' ', result)

  return result

def remove_words(text, word_list):
  for word in word_list:
    text = re.sub(r''+word+'', ' ', text)
  text = re.sub(r'\s+', ' ', text)
  return text

def from_raw_to_df(text_dict):
  nlp.max_length = 9000000
  text = text_dict['text']
  text = remove_words(text, text_dict['words to remove'])
  text = baseline_clean(text, capitals=text_dict['remove capitals'],
                        bracketed_fn=text_dict['bracketed fn'])
  text_nlp = nlp(text, disable=['ner'])
  text_df = pd.DataFrame(columns=['title', 'author', 'school', 'sentence_spacy'])
  text_df['sentence_spacy'] = list(text_nlp.sents)
  text_df['author'] = text_dict['author']
  text_df['title'] = text_dict['title']
  text_df['school'] = text_dict['school']
  text_df['sentence_str'] = text_df['sentence_spacy'].apply(lambda x: ''.join(list(str(x))))
  return text_df
  
def make_word_cloud(text, stopwords=stopwords.words('english')):
    cloud = wordcloud.WordCloud(width=2000, 
                            height=1100, 
                            background_color='#D1D1D1', 
                            max_words=30, 
                            stopwords=stopwords, 
                            color_func=lambda *args, **kwargs: (95,95,95)).generate(text)
    return cloud

def plot_pretty_cf(predictor, xtest, ytest, cmap='Greys', normalize='true', 
                   title=None, label_dict={}):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_confusion_matrix(predictor, xtest, ytest, cmap=cmap, normalize=normalize, ax=ax)
    ax.set_title(title, size='xx-large', pad=20, fontweight='bold')
    if label_dict != {}:
      ax.set_xticklabels([label_dict[int(x.get_text())] for x in ax.get_xticklabels()], rotation=35)
      ax.set_yticklabels([label_dict[int(x.get_text())] for x in ax.get_yticklabels()])
    else: 
      ax.set_xticklabels([str(x).replace('_', ' ').title()[12:-2] for x in ax.get_xticklabels()], rotation=35)
      ax.set_yticklabels([str(x).replace('_', ' ').title()[12:-2] for x in ax.get_yticklabels()])
    ax.set_xlabel('Predicted Label', size='x-large')
    ax.set_ylabel('True Label', size='x-large')
    plt.show()

def classify_text(to_classify, model, vectorizer, verbose=5):
    predictor_pipeline = make_pipeline(vectorizer, model) 
    class_names = ['analytic', 'continental', 'phenomenology', 'german_idealism', 'plato', 'aristotle', 'empiricism', 'rationalism']
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(to_classify, predictor_pipeline.predict_proba, num_features=8, labels=[0, 1, 2, 3, 4, 5, 6, 7])
    exp.show_in_notebook(text=True)

def make_w2v(series, stopwords=[], size=200, window=5, min_count=5, workers=-1, 
             epochs=20, lowercase=True, sg=0, seed=17, cbow_mean=1, alpha=0.025,
             sample=0.001, use_bigrams=True, threshold=10, bigram_min=5):
  # turn the series into a list, lower it, clean it
    sentences = [sentence for sentence in series]
    if lowercase:
      cleaned = []
      for sentence in sentences:
        cleaned_sentence = [word.lower() for word in sentence]
        cleaned_sentence = [word for word in sentence if word not in stopwords]
        cleaned.append(cleaned_sentence)
    else:
      cleaned = []
      for sentence in sentences:
        cleaned_sentence = [word for word in sentence]
        cleaned_sentence = [word for word in sentence if word not in stopwords]
        cleaned.append(cleaned_sentence)

  # incorporate bigrams
    if use_bigrams:
      bigram = Phrases(cleaned, min_count=bigram_min, threshold=threshold, delimiter=b' ')
      bigram_phraser = Phraser(bigram)
      tokens_list = []
      for sent in cleaned:
        tokens_ = bigram_phraser[sent]
        tokens_list.append(tokens_)
      cleaned = tokens_list
    else:
      cleaned = cleaned

  # build the model
    model = Word2Vec(cleaned, size=size, window=window, 
                     min_count=min_count, workers=workers, seed=seed, sg=sg,
                     cbow_mean=cbow_mean, alpha=alpha, sample=sample)
    model.train(series, total_examples=model.corpus_count, epochs=epochs)
    model_wv = model.wv
    
  # clear it to avoid unwanted transference
    del model

    return model_wv

def test_w2v(model, pairs):
  for (pos, neg) in pairs:
    math_result = model.most_similar(positive=pos, negative=neg)
    print(f'Positive - {pos}\tNegative - {neg}')
    [print(f"- {result[0]} ({round(result[1],5)})") for result in math_result[:5]]
    print()

# def space_words(str, 
#                 dict_path=pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt"),
#                 edit_distance=1,
#                 prefix_length=3):
#   sym_spell = SymSpell(max_dictionary_edit_distance=edit_distance, prefix_length=prefix_length)
#   sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)

#   input_term = str
#   result = sym_spell.word_segmentation(input_term)

#   return result.corrected_string


def test_w2v_pos_neg(model, pairs):
  for (pos, neg) in pairs:
    math_result = model.most_similar(positive=pos, negative=neg)
    print(f'Positive - {pos}\tNegative - {neg}')
    [print(f"- {result[0]} ({round(result[1],5)})") for result in math_result[:5]]
    print()

def plot_pretty_nn_cf(test_labels, predicted_labels, cmap='Greys', label_dict={}, title='CF'):
  fig, ax = plt.subplots(figsize=(8,8))
  cm = confusion_matrix(test_labels, predicted_labels)
  cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  ax = sns.heatmap(cm, cmap=cmap, annot=True, cbar=False, square=True)
  if label_dict != {}:
    ax.set_xticklabels([label_dict[int(x.get_text())] for x in ax.get_xticklabels()], rotation=35)
    ax.set_yticklabels([label_dict[int(x.get_text())] for x in ax.get_yticklabels()], rotation=35)
  ax.set_title(title, fontsize='x-large')
  plt.show()

def show_curves_nn(model):
  history_df = pd.DataFrame(model.history)
  fig, [ax1, ax2] = plt.subplots(1,2, figsize=(10,5))
  fig.suptitle('Accuracy and Loss', fontsize='xx-large')
  ax1.plot(history_df['accuracy'], label='train_accuracy', color='black')
  ax1.plot(history_df['val_accuracy'], label='val_accuracy', color='red')
  legend = ax1.legend()
  ax2.plot(history_df['loss'], label='train_loss', color='black')
  ax2.plot(history_df['val_loss'], label='val_loss', color='red')
  legend = ax2.legend()

def summarize_model(model, test, preds, labels, title):
  show_curves_nn(model)
  plot_pretty_nn_cf(test, preds, label_dict=labels, title=title)
  print(f'\n\n')
  print(f'\t\t\tCLASSIFICATION REPORT')
  print(classification_report(test, preds))

def nn_setup(x, y, max_length=450, tokenizer_name='nn_model.pkl'):  
  x_train, x_test, y_train, y_test = train_test_split(x,y)

  tokenizer = text.Tokenizer(num_words=28331)

  tokenizer.fit_on_texts(x_train)
  train_sequences = tokenizer.texts_to_sequences(x_train)
  test_sequences = tokenizer.texts_to_sequences(x_test)

  tokenizer_pkl = open(tokenizer_name, 'wb')
  pickle.dump(tokenizer, tokenizer_pkl)
  files.download(tokenizer_name)
  tokenizer_pkl.close()

  x_train_seq = sequence.pad_sequences(train_sequences, maxlen=max_length)
  x_test_seq = sequence.pad_sequences(test_sequences, maxlen=max_length)

  y_train_seq = to_categorical(y_train)
  y_test_seq = to_categorical(y_test)

  weights= compute_class_weight('balanced', np.unique(y_train), y_train)
  weights_dict = dict(zip( np.unique(y_train),weights))
  return tokenizer, x_train_seq, x_test_seq, y_train_seq, y_test_seq, weights_dict

def set_early_stop(monitor='val_accuracy',patience=3, restore_best_weights=False):
    args = locals()
    return EarlyStopping(**args)






