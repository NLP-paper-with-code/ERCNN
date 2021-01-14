import argparse

from pathlib import Path
from typing import List

import jieba

def get_args():
  parser = argparse.ArgumentParser(description='preprocess the dataset')
  parser.add_argument('--data-path', help='dataset to be processed', type=str, default='./data')

  return parser.parse_args()

def get_stopwords(path: str) -> List[str]:
  """read and give the stop words list from given path"""
  stopwords = []
  with open(path, 'r', encoding='utf-8') as stopwords_file:
    lines = stopwords_file.readlines()
    stopwords = [line.strip() for line in lines]

  return stopwords

def parse_sentence_to_char(sentence: str, stopwords: List[str]) -> List[str]:
  """read sentence, remove stopword, and make it character"""
  characters = []
  
  for char in sentence:
    if char not in stopwords and char != ' ':
      characters.append(char)

  return characters

def parse_sentence_to_word(sentence: str, stopwords: List[str]) -> List[str]:
  """read sentence, remove stopwords, and tokenize it into words"""
  words = []

  for word in jieba.cut(sentence):
    if word not in stopwords and word != ' ':
      words.append(word)

  return words

def cleanup_corpus(corpus_lines, split_symbol, start_index, number_of_columns, stopwords):
  """clean up given corpus"""
  sentences = []
   
  for epidemic_line in corpus_lines:
    raw_data = epidemic_line.split(split_symbol)
    sentence_one_tokens = parse_sentence_to_word(sentence=raw_data[start_index], stopwords=stopwords)
    sentence_two_tokens = parse_sentence_to_word(sentence=raw_data[start_index + 1], stopwords=stopwords)

    if len(raw_data) != number_of_columns or len(sentence_one_tokens) == 0 or len(sentence_two_tokens) == 0:
     continue
    
    sentence_one = ' '.join(sentence_one_tokens)
    sentence_two = ' '.join(sentence_two_tokens)
    label = raw_data[start_index + 2]

    processed_sentence = '\t'.join([sentence_one, sentence_two, label]).strip()
    sentences.append(processed_sentence)
        
  return sentences
        

def cleanup_data(data_path: str):
  """clean up data to desired format"""
  stopwords = get_stopwords(path=f'{data_path}/stopwords.txt')
  parse_strategy = None

  with open(f'{data_path}/dictionary', 'r', encoding='utf-8') as dictionary, \
    open(f'{data_path}/not_word', 'r', encoding='utf-8') as not_word:

    dictionary_lines = dictionary.readlines()
    not_word_lines = not_word.readlines()

    for dictionary_line in dictionary_lines:
      dictionary_line = dictionary_line.strip()
      jieba.add_word(dictionary_line)

    for not_word_line in not_word_lines:
      not_word_line = not_word_line.strip()
      jieba.del_word(not_word_line)

  with open(f'{data_path}/ant_train', 'r', encoding='utf-8') as ant_train, \
    open(f'{data_path}/ant_train_add', 'r', encoding='utf-8') as ant_train_add, \
    open(f'{data_path}/epidemic_dev.csv', 'r', encoding='utf-8') as epidemic_dev, \
    open(f'{data_path}/epidemic_train.csv', 'r', encoding='utf-8') as epidemic_train, \
    open(f'{data_path}/icqmc_train.txt', 'r', encoding='utf-8') as icqmc_train, \
    open(f'{data_path}/icqmc_dev.txt', 'r', encoding='utf-8') as icqmc_dev, \
    open(f'{data_path}/icqmc_test.txt', 'r', encoding='utf-8') as icqmc_test, \
    open(f'{data_path}/simtrain_to05sts.txt', encoding='utf-8') as simtrain:

    ant_train_lines = ant_train.readlines() + ant_train_add.readlines()
    epidemic_lines = epidemic_dev.readlines()[1:] + epidemic_train.readlines()[1:]
    icqmc_lines = icqmc_train.readlines()[1:] + icqmc_dev.readlines()[1:] + icqmc_test.readlines()[1:]
    
    sentences = []
    
    sentences += cleanup_corpus(epidemic_lines, ',', 2, 5, stopwords)
    sentences += cleanup_corpus(ant_train_lines, '\t', 1, 4, stopwords)
    sentences += cleanup_corpus(icqmc_lines, '\t', 0, 3, stopwords)

    with open(f'./preprocessed/data', 'a+', encoding='utf-8') as ant_file:
      for sentence in sentences:
        ant_file.write(f'{sentence}\n')

if __name__ == '__main__':
  args = get_args()

  Path('preprocessed').mkdir(parents=True, exist_ok=True)

  cleanup_data(data_path=args.data_path)