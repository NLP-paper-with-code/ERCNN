from torch.utils.data import Dataset, DataLoader, random_split
from torchnlp.encoders.text import WhitespaceEncoder
from gensim.models.keyedvectors import KeyedVectors

import pandas as pd
import numpy as np
import torch

import pickle
import os

class ToTensor(object):
  def __call__(self, sample, tokenizer, max_len):  
    X1, X2, Y = sample['sentence_one'], sample['sentence_two'], sample['target']

    X1 = str(X1)
    X2 = str(X2)

    X1, X2 = self._tokenize_and_padding(X1, X2, tokenizer, max_len)
    Y = torch.tensor(Y, dtype=torch.float)
    
    if torch.isnan(Y) > 0:
       raise RuntimeError(f'{sample} contain NaN')

    return {'sentence_one': X1, 'sentence_two': X2, 'target': Y}

  def _pad_sequence(self, sequence, max_len):
    sequence_difference = max_len - len(sequence)
    if sequence_difference > 0:
      pad_sequence = torch.zeros(sequence_difference)
      return torch.cat((sequence, pad_sequence), 0)
    else:
      return sequence[:max_len]

  def _tokenize_and_padding(self, X1, X2, tokenizer, max_len):
    tokenized_X1 = tokenizer.encode(X1)
    tokenized_X2 = tokenizer.encode(X2)

    padded_token_X1 = self._pad_sequence(tokenized_X1, max_len=max_len).type(torch.LongTensor)
    padded_token_X2 = self._pad_sequence(tokenized_X2, max_len=max_len).type(torch.LongTensor)

    return padded_token_X1, padded_token_X2

class SentenceMatchingDataset(Dataset):
  def __init__(self, data_path, max_len, transform=None):
    super(SentenceMatchingDataset, self).__init__()

    self.data = pd.read_csv(
      data_path, sep='\t',
      lineterminator='\n',
      header=None,
      names=['sentence_one', 'sentence_two', 'target'],
      index_col=False,
    )
    
    self.tokenizer = self._get_tokenizer(self.data['sentence_one'].values, self.data['sentence_two'].values)
    self.transform = transform
    self.max_len = max_len

  def __len__(self):
    return len(self.data)
  
  def _get_tokenizer(self, X1, X2):
    tokenizer_pickle_file = 'word2vec/data_tokenizer.pickle'

    if os.path.isfile(tokenizer_pickle_file):
      with open(tokenizer_pickle_file, 'rb') as handle:
        tokenizer = pickle.load(handle)
    else:
      X1 = list(map(str, list(X1)))
      X2 = list(map(str, list(X2)))
      
      # use all sentence to build dictionary
      tokenizer = WhitespaceEncoder(X1 + X2)

      with open(tokenizer_pickle_file, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return tokenizer

  def get_embedding(self):
    embed_pickle_file = 'word2vec/data_embed_matrix.pickle'

    # Load embedding matrix
    print('Loading embedding matrix...')
    embeddings_matrix = None

    if os.path.isfile(embed_pickle_file):
      with open(embed_pickle_file, 'rb') as handle:
        embeddings_matrix = pickle.load(handle)
    else:
      embed_model = KeyedVectors.load_word2vec_format('word2vec/cc.zh.300.vec', binary=False, encoding='utf8')
      embeddings_matrix = np.zeros((self.tokenizer.vocab_size, embed_model.vector_size))

      for index, token in enumerate(self.tokenizer.vocab):
        if token in embed_model:
          embeddings_matrix[index] = embed_model[token]

      with open(embed_pickle_file, 'wb') as handle:
        pickle.dump(embeddings_matrix, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return torch.Tensor(embeddings_matrix)

  def __getitem__(self, index):
    sentence_1 = self.data.iloc[index, 0]
    sentence_2 = self.data.iloc[index, 1]
    target = self.data.iloc[index, 2]

    sample = {'sentence_one': sentence_1, 'sentence_two': sentence_2, 'target': target}

    if self.transform:
      sample = self.transform(sample, self.tokenizer, self.max_len)

    return sample

if __name__ == '__main__':
  dataset = SentenceMatchingDataset('./preprocessed/data', 48, transform=ToTensor())
  
  # split dataset into [0.7, 0.15, 0.15] for train, valid and test set
  train_length, valid_length = int(len(dataset) * 0.7), int(len(dataset) * 0.15)
  lengths = [train_length, valid_length, len(dataset) - train_length - valid_length]

  train_dataset, valid_dataset, test_dataset = random_split(dataset, lengths)

  train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

  for batch_index, sample_batched in enumerate(train_dataloader):
    print(f'[batch-{batch_index}] {sample_batched}')