from torch.utils.data import Dataset, DataLoader, random_split
from torchnlp.encoders.text import WhitespaceEncoder
from gensim.models.keyedvectors import KeyedVectors

import torch.nn.functional as F
import pandas as pd
import numpy as np
import torch

import pickle
import os

class ToTensor(object):
  def __call__(self, sample, tokenizer, max_len):  
    self.max_len = max_len
    
    X1, X2, Y = sample['sentence_one'], sample['sentence_two'], sample['target']

    X1 = str(X1)
    X2 = str(X2)

    X1, X2 = self._tokenize(X1, X2, tokenizer)
    Y = torch.tensor(Y, dtype=torch.float)
    
    if torch.isnan(Y) > 0:
       raise RuntimeError(f'{sample} contain NaN')

    X1_pad = F.pad(X1, (0, max_len - len(X1)), value=0)
    X2_pad = F.pad(X2, (0, max_len - len(X2)), value=0)
  
    # mask padding for transformer
    X1_mask = (X1_pad == 0)
    X2_mask = (X2_pad == 0)

    return {
      'sentence_one': X1_pad,
      'sentence_two': X2_pad,
      'sentence_one_mask': X1_mask,
      'sentence_two_mask': X2_mask,
      'target': Y,
    }

  def handle_max_len(self, sentence):
    if len(sentence) <= self.max_len:
      return sentence
    else:
      return sentence[:self.max_len]

  def _tokenize(self, X1, X2, tokenizer):
    tokenized_X1 = tokenizer.encode(X1).type(torch.LongTensor)
    tokenized_X2 = tokenizer.encode(X2).type(torch.LongTensor)

    tokenized_X1 = self.handle_max_len(tokenized_X1)
    tokenized_X2 = self.handle_max_len(tokenized_X2)

    return tokenized_X1, tokenized_X2

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

# def pad_collate(batch):
#   sentence_one, sentence_two, target = [], [], []

#   for sample in batch:
#     sentence_one.append(sample['sentence_one'])
#     sentence_two.append(sample['sentence_two'])
#     target.append(sample['target'])

#   sentence_one_pad = torch.stack([F.pad(sentence, (0, 48 - len(sentence)), value=0) for sentence in sentence_one])
#   sentence_two_pad = torch.stack([F.pad(sentence, (0, 48 - len(sentence)), value=0) for sentence in sentence_two])
  
#   # mask padding for transformer
#   sentence_one_mask = (sentence_one_pad == 0)
#   sentence_two_mask = (sentence_two_pad == 0)

#   return {
#     'sentence_one': sentence_one_pad,
#     'sentence_two': sentence_two_pad,
#     'sentence_one_mask': sentence_one_mask,
#     'sentence_two_mask': sentence_one_mask,
#     'target': torch.FloatTensor(target),
#   }

if __name__ == '__main__':
  dataset = SentenceMatchingDataset('./preprocessed/data', 48, transform=ToTensor())
  
  # split dataset into [0.8, 0.1, 0.1] for train, valid and test set
  train_length, valid_length = int(len(dataset) * 0.8), int(len(dataset) * 0.1)
  lengths = [train_length, valid_length, len(dataset) - train_length - valid_length]

  train_dataset, valid_dataset, test_dataset = random_split(dataset, lengths)

  train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

  for batch_index, sample_batched in enumerate(train_dataloader):
    print(f'[batch-{batch_index}] {sample_batched}')