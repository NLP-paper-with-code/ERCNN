from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader, random_split
from tensorboardX import SummaryWriter

import torch.nn.functional as F
import torch.optim as optim
import torch

from dataloader import SentenceMatchingDataset, ToTensor

from models.rcnn import EnhancedRCNN
from trainer import Trainer

import argparse
import os

def get_parser():
  parser = argparse.ArgumentParser(description='Enhabced RCNN on Sentence Similarity')

  parser.add_argument('--dataset', type=str, metavar='path', default='preprocessed/data', help='dataset path')
  parser.add_argument('--max-len', type=int, metavar='N', default=48, help='max length of a sentence could be, will pad 0 if it is shorter than N')
  parser.add_argument('--fix-embed', action='store_true', default=False, help='if freeze the embedding parameters')
  parser.add_argument('--batch-size', type=int, default=256, metavar='N', help='input batch size for training (default: 256)')
  parser.add_argument('--epoch', type=int, default=10, metavar='N', help='number of epochs to train')
  parser.add_argument('--lr', type=float, default=0.00001, metavar='N', help='learning rate (default: 0.00001)')
  parser.add_argument('--beta1', type=float, default=0.9, metavar='N', help='beta 1 for Adam optimizer (default: 0.9)')
  parser.add_argument('--beta2', type=float, default=0.999, metavar='N', help='beta 2 for Adam optimizer (default: 0.999)')
  parser.add_argument('--epsilon', type=float, default=1e-08, metavar='N', help='epsilon for Adam optimizer (default: 1e-08)')
  parser.add_argument('--decay', type=float, default=0.01, metavar='N', help='decay for Adam optimizer (default: 0.01)')
  parser.add_argument('--seed', type=int, default=16, metavar='N', help='random seed (default: 16)')
  parser.add_argument('--logdir', type=str, default='rcnn', metavar='path', help='set log directory (default: ./log)')
  
  return parser

def get_model_parameters(model, trainable_only=False):
  if trainable_only:
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  else:
    pytorch_total_params = sum(p.numel() for p in model.parameters())
  
  return pytorch_total_params


if __name__ == '__main__':
  args = get_parser().parse_args()
  
  os.makedirs('./checkpoints', exist_ok=True)
  os.makedirs('./tensorboards', exist_ok=True)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  torch.backends.cudnn.deterministic = True

  dataset = SentenceMatchingDataset(args.dataset, args.max_len, transform=ToTensor())

  embeddings_matrix = dataset.get_embedding()

  # split dataset into [0.8, 0.1, 0.1] for train, valid and test set
  train_length, valid_length = int(len(dataset) * 0.8), int(len(dataset) * 0.1)
  lengths = [train_length, valid_length, len(dataset) - train_length - valid_length]

  train_dataset, valid_dataset, test_dataset = random_split(dataset, lengths)
#   train_classes = Counter([sample['target'] for sample in train_dataset])
#   train_sample_weights = [5 if sample['target'] == 1 else 1 for sample in train_dataset]
  
#   train_sampler = WeightedRandomSampler(train_sample_weights, num_samples=len(train_dataset))

#   train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
  train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
  valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
  test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

  model = EnhancedRCNN(
    embeddings_matrix=embeddings_matrix,
    max_len=args.max_len,
    freeze_embed=args.fix_embed,
  ).to(device)

  optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), eps=args.epsilon, weight_decay=args.decay)
  cross_entropy = F.binary_cross_entropy

  total_parameters = get_model_parameters(model=model, trainable_only=True)

  print(f'The model has {total_parameters} trainable parameters')
  print(model)

  writer = SummaryWriter(f'tensorboards/{args.logdir}')

  trainer = Trainer(
    optimizer=optimizer,
    criterion=cross_entropy,
    args=args,
    device=device,
  )

  trainer.train(
    model=model,
    train_dataloader=train_dataloader,
    valid_dataloader=valid_dataloader,
    writer=writer,
  )

  model.load_state_dict(torch.load('./checkpoints/checkpoint_best.pt'))
  trainer.test(model=model, test_dataloader=test_dataloader)