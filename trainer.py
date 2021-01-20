from sklearn.metrics import f1_score, confusion_matrix, classification_report
from tqdm import tqdm

import torch.nn as nn
import numpy as np
import torch

from typing import Tuple

import time

def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  
  return elapsed_mins, elapsed_secs

class Trainer(object):
  def __init__(
    self,
    optimizer: torch.optim.Optimizer,
    criterion,
    args,
    device,
  ):
    self.optimizer = optimizer
    self.criterion = criterion
    self.args = args
    self.device = device

  def train_step(
    self,
    model: nn.Module,
    train_dataloader,
    epoch: int) -> Tuple[float]:

    model.train()
    epoch_loss = 0
    correctness = 0

    steps = tqdm(enumerate(train_dataloader), total=len(train_dataloader), leave=False)
    predictions, targets = [], []
    
    step_number = len(steps) * epoch

    for step, sample_batched in steps:
      step_number += 1
      input_1, input_2, target = sample_batched['sentence_one'], sample_batched['sentence_two'], sample_batched['target']
      input_1_mask, input_2_mask = sample_batched['sentence_one_mask'], sample_batched['sentence_two_mask']
    
      input_1, input_2, target = input_1.to(self.device), input_2.to(self.device), target.to(self.device)
      input_1_mask, input_2_mask = sample_batched['sentence_one_mask'].to(self.device), sample_batched['sentence_two_mask'].to(self.device)

      self.optimizer.zero_grad()

      output = model(input_1, input_2, input_1_mask, input_2_mask)
      loss = self.criterion(output, target.view_as(output))
    
      loss.backward()
      self.optimizer.step()
      
      epoch_loss += loss.item()

      prediction = output.round()

      # F1 score
      correctness += prediction.eq(target.view_as(prediction)).sum().item()
      predictions.extend(
        prediction.view(len(prediction)).tolist()
      )
      targets.extend(
        target.view(len(target)).tolist()
      )

      steps.set_description(f'Epoch [{epoch+1:02.2f}/{self.args.epoch}][train] steps: {step_number}')
      steps.set_postfix(loss=epoch_loss / (step + 1))
    
    accuracy = correctness / len(train_dataloader.dataset)
    f1 = f1_score(targets, predictions, average='macro', labels=np.unique(predictions))

    return epoch_loss / len(train_dataloader), loss.item(), accuracy, f1

  def evaluate_step(
    self,
    model: nn.Module,
    evaluate_dataloader,
    epoch:int=None) -> float:

    model.eval()
    epoch_loss = 0
    correctness = 0

    steps = tqdm(enumerate(evaluate_dataloader), total=len(evaluate_dataloader), leave=False)
    predictions, targets = [], []

    with torch.no_grad():
      for step, sample_batched in steps:
        input_1, input_2, target = sample_batched['sentence_one'], sample_batched['sentence_two'], sample_batched['target']
        input_1_mask, input_2_mask = sample_batched['sentence_one_mask'], sample_batched['sentence_two_mask']
    
        input_1, input_2, target = input_1.to(self.device), input_2.to(self.device), target.to(self.device)
        input_1_mask, input_2_mask = sample_batched['sentence_one_mask'].to(self.device), sample_batched['sentence_two_mask'].to(self.device)

        output = model(input_1, input_2, input_1_mask, input_2_mask)
        epoch_loss += self.criterion(output, target.view_as(output)).item()

        prediction = output.round()

        # F1 score
        correctness += prediction.eq(target.view_as(prediction)).sum().item()
        predictions.extend(
          prediction.view(len(prediction)).tolist()
        )
        targets.extend(
          target.view(len(target)).tolist()
        )

        if epoch is not None:
          steps.set_description(f'[{epoch+1:02.2f}/{self.args.epoch}][evaluate]')
          steps.set_postfix(loss=epoch_loss / (step + 1))
    
    evaluate_loss = epoch_loss / len(evaluate_dataloader)
    accuracy = correctness / len(evaluate_dataloader.dataset)
    
    f1 = f1_score(targets, predictions, average='macro', labels=np.unique(predictions))
    confusion = confusion_matrix(targets, predictions)
    report = classification_report(targets, predictions)

    return evaluate_loss, accuracy, f1, confusion, report
  
  def train(
    self,
    model: nn.Module,
    train_dataloader, 
    valid_dataloader,
    writer):

    best_valid_loss = float('inf')

    for epoch in range(self.args.epoch):
      start_time = time.time()
      
      train_loss, loss, train_accuracy, train_f1 = self.train_step(model=model, epoch=epoch, train_dataloader=train_dataloader)
      valid_loss, accuracy, f1, _, _ = self.evaluate_step(model=model, epoch=epoch, evaluate_dataloader=valid_dataloader)
      
      end_time = time.time()
      epoch_mins, epoch_secs = epoch_time(start_time, end_time)

      writer.add_scalar('data/train/loss', loss, global_step=epoch + 1)
      writer.add_scalar('data/train/accuracy', train_accuracy, global_step=epoch + 1)
      writer.add_scalar('data/train/f1', train_f1, global_step=epoch + 1)
      writer.add_scalar('data/valid/loss', valid_loss, global_step=epoch + 1)
      writer.add_scalar('data/valid/accuracy', accuracy, global_step=epoch + 1)
      writer.add_scalar('data/valid/f1', f1, global_step=epoch + 1)

      checkpoint = {'state_dict': model.state_dict(), 'optimizer': self.optimizer.state_dict()}
      
      if (epoch + 1) % 10 == 0:
        torch.save(checkpoint, f'checkpoints/checkpoint_{epoch+1}.pt')
    
      if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), './checkpoints/checkpoint_best.pt')

      print(f'''Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s ''' +
        f'''| train_loss: {train_loss:02.2f} | train_acc: {train_accuracy:02.2f} | train_f1: {train_f1:02.2f} | ''' + 
        f'''valid_loss: {valid_loss:02.2f} | valid_acc: {accuracy:02.2f} | valid_f1: {f1:02.2f}\n''')

      current_time = time.strftime('%H:%M:%S', time.localtime())
      
      if epoch == 0:
        current_time = f'starting time: \n{current_time}'
      elif epoch == self.args.epoch:
        current_time = f'ending time: \n{current_time}'

      print('Local Time: ', current_time)

  def test(self, model: nn.Module, test_dataloader):
    _, _, _, confusion, report = self.evaluate_step(model=model, evaluate_dataloader=test_dataloader)
    print(f'Confusion Matrix:\n{confusion}')
    print(f'Classification Report:\n{report}')