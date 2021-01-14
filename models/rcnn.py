import torch
import torch.nn as nn
import torch.nn.functional as F

def mean_max(x):
  return torch.mean(x, dim=1), torch.max(x, dim=1)[0]

class Inception1(nn.Module):
  def __init__(self, input_dim, conv_dim=64):
    super(Inception1, self).__init__()

    self.cnn = nn.Sequential(
      nn.Conv1d(input_dim, conv_dim, kernel_size=1),
      nn.ReLU(),
    )

  def forward(self, x):
    x = self.cnn(x)
    avg_pool, max_pool = mean_max(x)

    return torch.cat((avg_pool, max_pool), dim=1)

class Inception2(nn.Module):
  def __init__(self, input_dim, conv_dim=64):
    super(Inception2, self).__init__()

    self.cnn = nn.Sequential(
      nn.Conv1d(input_dim, conv_dim, kernel_size=1),
      nn.ReLU(),
      nn.Conv1d(conv_dim, conv_dim, kernel_size=2),
      nn.ReLU(),
    )

  def forward(self, x):
    x = self.cnn(x)
    avg_pool, max_pool = mean_max(x)

    return torch.cat((avg_pool, max_pool), dim=1)

class Inception3(nn.Module):
  def __init__(self, input_dim, conv_dim=64):
    super(Inception3, self).__init__()

    self.cnn = nn.Sequential(
      nn.Conv1d(input_dim, conv_dim, kernel_size=1),
      nn.ReLU(),
      nn.Conv1d(conv_dim, conv_dim, kernel_size=3),
      nn.ReLU(),
    )

  def forward(self, x):
    x = self.cnn(x)
    avg_pool, max_pool = mean_max(x)

    return torch.cat((avg_pool, max_pool), dim=1)

# TODO(jamfly): not quite sure the detail of this implementation
class FusionSubtract(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(FusionSubtract, self).__init__()
    self.dense = nn.Linear(input_dim, output_dim)

  def forward(self, input_1, input_2):
    result_sub = torch.sub(input_1, input_2)
    result_mul = torch.mul(result_sub, result_sub)

    out = self.dense(result_mul)

    return F.relu(out)

class FusionMultiply(nn.Module):
  def __init__(self, input_dim, output_dim):
    super(FusionMultiply, self).__init__()
    self.dense = nn.Linear(input_dim, output_dim)

  def forward(self, input_1, input_2):
    result = torch.mul(input_1, input_2)
    out = self.dense(result)

    return F.relu(out)

class EnhancedRCNN(nn.Module):
  def __init__(
    self,
    embeddings_matrix,
    max_len,
    number_of_class=1,
    transformer_dim=384,
    number_of_head=5,
    number_of_transformer_layer=2,
    dropout_rate=0.2,
    linear_size=384,
    conv_dim=64,
    freeze_embed=False,
  ):
    super(EnhancedRCNN, self).__init__()

    full_connect_out_dim = int(linear_size // 2)

    self.max_len = max_len
    self.embedding = nn.Embedding.from_pretrained(
      embeddings_matrix, freeze=freeze_embed,
    )
    self.batchnorm = nn.BatchNorm1d(max_len)
    self.dropout = nn.Dropout(p=dropout_rate)

    self.cnn1 = Inception1(self.embedding.embedding_dim)
    self.cnn2 = Inception2(self.embedding.embedding_dim)
    self.cnn3 = Inception3(self.embedding.embedding_dim)

    encoder_layer = nn.TransformerEncoderLayer(
      self.embedding.embedding_dim, nhead=5, dim_feedforward=transformer_dim, dropout=dropout_rate,
    )
    
    self.sentence1_transformer = nn.TransformerEncoder(
      encoder_layer, num_layers=number_of_transformer_layer,
    )
    self.sentence2_transformer = nn.TransformerEncoder(
      encoder_layer, num_layers=number_of_transformer_layer,
    )
    
    # after cnn with kernal size 1, 2, and 3, the output dimension will be (max_len, max_len - 1, max_len - 2)
    # sum them up, will be 3 (max_len - 1) since we concate max and avag pool, so it will be 3(max_len - 1) * 2
    # see more detail on the implementation of Inception
    self.full_connect_sub = FusionSubtract(3 * (max_len - 1) * 2 + self.embedding.embedding_dim * 2, full_connect_out_dim)
    self.full_connect_mul = FusionMultiply(3 * (max_len - 1) * 2 + self.embedding.embedding_dim * 2, full_connect_out_dim)
    self.dense = nn.Sequential(
      nn.Linear(full_connect_out_dim * 2, number_of_class),
      nn.Sigmoid()
    )
  
  def soft_align(self, input_1, input_2):
    attention = torch.bmm(input_1, input_2.permute(0, 2, 1))
    
    w_1 = F.softmax(attention, dim=1)
    w_2 = F.softmax(attention, dim=2).permute(0, 2, 1)

    input_1_align = torch.bmm(w_1, input_1)
    input_2_align = torch.bmm(w_2, input_2)

    return input_1_align, input_2_align

  def forward(self, sentence_1, sentence_2):
    sentence_1_embedding = self.batchnorm(self.embedding(sentence_1))
    sentence_2_embedding = self.batchnorm(self.embedding(sentence_2))

    x_1 = self.dropout(sentence_1_embedding)
    x_2 = self.dropout(sentence_1_embedding)
    
    # [batch_size, max_len, embedding_dim]
    sentence_1_representation = self.sentence1_transformer(x_1)
    sentence_2_representation = self.sentence1_transformer(x_2)

    # [batch_size, embedding_dim, max_len]
    sentence_1_encoded_permute = sentence_1_representation.permute(0, 2, 1)
    sentence_2_encoded_permute = sentence_2_representation.permute(0, 2, 1)

    # [batch_size, max_len, embedding_dim]
    sentence_1_alignment, sentence_2_alignment = self.soft_align(sentence_1_representation, sentence_2_representation)

    # [batch_size, embedding_dim]
    sentence_1_attention_mean, sentence_1_attention_max = mean_max(sentence_1_alignment)
    sentence_2_attention_mean, sentence_2_attention_max = mean_max(sentence_2_alignment)

    sentence_1_cnn = torch.cat(
      (
        self.cnn1(sentence_1_encoded_permute),
        self.cnn2(sentence_1_encoded_permute),
        self.cnn3(sentence_1_encoded_permute),
      ),
      dim=1,
    )

    sentence_2_cnn = torch.cat(
      (
        self.cnn1(sentence_2_encoded_permute),
        self.cnn2(sentence_2_encoded_permute),
        self.cnn3(sentence_2_encoded_permute),
      ),
      dim=1,
    )

    sentence_1 = torch.cat(
      (sentence_1_attention_mean, sentence_1_cnn, sentence_1_attention_max), dim=1,
    )
    
    sentence_2 = torch.cat(
      (sentence_2_attention_mean, sentence_2_cnn, sentence_2_attention_max), dim=1,
    )

    result_sub = self.full_connect_sub(sentence_1, sentence_2)
    result_mul = self.full_connect_mul(sentence_1, sentence_2)
    
    result = torch.cat((result_sub, result_mul), dim=1)
    
    return self.dense(result)