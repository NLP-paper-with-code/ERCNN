import torch.nn.functional as F
import torch.nn as nn
import torch

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

class FusionLayer(nn.Module):
  """
  vector based fusion
  m(x, y) = W([x, y, x * y, x - y]) + b
  g(x, y) = w([x, y, x * y, x - y]) + b
  :returns g(x, y) * m(x, y) + (1 - g(x, y)) * x
  """

  def __init__(self, input_dim):
    super(FusionLayer, self).__init__()
    self.linear_f = nn.Linear(input_dim * 4, input_dim, bias=True)
    self.linear_g = nn.Linear(input_dim * 4, input_dim, bias=True)
    self.tanh = nn.Tanh()
    self.sigmoid = nn.Sigmoid()

  def forward(self, x, y):
    z = torch.cat([x, y, x * y, x - y], dim=1)
    gated = self.sigmoid(self.linear_g(z))
    fusion = self.tanh(self.linear_f(z))
    return gated * fusion + (1 - gated) * x

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
    # interaction model will be [x, y, x - y, x * y], so it ended up being embedding_dim * 4
    input_dim = 3 * (max_len - 1) * 2 + self.embedding.embedding_dim * 4 * 2
    self.fusion_layer = FusionLayer(input_dim)
    self.out = nn.Sequential(
      nn.Linear(input_dim * 2, number_of_class),
      nn.Sigmoid()
    )
  
  def soft_align(self, input_1, input_2):
    attention = torch.bmm(input_1, input_2.permute(0, 2, 1))
    
    w_1 = F.softmax(attention, dim=1)
    w_2 = F.softmax(attention, dim=2).permute(0, 2, 1)

    input_1_align = torch.bmm(w_1, input_1)
    input_2_align = torch.bmm(w_2, input_2)

    return input_1_align, input_2_align

  def forward(self, sentence_1, sentence_2, sentence_1_mask, sentence_2_mask):
    sentence_1_embedding = self.batchnorm(self.embedding(sentence_1))
    sentence_2_embedding = self.batchnorm(self.embedding(sentence_2))

    X_1 = self.dropout(sentence_1_embedding).transpose(1, 0)
    X_2 = self.dropout(sentence_2_embedding).transpose(1, 0)

    # [max_len, batch_size, embedding_dim]
    sentence_1_representation = self.sentence1_transformer(X_1, src_key_padding_mask=sentence_1_mask)
    sentence_2_representation = self.sentence1_transformer(X_2, src_key_padding_mask=sentence_2_mask)

    # [max_len, batch_size, embedding_dim] -> [batch_size, embedding_dim, max_len]
    sentence_1_encoded_permute = sentence_1_representation.permute(1, 2, 0)
    sentence_2_encoded_permute = sentence_2_representation.permute(1, 2, 0)
    
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

    # [max_len, batch_size, embedding_dim] -> [batch_size, max_len, embedding_dim]
    sentence_1_representation = sentence_1_representation.transpose(1, 0)
    sentence_2_representation = sentence_2_representation.transpose(1, 0)

    sentence_1_alignment, sentence_2_alignment = self.soft_align(sentence_1_representation, sentence_2_representation)

    # [batch_size, max_len, 4 * embedding_dim]
    sentence_1_interaction = torch.cat((
      sentence_1_representation,
      sentence_1_alignment,
      sentence_1_representation - sentence_1_alignment,
      sentence_1_representation * sentence_1_alignment
    ), dim=2)

    sentence_2_interaction = torch.cat((
      sentence_2_representation,
      sentence_2_alignment,
      sentence_2_representation - sentence_2_alignment,
      sentence_2_representation * sentence_2_alignment
    ), dim=2)

    # [batch_size, embedding_dim * 4]
    sentence_1_interaction_mean, sentence_1_interaction_max = mean_max(sentence_1_interaction)
    sentence_2_interaction_mean, sentence_2_interaction_max = mean_max(sentence_2_interaction)

    sentence_1 = torch.cat(
      (sentence_1_interaction_mean, sentence_1_cnn, sentence_1_interaction_max), dim=1,
    )
    
    sentence_2 = torch.cat(
      (sentence_2_interaction_mean, sentence_2_cnn, sentence_2_interaction_max), dim=1,
    )

    fused_sentence_1 = self.fusion_layer(sentence_1, sentence_2)
    fused_sentence_2 = self.fusion_layer(sentence_2, sentence_1)
    
    result = torch.cat((fused_sentence_1, fused_sentence_2), dim=1)
    
    return self.out(result)