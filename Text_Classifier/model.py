import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.autograd import Variable

import torch
import torch.nn as nn
import torch.nn.functional as F

class CBOW(nn.Module):

	def __init__(self, **kwargs):
		super(CBOW, self).__init__()
        
		self.MODEL = kwargs["MODEL"]
		self.BATCH_SIZE = kwargs["BATCH_SIZE"]
		self.WORD_DIM = kwargs["WORD_DIM"]
		self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
		self.CLASS_SIZE = kwargs["CLASS_SIZE"]
		self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
        
		self.embeddings = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
		self.WV_MATRIX = kwargs["WV_MATRIX"]
		print("WV matrix size: " + str(self.WV_MATRIX.shape))
		self.embeddings.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
		if self.MODEL == "static":
			self.embeddings.weight.requires_grad = False
		self.linear1 = nn.Linear(self.WORD_DIM, 128)
		self.linear2 = nn.Linear(128, self.CLASS_SIZE)     

	def forward(self, inputs):
		#embeds = terch.tensor(50,300)
		#for inp in self.embeddings(inputs):
		#    embeds.append(sum(in))
		#embeds = sum(self.embeddings(inputs)).view((1, -1))
		embeds = torch.sum(self.embeddings(inputs), dim=1)
		#print("Inputs size: " + str(self.embeddings(inputs).size()))
		#print("Embeds size: " + str(embeds.size()))
		out = F.relu(self.linear1(embeds))
		out = F.dropout(out, p=self.DROPOUT_PROB, training=self.training)
		out = self.linear2(out)
		log_probs = F.log_softmax(out, dim=1)
		return log_probs

class CNNClassifier_2(nn.Module):
	def __init__(self, **kwargs):
		super(CNN, self).__init__()

		self.MODEL = kwargs["MODEL"]
		self.BATCH_SIZE = kwargs["BATCH_SIZE"]
		self.MAX_SENT_LEN = kwargs["MAX_SENT_LEN"]
		self.WORD_DIM = kwargs["WORD_DIM"]
		self.VOCAB_SIZE = kwargs["VOCAB_SIZE"]
		self.CLASS_SIZE = kwargs["CLASS_SIZE"]
		self.FILTERS = kwargs["FILTERS"]
		self.FILTER_NUM = kwargs["FILTER_NUM"]
		self.DROPOUT_PROB = kwargs["DROPOUT_PROB"]
		self.IN_CHANNEL = 1

		assert (len(self.FILTERS) == len(self.FILTER_NUM))

		# one for UNK and one for zero padding
		self.embedding = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
		if self.MODEL == "static" or self.MODEL == "non-static" or self.MODEL == "multichannel":
			self.WV_MATRIX = kwargs["WV_MATRIX"]
			self.embedding.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
			if self.MODEL == "static":
				self.embedding.weight.requires_grad = False
			elif self.MODEL == "multichannel":
				self.embedding2 = nn.Embedding(self.VOCAB_SIZE + 2, self.WORD_DIM, padding_idx=self.VOCAB_SIZE + 1)
				self.embedding2.weight.data.copy_(torch.from_numpy(self.WV_MATRIX))
				self.embedding2.weight.requires_grad = False
				self.IN_CHANNEL = 2

		for i in range(len(self.FILTERS)):
			conv = nn.Conv1d(self.IN_CHANNEL, self.FILTER_NUM[i], self.WORD_DIM * self.FILTERS[i], stride=self.WORD_DIM)
			setattr(self, f'conv_{i}', conv)

		self.fc = nn.Linear(sum(self.FILTER_NUM), self.CLASS_SIZE)

	def get_conv(self, i):
		return getattr(self, f'conv_{i}')

	def forward(self, inp):
		x = self.embedding(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
		if self.MODEL == "multichannel":
			x2 = self.embedding2(inp).view(-1, 1, self.WORD_DIM * self.MAX_SENT_LEN)
			x = torch.cat((x, x2), 1)

			conv_results = [
            F.max_pool1d(F.relu(self.get_conv(i)(x)), self.MAX_SENT_LEN - self.FILTERS[i] + 1)
                .view(-1, self.FILTER_NUM[i])
            for i in range(len(self.FILTERS))]

		x = torch.cat(conv_results, 1)
		x = F.dropout(x, p=self.DROPOUT_PROB, training=self.training)
		x = self.fc(x)

		return x

class CNNClassifier(nn.Module):
	def __init__(self, in_channels, out_channels, voca_size, embed_dim, num_classes, kernel_sizes, dropout_p, embedding_weight):
		super(CNNClassifier, self).__init__()
		self.embedding_weight = embedding_weight
		self.embedding = nn.Embedding(voca_size, embed_dim)
		self.convs = nn.ModuleList([nn.Conv2d(in_channels, out_channels,(k_size, embed_dim)) for k_size in kernel_sizes])

		self.dropout = nn.Dropout(dropout_p) 
		self.fc = nn.Linear(len(kernel_sizes) * out_channels , num_classes)
		self.init_weights()

	def init_weights(self):
		self.embedding.weight = nn.Parameter(self.embedding_weight)
		self.fc.bias.data.normal_(0, 0.01)
		self.fc.weight.data.normal_(0, 0.01)

		for layer in self.convs:
			nn.init.xavier_normal_(layer.weight)


	def forward(self, x):
		
		"""
		parameters of x:
		                N: batch_size 
		                C: num of in_channels
		                W: len of text 
		                D: num of embed_dim 
		"""

		x = self.embedding(x) # (N,W,D)
		#x= Variable(x)
		x = x.unsqueeze(1) #(N,1,W,D)
		x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
		x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
		x = torch.cat(x,1)

		x = self.dropout(x)
		out = self.fc(x)

		return out 


class RNNClassifier(nn.Module):
	def __init__(self, voca_size, embed_size, hidden_size, num_layers, num_classes, embedding_weight, cuda):
		super(RNNClassifier,self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.cuda = cuda
		self.embedding_weight = embedding_weight
		self.embed = nn.Embedding(voca_size, embed_size)
		self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first= True)
		self.dropout = nn.Dropout(0.5)
		self.fc = nn.Linear(hidden_size, num_classes)
		self.init_weights()


	def init_weights(self):

		self.embed.weight = nn.Parameter(self.embedding_weight)
		self.fc.bias.data.normal_(0, 0.01)
		self.fc.weight.data.normal_(0, 0.01)
	

	def forward(self, x):

		x = self.embed(x)

		# Set initial states  & GPU run
		if self.cuda:
			h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
			c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda()
		else:
			h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))
			c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size))

		h0 = (nn.init.xavier_normal_(h0))
		c0 = (nn.init.xavier_normal_(c0))

		# Forward 
		out, _ = self.lstm(x, (h0,c0))
        # Decode hidden state of last time step/ many to one 
		out = self.dropout(out)
		out = self.fc(out[:, -1, :])
		return out

		
class RCNN_Classifier(nn.Module):
	def __init__(self, voca_size, embed_size, hidden_size, sm_hidden_size,  num_layers, num_classes, embedding_weight, cuda):
		super(RCNN_Classifier,self).__init__()
		self.hidden_size = hidden_size
		self.sm_hidden_size = sm_hidden_size
		self.num_layers = num_layers
		self.iscuda = cuda
		self.embedding_weight = embedding_weight
		self.embed = nn.Embedding(voca_size, embed_size)
		self.bi_lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first= True, bidirectional = True)
		self.sm_fc = nn.Linear(embed_size + hidden_size*2 , sm_hidden_size)
		self.fc = nn.Linear(sm_hidden_size, num_classes)
		self.init_weights()

	def init_weights(self):

		self.embed.weight = nn.Parameter(self.embedding_weight)

	def forward(self, x):
		x = self.embed(x)

		#Set inital states & GPU run
		if self.iscuda:
			h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)).cuda()
			c0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)).cuda() # *2 for bidirectional
		else:
			h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size))
			c0 = Variable(torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)) # *2 for bidirectional

		h0 = (nn.init.xavier_normal_(h0))
		c0 = (nn.init.xavier_normal_(c0))

		#Forward 
		self.bi_lstm.flatten_parameters()
		lstm_out, _ = self.bi_lstm(x, (h0, c0))
		out = torch.cat((lstm_out, x), 2)  # eq. 3 

		y2 = F.tanh(self.sm_fc(out)) # semantic layer eq.4  y2
		y2 = y2.unsqueeze(1)

		y3 = F.max_pool2d(y2, (y2.size(2),1)).squeeze() # eq.5  y3

		y4 = self.fc(y3) # eq.6

		final_out = F.softmax(y4, dim=-1) # eq.7

		return final_out
		