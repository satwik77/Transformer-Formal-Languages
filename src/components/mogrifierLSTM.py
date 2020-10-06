import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import pdb

def clones(module, N):
	"Produce N identical layers."
	return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MogrifyLayer(nn.Module):
	"Single Iteration of Mogrify"

	def __init__(self, hidden_size, input_size):
		super(MogrifyLayer, self).__init__()
		self.sigmoid = nn.Sigmoid()
		self.Qw = nn.Linear(hidden_size, input_size)
		self.Rw = nn.Linear(input_size, hidden_size)

	def forward(self, x, h):
		h_q  = self.Qw(h)
		x_i = (2*self.sigmoid(h_q))*x
		x_r = self.Rw(x_i)
		h_i = (2*self.sigmoid(x_r))*h

		return x_i, h_i

class Mogrify(nn.Module):
	"R iterations of mogrifying"

	def __init__(self, layer, R):
		super(Mogrify, self).__init__()
		self.layers = clones(layer, R)

	def forward(self, x, h):
		for layer in self.layers:
			x, h = layer(x,h)
		return x, h

class MogrifierLSTMModel(nn.Module):
	"""Container module with an encoder, a recurrent module, and a decoder."""

	def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, r=2):
		super(MogrifierLSTMModel, self).__init__()
		self.drop = nn.Dropout(dropout)
		self.encoder = nn.Embedding(ntoken, ninp)
		'''
		if rnn_type in ['LSTM', 'GRU']:
			self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
		else:
			try:
				nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
			except KeyError:
				raise ValueError( """An invalid option for `--model` was supplied,
								 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
			self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
		'''
		self.rnn = getattr(nn, 'LSTM')(ninp, nhid, nlayers, dropout=dropout)
		self.decoder = nn.Linear(nhid, ntoken)

		self.mogrify = Mogrify(MogrifyLayer(nhid, ninp), r)

		# Optionally tie weights as in:
		# "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
		# https://arxiv.org/abs/1608.05859
		# and
		# "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
		# https://arxiv.org/abs/1611.01462
		if tie_weights:
			if nhid != ninp:
				raise ValueError('When using the tied flag, nhid must be equal to emsize')
			self.decoder.weight = self.encoder.weight

		self.init_weights()

		self.rnn_type = rnn_type
		self.nhid = nhid
		self.nlayers = nlayers

	def init_weights(self):
		initrange = 0.1
		self.encoder.weight.data.uniform_(-initrange, initrange)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)

	def forward(self, input, hidden):
		inp_len = input.size(0)
		decoded_list = []
		for t in range(inp_len):
			input_t = input[t].unsqueeze(0)
			h_t, c_t = hidden
			emb_t = self.drop(self.encoder(input_t))
			emb_t, h_t = self.mogrify(emb_t, h_t)
			hidden = (h_t, c_t)
			output_t, hidden = self.rnn(emb_t, hidden)
			output_t = self.drop(output_t)
			decoded = self.decoder(output_t)
			decoded_list.append(decoded)
		
		decoded = torch.cat(decoded_list, dim = 0)
		return decoded, hidden

	def init_hidden(self, bsz):
		weight = next(self.parameters())
		if self.rnn_type == 'LSTM':
			return (weight.new_zeros(self.nlayers, bsz, self.nhid),
					weight.new_zeros(self.nlayers, bsz, self.nhid))
		else:
			return weight.new_zeros(self.nlayers, bsz, self.nhid)