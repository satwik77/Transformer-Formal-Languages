import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNNModel(nn.Module):
	"""Container module with an encoder, a recurrent module, and a decoder."""

	def __init__(self, rnn_type, ntoken, noutputs, ninp, nhid, nlayers, dropout=0.5, tie_weights=False, is_embedding=False):
		super(RNNModel, self).__init__()
		self.drop = nn.Dropout(dropout)
		if is_embedding:
			self.encoder = nn.Embedding(ntoken, ninp)
		else:
			ninp = ntoken
			self.encoder = nn.Embedding(ntoken, ninp)
			self.encoder.weight.data =torch.eye(ntoken)
			self.encoder.weight.requires_grad = False

		if rnn_type in ['LSTM', 'GRU']:
			self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
		else:
			try:
				nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
			except KeyError:
				raise ValueError( """An invalid option for `--model` was supplied,
								 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
			self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
		self.decoder = nn.Linear(nhid, noutputs)
		self.sigmoid = nn.Sigmoid()

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
		# self.encoder.weight.data.uniform_(-initrange, initrange)
		self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)

	def forward(self, input, hidden, lengths):
		emb = self.drop(self.encoder(input))
		emb_packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, enforce_sorted = False)
		output_packed, hidden = self.rnn(emb_packed, hidden)
		output_padded, _ = nn.utils.rnn.pad_packed_sequence(output_packed)
		output = self.drop(output_padded)
		decoded = self.decoder(output)
		decoded = self.sigmoid(decoded)
		return decoded, hidden

	def init_hidden(self, bsz):
		weight = next(self.parameters())
		if self.rnn_type == 'LSTM':
			return (weight.new_zeros(self.nlayers, bsz, self.nhid),
					weight.new_zeros(self.nlayers, bsz, self.nhid))
		else:
			return weight.new_zeros(self.nlayers, bsz, self.nhid)