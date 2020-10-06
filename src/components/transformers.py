import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb as pdb
from transformers import TransfoXLModel, TransfoXLConfig
from src.components.self_attention import MultiHeadedAttention
from src.components.transformer_encoder import Encoder, EncoderLayer, EncoderLayerFFN
from src.components.position_encodings import 	PositionalEncoding, CosineNpiPositionalEncoding, LearnablePositionalEncoding

class TransformerModel(nn.Module):
	"""Container module with an encoder, a recurrent or transformer module, and a decoder."""

	def __init__(self, ntoken,noutputs, d_model, nhead, d_ffn, nlayers, dropout=0.5, use_embedding=False, pos_encode = True, bias = False, pos_encode_type = 'absolute', max_period = 10000.0):
		super(TransformerModel, self).__init__()
		try:
			from torch.nn import TransformerEncoder, TransformerEncoderLayer
		except:
			raise ImportError('TransformerEncoder module does not exist in PyTorch 1.1 or lower.')
		self.model_type = 'Transformer'
		self.src_mask = None

		# if use_embedding:
		# 	self.pos_encoder = PositionalEncoding(d_model, dropout)
		# 	self.encoder = nn.Embedding(ntoken, d_model)
		# else:
		# 	self.pos_encoder = PositionalEncoding(ntoken, dropout)
		# 	self.encoder = nn.Embedding(ntoken, ntoken)
		# 	self.encoder.weight.data =torch.eye(ntoken)
		# 	self.encoder.weight.requires_grad = False
		if pos_encode_type == 'absolute':
			self.pos_encoder = PositionalEncoding(d_model, dropout, max_period)
		elif pos_encode_type == 'cosine_npi':
			self.pos_encoder = CosineNpiPositionalEncoding(d_model, dropout)
		elif pos_encode_type == 'learnable':
			self.pos_encoder = LearnablePositionalEncoding(d_model, dropout)
		self.pos_encode = pos_encode
		self.encoder = nn.Embedding(ntoken, d_model)

		encoder_layers = TransformerEncoderLayer(d_model, nhead, d_ffn, dropout)
		self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

		self.d_model = d_model
		self.decoder = nn.Linear(d_model, noutputs, bias=bias)
		self.sigmoid= nn.Sigmoid()
		self.bias = bias

		self.init_weights()

	def _generate_square_subsequent_mask(self, sz):
		mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
		mask = mask.float()
		mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		return mask

	def init_weights(self):
		initrange = 0.1
		self.encoder.weight.data.uniform_(-initrange, initrange)
		if self.bias:
			self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)

	def forward(self, src, has_mask=True, get_attns = False, get_encoder_reps = False):
		if has_mask:
			device = src.device
			mask = self._generate_square_subsequent_mask(len(src)).to(device)
			self.src_mask = mask
		else:
			self.src_mask = None
		src = self.encoder(src) * math.sqrt(self.d_model)
		if self.pos_encode:
			src = self.pos_encoder(src)
		if get_attns:
			attns = []
			encoder_layers = self.transformer_encoder.layers
			inp = src
			for layer in encoder_layers:
				attn = layer.self_attn(inp, inp, inp, attn_mask = self.src_mask)[1]
				inp = layer(inp, src_mask = self.src_mask) 
				attns.append(attn)


		transformer_output = self.transformer_encoder(src, self.src_mask)
		output = self.decoder(transformer_output)
		output = self.sigmoid(output)
		# return F.log_softmax(output, dim=-1)
		
		if get_attns:
			return output, attns	
		
		if get_encoder_reps:
			return output, transformer_output

		return output

class TransformerXLModel(nn.Module):
	def __init__(self, ntoken,noutputs, d_model, nhead, d_ffn, nlayers, dropout=0.5, use_embedding=False):
		super(TransformerXLModel, self).__init__()
		self.config = TransfoXLConfig(	vocab_size = ntoken, cutoffs = [],
									 	d_model = d_model, d_embed = d_model,
										n_head = nhead, d_inner = d_ffn,
									 	n_layer = nlayers, tie_weights = False,
										d_head = d_model // nhead,adaptive = False,
										dropout = dropout)
		
		self.transformer_encoder = TransfoXLModel(self.config)
		self.decoder = nn.Linear(d_model, noutputs)
		self.sigmoid= nn.Sigmoid()

	def forward(self, src):
		output = self.transformer_encoder(src.transpose(0,1), mems = None)[0].transpose(0,1)
		output = self.decoder(output)
		output = self.sigmoid(output)

		return output


class SimpleTransformerModel(nn.Module):
	def __init__(self, ntoken, noutputs,
				d_model, nhead, d_ffn,
				nlayers, dropout=0.5, use_embedding=False,
				pos_encode = True, bias = False, posffn = False,
				freeze_emb = False, freeze_q = False, freeze_k = False,
				freeze_v = False, freeze_f = False, zero_keys = False,
				pos_encode_type = 'absolute', max_period = 10000.0):

		super(SimpleTransformerModel, self).__init__()

		self.bias = bias
		self.pos_encode = pos_encode
		self.d_model = d_model

		if self.pos_encode:
			if pos_encode_type == 'absolute':
				self.pos_encoder = PositionalEncoding(d_model, dropout, max_period)
			elif pos_encode_type == 'cosine_npi':
				self.pos_encoder = CosineNpiPositionalEncoding(d_model, dropout)
			elif pos_encode_type == 'learnable':
				self.pos_encoder = LearnablePositionalEncoding(d_model, dropout)
		self.encoder = nn.Embedding(ntoken, d_model)
		if freeze_emb:
			self.encoder.requires_grad_(False)

		self_attn = MultiHeadedAttention(nhead, d_model, dropout, bias,
										freeze_q, freeze_k,
										freeze_v, zero_keys)
		if not posffn: 
			encoder_layers = EncoderLayer(self_attn)
		else:
			feed_forward = nn.Sequential(nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, d_model))
			encoder_layers = EncoderLayerFFN(self_attn, feed_forward)
		self.transformer_encoder = Encoder(encoder_layers, nlayers)

		self.decoder = nn.Linear(d_model, noutputs, bias=bias)
		self.sigmoid= nn.Sigmoid()
		if freeze_f:
			self.decoder.requires_grad_(False)

		#self.init_weights()
		

	def _generate_square_subsequent_mask(self, sz):
		mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
		mask = mask.float()
		mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		return mask

	def init_weights(self):
		initrange = 0.1
		self.encoder.weight.data.uniform_(-initrange, initrange)
		if self.bias:
			self.decoder.bias.data.zero_()
		self.decoder.weight.data.uniform_(-initrange, initrange)

	def forward(self, src):
		src_mask = self._generate_square_subsequent_mask(len(src)).to(src.device)

		src = self.encoder(src) * math.sqrt(self.d_model)
		if self.pos_encode:
			src = self.pos_encoder(src)

		src = src.transpose(0,1)
		transformer_output = self.transformer_encoder(src, src_mask)
		transformer_output = transformer_output.transpose(0,1)

		output = self.decoder(transformer_output)
		output = self.sigmoid(output)

		return output
		



