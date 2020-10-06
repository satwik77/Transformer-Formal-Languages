import os
import logging
import ipdb as pdb
import numpy as np
import re
import torch
from torch.utils.data import Dataset
import pandas as pd
import unicodedata
from collections import OrderedDict
from src.utils.dyck_generator import DyckLanguage
from src.utils.reset_dyck_generator import RDyck1Language
from src.utils.data_generator import CounterLanguage
from src.utils.shuffle_generator import ShuffleLanguage
from src.utils.parity_generator import ParityLanguage
from src.utils import starfree_generator
from src.utils import nonstarfree_generator
from src.utils import crl_generator
from src.utils import tomita_generator
from src.utils.boolean_expr_generator import NAryBooleanExpLang
from src.utils.sentence_processing import sents_to_idx

class DyckCorpus(object):
	def __init__(self, p_val, q_val, num_par, lower_window, upper_window, size, min_depth=0, max_depth=-1, debug=False):

		if debug:
			size =100

		self.Lang = DyckLanguage(num_par, p_val, q_val)
		self.source, self.target, st = self.generate_data(size, lower_window, upper_window, min_depth, max_depth)
		lx = [len(st[z]) for z in list(st.keys())]
		# self.source, self.target = zip(*sorted(zip(source, target), key = lambda x: len(x[0])))
		self.st =st
		self.noutputs = self.Lang.n_letters

	def generate_data(self, size, lower_window, upper_window, min_depth, max_depth):
		inputs, outputs, st = self.Lang.training_set_generator(size, lower_window, upper_window, min_depth, max_depth)
		return inputs, outputs, st

class RDyckCorpus(object):

	def __init__(self, p_val, q_val, lower_window, upper_window, size, debug = False):
		if debug:
			size = 100
		
		self.Lang = RDyck1Language(p_val, q_val)
		self.source, self.target, _ = self.generate_data(size, lower_window, upper_window)
		self.noutputs = 2

	def generate_data(self, size, lower_window, upper_window):

		inputs, outputs, st = self.Lang.training_set_generator(size, lower_window, upper_window)
		return inputs, outputs, st


class ShuffleCorpus(object):
	def __init__(self, p_val, q_val, num_par, lower_window, upper_window, size, min_depth=0, max_depth=-1, debug=False):

		if debug:
			size =100

		self.Lang = ShuffleLanguage(num_par, p_val, q_val)
		self.source, self.target, st = self.generate_data(size, lower_window, upper_window, min_depth, max_depth)
		lx = [len(st[z]) for z in list(st.keys())]
		# self.source, self.target = zip(*sorted(zip(source, target), key = lambda x: len(x[0])))
		self.st =st
		self.noutputs = self.Lang.n_letters

	def generate_data(self, size, lower_window, upper_window, min_depth, max_depth):
		inputs, outputs, st = self.Lang.training_set_generator(size, lower_window, upper_window, min_depth, max_depth)
		return inputs, outputs, st


class CounterCorpus(object):
	def __init__(self, num_par, lower_window, upper_window, size, debug=False, unique = False):
		if debug:
			size =50

		self.Lang = CounterLanguage(num_par)
		self.unique = unique
		self.source, self.target, st = self.generate_data(size, lower_window, upper_window)
		# self.source, self.target = zip(*sorted(zip(source, target), key = lambda x: len(x[0])))
		self.st =st
		self.noutputs = self.Lang.n_letters

	def generate_data(self, size, lower_window, upper_window):
		inputs, outputs, st = self.Lang.generate_sample(size, lower_window, upper_window)
		if self.unique:
			inputs, outputs = zip(*set(zip(inputs, outputs)))
			inputs = list(inputs)
			outputs = list(outputs)

		return inputs, outputs, st

class ParityCorpus(object):

	def __init__(self, lower_window, upper_window, size, debug = False):
		L = (lower_window + upper_window) // 2
		p = L / (2 * (1 + L))
		self.Lang = ParityLanguage(p, p)
		self.source, self.target = self.generate_data(size, lower_window, upper_window)
		self.noutputs = self.Lang.n_letters

	def generate_data(self, size, lower_window, upper_window):
		inputs, outputs = self.Lang.training_set_generator(size, lower_window, upper_window)
		return inputs, outputs

class CRLCorpus(object):

	def __init__(self,n, lower_window, upper_window, size, debug = False):
		assert n > 0 and n <= 5
		L = (lower_window + upper_window) // 2
		p = L / (2 * (1 + L))
		self.Lang = getattr(crl_generator, 'CyclicRegularLang{}'.format(n))(p, p)
		self.source, self.target = self.generate_data(size, lower_window, upper_window)
		self.noutputs = self.Lang.n_letters

	def generate_data(self, size, lower_window, upper_window):
		inputs, outputs = self.Lang.training_set_generator(size, lower_window, upper_window)
		return inputs, outputs

class StarFreeCorpus(object):

	def __init__(self, lang, n, lower_window, upper_window, size, debug = False, unique = False):
		self.Lang = getattr(starfree_generator, lang+'Language')(n)
		self.unique = unique
		self.source, self.target = self.generate_data(size, lower_window, upper_window)
		self.noutputs = self.Lang.n_letters		

	def generate_data(self, size, lower_window, upper_window):
		inputs, outputs = self.Lang.training_set_generator(size, lower_window, upper_window)
		if self.unique:
			inputs, outputs = zip(*set(zip(inputs, outputs)))
			inputs = list(inputs)
			outputs = list(outputs)

		return inputs, outputs

class NonStarFreeCorpus(object):
	def __init__(self, lang, num_par, lower_window, upper_window, size, debug=False, unique = False):
		if debug:
			size =50

		self.Lang = getattr(nonstarfree_generator, lang+'Language')(num_par)
		self.unique = unique
		self.source, self.target = self.generate_data(size, lower_window, upper_window)
		self.noutputs = 2

	def generate_data(self, size, lower_window, upper_window):
		inputs, outputs = self.Lang.training_set_generator(size, lower_window, upper_window)
		if self.unique:
			inputs, outputs = zip(*set(zip(inputs, outputs)))
			inputs = list(inputs)
			outputs = list(outputs)

		return inputs, outputs

class TomitaCorpus(object):
	def __init__(self,n, lower_window, upper_window, size,unique,leak = False, debug = False):
		assert n > 0 and n <= 7
		L = (lower_window + upper_window) // 2
		p = L / (2 * (1 + L))
		self.unique = unique
		self.leak = leak
		self.Lang = getattr(tomita_generator, 'Tomita{}Language'.format(n))(p, p)
		self.source, self.target = self.generate_data(size, lower_window, upper_window)
		self.noutputs = self.Lang.n_letters


	def generate_data(self, size, lower_window, upper_window):
		inputs, outputs = self.Lang.training_set_generator(size, lower_window, upper_window, self.leak)
		
		if self.unique:
			inputs, outputs = zip(*set(zip(inputs, outputs)))
			inputs = list(inputs)
			outputs = list(outputs)
		
		return inputs, outputs

class BooleanExprCorpus(object):
	def __init__(self, p, n, lower_window, upper_window, size, debug = False):
		assert n > 0 and n <= 3
		self.Lang = NAryBooleanExpLang(n = n, p = p)
		self.source, self.target = self.generate_data(size, lower_window, upper_window)
		self.noutputs = 2

	def generate_data(self, size, lower_window, upper_window):
		inputs, outputs = self.Lang.training_set_generator(size, lower_window, upper_window)
		return inputs, outputs

class CAB_n_ABDCorpus(object):

	def __init__(self, lower_window, upper_window, size, debug = False):
		self.Lang = starfree_generator.CAB_n_ABDLanguage()
		self.source, self.target = self.generate_data(size, lower_window, upper_window)
		self.noutputs = 4

	def generate_data(self, size, lower_window, upper_window):
		inputs, outputs = self.Lang.training_set_generator(size, lower_window, upper_window)
		return inputs, outputs

class Sampler(object):
	def __init__(self, corpus, voc, batch_size, bptt=None):
		self.voc = voc
		self.batch_size = batch_size
		self.Lang = corpus.Lang
		self.data = corpus.source
		self.targets = corpus.target
		self.bptt= bptt
		self.num_batches = np.ceil(len(self.data)/ batch_size)
		self.noutputs = corpus.noutputs

	def get_batch(self, i):
		batch_size = min(self.batch_size, len(self.data) - 1 -i)

		word_batch = self.data[i: i+batch_size]
		target_batch = self.targets[i: i+batch_size]
		word_lens = torch.tensor([len(x) for x in word_batch], dtype= torch.long)

		batch_ids = sents_to_idx(self.voc, word_batch)
		source = batch_ids[:,:-1].transpose(0,1)
		max_length = word_lens.max().item()
		target_tensors = [self.Lang.lineToTensorOutput(line)[:len(word)] for line,word in zip(target_batch, word_batch)]
		target_tensors_padded = [torch.cat([t, torch.zeros(max_length - len(t), self.noutputs)]).unsqueeze(0) for t in target_tensors]
		target = torch.cat(target_tensors_padded).transpose(0,1)

		return source, target, word_lens

	def batchify(self, data, bsz):
		# Work out how cleanly we can divide the dataset into bsz parts.
		nbatch = data.size(0) // bsz
		# Trim off any extra elements that wouldn't cleanly fit (remainders).
		data = data.narrow(0, 0, nbatch * bsz)
		# Evenly divide the data across the bsz batches.
		data = data.view(bsz, -1).t().contiguous()
		return data

	def __len__(self):
		return len(self.data)

