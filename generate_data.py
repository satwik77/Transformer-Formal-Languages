import os
import sys
import math
import logging
import pdb
import random
import json
import numpy as np
from attrdict import AttrDict
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from collections import OrderedDict
import copy
try:
	import cPickle as pickle
except ImportError:
	import pickle

from src.args import build_parser
from src.utils.helper import *
from src.utils.logger import get_logger, print_log, store_results
from src.dataloader import DyckCorpus, Sampler, CounterCorpus, ShuffleCorpus, ParityCorpus, CRLCorpus, StarFreeCorpus, NonStarFreeCorpus, TomitaCorpus, BooleanExprCorpus, RDyckCorpus, CAB_n_ABDCorpus
from src.model import LanguageModel, build_model, train_model, run_validation
from src.utils.dyck_generator import DyckLanguage
from src.utils.shuffle_generator import ShuffleLanguage

star_free_langs = ['AAstarBBstar', 'ABStar']

def load_data(config, num_bins = 2):
	'''
		Loads the data from the datapath in torch dataset form

		Args:
			config (dict) : configuration/args
			num_bins (int) : Number of validation bins
		Returns:
			dataobject (dict) 
	'''
	if config.mode == 'train':
		#logger.debug('Loading Training Data...')

		'''Load Datasets'''
		if config.lang == 'Dyck':
			train_corpus 	    = DyckCorpus(config.p_val, config.q_val, config.num_par, config.lower_window, config.upper_window, config.training_size, config.lower_depth, config.upper_depth, config.debug)
			val_corpus_bins     = [DyckCorpus(config.p_val, config.q_val, config.num_par, config.lower_window, config.upper_window, config.test_size, config.lower_depth, config.upper_depth, config.debug)]
			val_corpus_bin = []
			lower_window = config.bin1_lower_window
			upper_window = config.bin1_upper_window
			lower_depth = config.bin1_lower_depth
			upper_depth = config.bin1_upper_depth
			for i in range(num_bins):
				print("Generating Data for depths [{}, {}] and Lengths [{}, {}]".format(lower_depth, upper_depth, lower_window, upper_window))
				val_corpus_bin = DyckCorpus(config.p_val, config.q_val, config.num_par, lower_window, upper_window, config.test_size, lower_depth, upper_depth, config.debug)
				val_corpus_bins.append(val_corpus_bin)
				
				if config.vary_len:
					lower_window = upper_window
					upper_window = upper_window + config.len_incr
				
				if config.vary_depth:
					lower_depth = upper_depth
					upper_depth = upper_depth + config.depth_incr
			
		elif config.lang == 'Counter':
			train_corpus 	= CounterCorpus( config.num_par, config.lower_window, config.upper_window, config.training_size, config.debug)
			val_corpus_bins = [CounterCorpus( config.num_par, config.lower_window, config.upper_window, config.test_size, config.debug, unique = True)]

			lower_window = config.upper_window + 1
			upper_window = config.upper_window + config.len_incr

			for i in range(num_bins):
				val_corpus_bin = CounterCorpus(config.num_par, lower_window, upper_window, config.test_size, config.debug, unique = True)
				val_corpus_bins.append(val_corpus_bin)

				lower_window = upper_window
				upper_window += config.len_incr

		elif config.lang == 'Shuffle':
			train_corpus = ShuffleCorpus(config.p_val, config.q_val, config.num_par, config.lower_window, config.upper_window, config.training_size, config.lower_depth, config.upper_depth, config.debug)
			val_corpus_bins = [ShuffleCorpus(config.p_val, config.q_val, config.num_par, config.lower_window, config.upper_window, config.test_size, config.lower_depth, config.upper_depth, config.debug)]
			lower_window = config.upper_window + 2
			upper_window = config.upper_window + config.len_incr
			lower_depth = config.bin1_lower_depth
			upper_depth = config.bin1_upper_depth
			for i in range(num_bins):
				print("Generating Data for depths [{}, {}] and Lengths [{}, {}]".format(lower_depth, upper_depth, lower_window, upper_window))
				val_corpus_bin = ShuffleCorpus(config.p_val, config.q_val, config.num_par, lower_window, upper_window, config.test_size, lower_depth, upper_depth, config.debug)
				val_corpus_bins.append(val_corpus_bin)				
				lower_window = upper_window
				upper_window = upper_window + config.len_incr
				
				if config.vary_depth:
					lower_depth = upper_depth
					upper_depth = upper_depth + config.depth_incr

		elif config.lang == 'Parity':
			print("Generating Training and Validation Bin0 Data")
			corpus = ParityCorpus(config.lower_window, config.upper_window, config.training_size + config.test_size, debug = config.debug)
			train_corpus = copy.deepcopy(corpus)
			train_corpus.source, train_corpus.target = corpus.source[:config.training_size], corpus.target[:config.training_size]
			val_corpus = copy.deepcopy(corpus)
			val_corpus.source, val_corpus.target = corpus.source[config.training_size:], corpus.target[config.training_size:]
			val_corpus_bins = [val_corpus]
			lower_window = config.upper_window + 1
			upper_window = config.upper_window + config.len_incr
			for i in range(num_bins):
				print("Generating Data for Lengths [{}, {}]".format(lower_window, upper_window))
				val_corpus_bin = ParityCorpus(lower_window, upper_window, config.test_size, debug = config.debug)
				val_corpus_bins.append(val_corpus_bin)
				lower_window = upper_window
				upper_window = upper_window + config.len_incr

		elif config.lang == 'CRL':
			print("Generating Training and Validation Bin0 Data")
			corpus = CRLCorpus(config.crl_n, config.lower_window, config.upper_window, config.training_size + config.test_size, debug = config.debug)
			train_corpus = copy.deepcopy(corpus)
			train_corpus.source, train_corpus.target = corpus.source[:config.training_size], corpus.target[:config.training_size]
			val_corpus = copy.deepcopy(corpus)
			val_corpus.source, val_corpus.target = corpus.source[config.training_size:], corpus.target[config.training_size:]
			val_corpus_bins = [val_corpus]
			lower_window = config.upper_window + 1
			upper_window = config.upper_window + config.len_incr
			for i in range(num_bins):
				print("Generating Data for Lengths [{}, {}]".format(lower_window, upper_window))
				val_corpus_bin = CRLCorpus(config.crl_n, lower_window, upper_window, config.test_size, debug = config.debug)
				val_corpus_bins.append(val_corpus_bin)
				lower_window = upper_window
				upper_window = upper_window + config.len_incr

		elif config.lang == 'Tomita':
			if not config.leak:
				print("Generating Training and Validation Bin0 Data")
				corpus = TomitaCorpus(config.num_par, config.lower_window, config.upper_window, config.training_size + config.test_size, unique = True, debug = config.debug)
				train_corpus = copy.deepcopy(corpus)
				train_corpus.source, train_corpus.target = corpus.source[:config.training_size], corpus.target[:config.training_size]
				val_corpus = copy.deepcopy(corpus)
				val_corpus.source, val_corpus.target = corpus.source[config.training_size:], corpus.target[config.training_size:]
				val_corpus_bins = [val_corpus]
				lower_window = config.upper_window + 1
				upper_window = config.upper_window + config.len_incr
				for i in range(num_bins):
					print("Generating Data for Lengths [{}, {}]".format(lower_window, upper_window))
					val_corpus_bin = TomitaCorpus(config.num_par, lower_window, upper_window, config.test_size, unique = True, debug = config.debug)
					val_corpus_bins.append(val_corpus_bin)
					lower_window = upper_window
					upper_window = upper_window + config.len_incr
			else:
				train_corpus 		= TomitaCorpus(config.num_par, config.lower_window, config.upper_window, config.training_size, unique = False, leak = True,  debug = config.debug)
				val_corpus_bins 	= [TomitaCorpus(config.num_par, config.lower_window, config.upper_window, config.test_size, unique = True, leak = True, debug = config.debug)]
				lower_window = config.upper_window + 1
				upper_window = config.upper_window + config.len_incr

				for i in range(num_bins):
					val_corpus_bin = TomitaCorpus(config.num_par, lower_window, upper_window, config.test_size, unique = True, leak = True, debug = config.debug)
					val_corpus_bins.append(val_corpus_bin)		


		elif config.lang == 'AAStarBBStar':
			print("Generating Training and Validation Bin0 Data")
			corpus = StarFreeCorpus(config.lang, config.num_par, config.lower_window, config.upper_window, config.training_size + config.test_size, debug = config.debug)
			train_corpus = copy.deepcopy(corpus)
			train_corpus.source, train_corpus.target = corpus.source[:config.training_size], corpus.target[:config.training_size]
			val_corpus = copy.deepcopy(corpus)
			val_corpus.source, val_corpus.target = corpus.source[config.training_size:], corpus.target[config.training_size:]
			val_corpus_bins = [val_corpus]
			lower_window = config.upper_window + 1
			upper_window = config.upper_window + config.len_incr
			for i in range(num_bins):
				print("Generating Data for Lengths [{}, {}]".format(lower_window, upper_window))
				val_corpus_bin = StarFreeCorpus(config.lang, config.num_par, lower_window, upper_window, config.test_size, debug = config.debug)
				val_corpus_bins.append(val_corpus_bin)
				lower_window = upper_window
				upper_window = upper_window + config.len_incr

		elif config.lang == 'ABStar':
			train_corpus 		= StarFreeCorpus(config.lang, config.num_par, config.lower_window, config.upper_window, config.training_size, config.debug)
			val_corpus_bins 	= [StarFreeCorpus(config.lang, config.num_par, config.lower_window, config.upper_window, config.training_size, config.debug,unique = True)]
			lower_window = config.upper_window + 1
			upper_window = config.upper_window + config.len_incr

			for i in range(num_bins):
				val_corpus_bin = StarFreeCorpus(config.lang, config.num_par, lower_window, upper_window, config.test_size, config.debug, unique = True)
				val_corpus_bins.append(val_corpus_bin)

		elif config.lang == 'CStarAnCStar' or config.lang == 'CStarAnCStarBnCStar' or config.lang == 'CStarAnCStarv2' or config.lang == 'D_n':
			print("Generating Training and Validation Bin0 Data")
			corpus = StarFreeCorpus(config.lang, config.num_par, config.lower_window, config.upper_window, config.training_size + config.test_size, debug = config.debug)
			train_corpus = copy.deepcopy(corpus)
			train_corpus.source, train_corpus.target = corpus.source[:config.training_size], corpus.target[:config.training_size]
			val_corpus = copy.deepcopy(corpus)
			val_corpus.source, val_corpus.target = corpus.source[config.training_size:], corpus.target[config.training_size:]
			val_corpus_bins = [val_corpus]
			lower_window = config.upper_window + 1
			upper_window = config.upper_window + config.len_incr
			for i in range(num_bins):
				print("Generating Data for Lengths [{}, {}]".format(lower_window, upper_window))
				val_corpus_bin = StarFreeCorpus(config.lang, config.num_par, lower_window, upper_window, config.test_size, debug = config.debug)
				val_corpus_bins.append(val_corpus_bin)
				lower_window = upper_window
				upper_window = upper_window + config.len_incr
		
		elif config.lang == 'CAB_n_ABD':
			print("Generating Training and Validation Bin0 Data")
			corpus = CAB_n_ABDCorpus(config.lower_window, config.upper_window, config.training_size + config.test_size, debug = config.debug)
			train_corpus = copy.deepcopy(corpus)
			train_corpus.source, train_corpus.target = corpus.source[:config.training_size], corpus.target[:config.training_size]
			val_corpus = copy.deepcopy(corpus)
			val_corpus.source, val_corpus.target = corpus.source[config.training_size:], corpus.target[config.training_size:]
			val_corpus_bins = [val_corpus]
			lower_window = config.upper_window + 1
			upper_window = config.upper_window + config.len_incr
			for i in range(num_bins):
				print("Generating Data for Lengths [{}, {}]".format(lower_window, upper_window))
				val_corpus_bin = CAB_n_ABDCorpus(config.lower_window, config.upper_window, config.test_size, debug = config.debug)
				val_corpus_bins.append(val_corpus_bin)
				lower_window = upper_window
				upper_window = upper_window + config.len_incr

		elif config.lang == 'ABABStar':
			train_corpus 	= NonStarFreeCorpus(config.lang, config.num_par, config.lower_window, config.upper_window, config.training_size, config.debug)
			val_corpus_bins = [NonStarFreeCorpus(config.lang, config.num_par, config.lower_window, config.upper_window, config.test_size, config.debug, unique = True)]

			lower_window = config.upper_window + 1
			upper_window = config.upper_window + config.len_incr

			for i in range(num_bins):
				val_corpus_bin = NonStarFreeCorpus(config.num_par, lower_window, upper_window, config.test_size, config.debug, unique = True)
				val_corpus_bins.append(val_corpus_bin)

				lower_window = upper_window
				upper_window += config.len_incr

		elif config.lang == 'AAStar':
			train_corpus 	= NonStarFreeCorpus(config.lang, config.num_par, config.lower_window, config.upper_window, config.training_size, config.debug)
			val_corpus_bins = [NonStarFreeCorpus(config.lang, config.num_par, config.lower_window, config.upper_window, config.test_size, config.debug, unique = True)]

			lower_window = config.upper_window + 1
			upper_window = config.upper_window + config.len_incr

			for i in range(num_bins):
				val_corpus_bin = NonStarFreeCorpus(config.lang, config.num_par, lower_window, upper_window, config.test_size, config.debug, unique = True)
				val_corpus_bins.append(val_corpus_bin)

				lower_window = upper_window
				upper_window += config.len_incr

		elif config.lang == 'AnStarA2':
			train_corpus 	= NonStarFreeCorpus(config.lang, config.num_par, config.lower_window, config.upper_window, config.training_size, config.debug)
			val_corpus_bins = [NonStarFreeCorpus(config.lang, config.num_par, config.lower_window, config.upper_window, config.test_size, config.debug, unique = True)]

			lower_window = config.upper_window + 1
			upper_window = config.upper_window + config.len_incr

			for i in range(num_bins):
				val_corpus_bin = NonStarFreeCorpus(config.lang, config.num_par, lower_window, upper_window, config.test_size, config.debug, unique = True)
				val_corpus_bins.append(val_corpus_bin)

				lower_window = upper_window
				upper_window += config.len_incr

		elif config.lang == 'Boolean':
			corpus = BooleanExprCorpus(config.p_val, config.num_par, config.lower_window, config.upper_window, config.training_size + config.test_size, config.debug)
			train_corpus = copy.deepcopy(corpus)
			train_corpus.source, train_corpus.target = corpus.source[:config.training_size], corpus.target[:config.training_size]
			val_corpus = copy.deepcopy(corpus)
			val_corpus.source, val_corpus.target = corpus.source[config.training_size:], corpus.target[config.training_size:]
			val_corpus_bins = [val_corpus]
			lower_window = config.upper_window + 1
			upper_window = config.upper_window + config.len_incr
			for i in range(num_bins):
				print("Generating Data for Lengths [{}, {}]".format(lower_window, upper_window))
				val_corpus_bin = BooleanExprCorpus(config.p_val, config.num_par, lower_window, upper_window, config.test_size, debug = config.debug)
				val_corpus_bins.append(val_corpus_bin)
				lower_window = upper_window
				upper_window = upper_window + config.len_incr

		elif config.lang == 'RDyck':
			corpus = RDyckCorpus(config.p_val, config.q_val, config.lower_window, config.upper_window, config.training_size + config.test_size, debug = config.debug)
			train_corpus = copy.deepcopy(corpus)
			train_corpus.source, train_corpus.target = corpus.source[:config.training_size], corpus.target[:config.training_size]
			val_corpus = copy.deepcopy(corpus)
			val_corpus.source, val_corpus.target = corpus.source[config.training_size:], corpus.target[config.training_size:]
			val_corpus_bins = [val_corpus]
			lower_window = config.upper_window + 1
			upper_window = config.upper_window + config.len_incr

			for i in range(num_bins):
				print("Generating Data for Lengths [{}, {}]".format(lower_window, upper_window))
				val_corpus_bin = RDyckCorpus(config.p_val, config.q_val, lower_window, upper_window, config.test_size, debug = config.debug)
				val_corpus_bins.append(val_corpus_bin)
				lower_window = upper_window
				upper_window = upper_window + config.len_incr



	msg = 'Training and Validation Data Loaded'
	print(msg)


	return train_corpus, val_corpus_bins

def main():
	'''read arguments'''
	parser = build_parser()
	args = parser.parse_args()
	config = args

	print("Loading Data!")
	train_corpus, val_corpus_bins = load_data(config, num_bins = config.bins)
	data_dir = os.path.join('data', config.dataset)

	if os.path.exists(data_dir) == False:
		os.mkdir(data_dir)

	print("Writing Train corpus")
	with open(os.path.join(data_dir, 'train_corpus.pk'), 'wb') as f:
		pickle.dump(file = f, obj = train_corpus)
	print("Done")

	print("Writing Val corpus bins")
	with open(os.path.join(data_dir, 'val_corpus_bins.pk'), 'wb') as f:
		pickle.dump(file = f, obj = val_corpus_bins)
	print("Done")

	print("Writing Train text files")
	with open(os.path.join(data_dir, 'train_src.txt'), 'w') as f:
		f.write('\n'.join(train_corpus.source))

	with open(os.path.join(data_dir, 'train_tgt.txt'), 'w') as f:
		f.write('\n'.join(train_corpus.target))
	print("Done")
	
	print("Writing Val text files")
	for i, val_corpus_bin in enumerate(val_corpus_bins):
		with open(os.path.join(data_dir, 'val_src_bin{}.txt'.format(i)), 'w') as f:
			f.write('\n'.join(val_corpus_bin.source))

		with open(os.path.join(data_dir, 'val_tgt_bin{}.txt'.format(i)), 'w') as f:
			f.write('\n'.join(val_corpus_bin.target))
	print("Done")

	print("Gathering Length and Depth info of the dataset")
	train_depths = list(set([train_corpus.Lang.depth_counter(line).sum(1).max() for line in train_corpus.source]))
	train_lens = list(set([len(line) for line in train_corpus.source]))

	val_lens_bins, val_depths_bins = [], []
	for i, val_corpus in enumerate(val_corpus_bins):
		val_depths = list(set([val_corpus.Lang.depth_counter(line).sum(1).max() for line in val_corpus.source]))
		val_depths_bins.append(val_depths)

		val_lens = list(set([len(line) for line in val_corpus.source]))
		val_lens_bins.append(val_lens)

	info_dict = {}
	info_dict['Lang'] = '{}-{}'.format(config.lang, config.num_par) 
	info_dict['Train Lengths'] = (min(train_lens), max(train_lens))
	info_dict['Train Depths'] = (int(min(train_depths)), int(max(train_depths)))
	info_dict['Train Size'] = len(train_corpus.source)
	
	for i, (val_lens, val_depths) in enumerate(zip(val_lens_bins, val_depths_bins)):
		info_dict['Val Bin-{} Lengths'.format(i)] = (min(val_lens), max(val_lens))
		info_dict['Val Bin-{} Depths'.format(i)] = (int(min(val_depths)), int(max(val_depths)))
		info_dict['Val Bin-{} Size'.format(i)] = len(val_corpus_bins[i].source)

	with open(os.path.join('data', config.dataset, 'data_info.json'), 'w') as f:
		json.dump(obj = info_dict, fp = f)

	print("Done")



if __name__ == "__main__":
	main()