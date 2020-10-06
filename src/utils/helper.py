import logging
import pdb
import torch
from glob import glob
from torch.autograd import Variable
import numpy as np
import os
import sys
import re
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")



def gpu_init_pytorch(gpu_num):
	'''
		Initialize GPU
	'''
	torch.cuda.set_device(int(gpu_num))
	device = torch.device("cuda:{}".format(
		gpu_num) if torch.cuda.is_available() else "cpu")
	return device


def create_save_directories(log_path, req_path):
	if not os.path.exists(log_path):
		os.makedirs(log_path)

	if not os.path.exists(req_path):
		os.makedirs(req_path)


def save_checkpoint(state, epoch, logger, model_path, ckpt):
	'''
		Saves the model state along with epoch number. The name format is important for 
		the load functions. Don't mess with it.

		Args:
			model state
			epoch number
			logger variable
			directory to save models
			checkpoint name
	'''
	ckpt_path = os.path.join(model_path, '{}_{}.pt'.format(ckpt, epoch))
	logger.info('Saving Checkpoint at : {}'.format(ckpt_path))
	torch.save(state, ckpt_path)


def get_latest_checkpoint(model_path, logger):
	'''
		Looks for the checkpoint with highest epoch number in the directory "model_path" 

		Args:
			model_path: including the run_name
			logger variable: to log messages
		Returns:
			checkpoint: path to the latest checkpoint 
	'''

	ckpts = glob('{}/*.pt'.format(model_path))
	ckpts = sorted(ckpts)

	if len(ckpts) == 0:
		logger.warning('No Checkpoints Found')

		return None
	else:
		latest_epoch = max([int(x.split('_')[-1].split('.')[0]) for x in ckpts])
		ckpts = sorted(ckpts, key= lambda x: int(x.split('_')[-1].split('.')[0]) , reverse=True )
		ckpt_path = ckpts[0]
		logger.info('Checkpoint found with epoch number : {}'.format(latest_epoch))
		logger.debug('Checkpoint found at : {}'.format(ckpt_path))

		return ckpt_path

def load_checkpoint(model, mode, ckpt_path, logger, device, bins = -1):
	start_epoch = None
	train_loss = None
	val_loss = None
	voc = None
	score = -1

	try:
		checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
		model.load_state_dict(checkpoint['model_state_dict'])
		model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		start_epoch = checkpoint['epoch']
		train_loss  =checkpoint['train_loss']
		#val_acc = checkpoint['val_acc_epoch']
		voc = checkpoint['voc']
		if bins != -1:
			score = [checkpoint['max_val_acc_bin{}'.format(i)] for i in range(bins)]
		else:
			score = checkpoint['max_val_acc']

		model.to(device)

		if mode == 'train':
			model.train()
		else:
			model.eval()

		if logger:
			logger.info('Successfully Loaded Checkpoint from {}, with epoch number: {} for {}'.format(ckpt_path, start_epoch, mode))

		return start_epoch, train_loss, score, voc
	except Exception as e:
		print(e)
		if logger:
			logger.warning('Could not Load Checkpoint from {}  \t \"at load_checkpoint() in helper.py \"'.format(ckpt_path))
		return start_epoch, train_loss, score, voc



class Voc:
	def __init__(self):

		self.w2id= {'T': 0}
		self.id2w = {0:'T'}
		self.w2c = {'T':0}
		self.nwords = 1

	def add_word(self, word):
		if word not in self.w2id:
			self.w2id[word] = self.nwords
			self.id2w[self.nwords] = word
			self.w2c[word] = 1
			self.nwords += 1
		else:
			try:
				self.w2c[word] += 1
			except:
				pdb.set_trace()

	def add_sent(self, sent):
		for word in sent.split():
			self.add_word(word)

	def get_id(self, idx):
		return self.w2id[idx]

	def get_word(self, idx):
		return self.id2w[idx]

	def create_vocab_dict(self, corpus, debug=False):
		sents= corpus.source[-10:]
		merged_sent = ''.join(sents)
		all_letters = list(set(list(merged_sent)))

		for letter in all_letters:
			self.add_word(letter)


		# self.most_frequent(args.vocab_size)
		assert len(self.w2id) == self.nwords
		assert len(self.id2w) == self.nwords
