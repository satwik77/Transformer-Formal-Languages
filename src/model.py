import os
import sys
import math
import logging
import random
from time import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from tensorboardX import SummaryWriter
from gensim import models

from src.components.rnns import RNNModel
from src.components.transformers import TransformerModel, TransformerXLModel, SimpleTransformerModel
from src.components.mogrifierLSTM import MogrifierLSTMModel
from src.components.sa_rnn import SARNNModel

from src.utils.sentence_processing import *
from src.utils.logger import print_log, store_results
from src.utils.helper import save_checkpoint

from src.visualize_san import generate_visualizations

import ipdb as pdb

from collections import OrderedDict
import copy


class LanguageModel(nn.Module):
	def __init__(self, config, voc, device, logger):
		super(LanguageModel, self).__init__()

		self.config = config
		self.device = device
		self.logger= logger
		self.voc = voc
		self.lr =config.lr
		self.epsilon = 0.5

		self.logger.debug('Initalizing Model...')
		self._initialize_model()

		self.logger.debug('Initalizing Optimizer and Criterion...')
		self._initialize_optimizer()

		# nn.CrossEntropyLoss() does both F.log_softmax() and nn.NLLLoss() 
		# self.criterion = nn.NLLLoss() 
		# self.criterion = nn.CrossEntropyLoss(reduction= 'none')
		self.criterion = nn.MSELoss(reduction = 'none')

	def _initialize_model(self):
		if self.config.model_type == 'RNN':
			self.model = RNNModel(self.config.cell_type, self.voc.nwords, self.voc.noutputs, self.config.emb_size, self.config.hidden_size, self.config.depth, self.config.dropout, self.config.tied, self.config.use_emb).to(self.device)
		elif self.config.model_type == 'SAN':
			self.model = TransformerModel(self.voc.nwords, self.voc.noutputs,
										self.config.d_model, self.config.heads,
										self.config.d_ffn, self.config.depth,
										self.config.dropout, pos_encode = self.config.pos_encode,
										bias = self.config.bias, pos_encode_type= self.config.pos_encode_type,
										max_period = self.config.max_period).to(self.device)
										
		elif self.config.model_type == 'SAN-Simple':
			self.model = SimpleTransformerModel(self.voc.nwords, self.voc.noutputs, self.config.d_model,
												self.config.heads, self.config.d_ffn, self.config.depth,
												self.config.dropout, pos_encode = self.config.pos_encode, bias = self.config.bias,
												posffn= self.config.posffn, freeze_emb= self.config.freeze_emb,
												freeze_q = self.config.freeze_q, freeze_k = self.config.freeze_k,
												freeze_v = self.config.freeze_v, freeze_f = self.config.freeze_f,
												zero_keys = self.config.zero_k, pos_encode_type= self.config.pos_encode_type,
												max_period = self.config.max_period).to(self.device)
		elif self.config.model_type == 'SAN-Rel':
			self.model = TransformerXLModel(self.voc.nwords, self.voc.noutputs, self.config.d_model, self.config.heads, self.config.d_ffn, self.config.depth, self.config.dropout).to(self.device)
		elif self.config.model_type == 'Mogrify':
			self.model = MogrifierLSTMModel(self.config.cell_type, self.voc.nwords, self.config.emb_size, self.config.hidden_size, self.config.depth, self.config.dropout, self.config.tied).to(self.device)
		elif self.config.model_type == 'SARNN':
			self.model = SARNNModel(self.config.cell_type, self.voc.nwords, self.config.emb_size, self.config.hidden_size, self.config.depth, self.config.dropout, self.config.tied).to(self.device)


	def _initialize_optimizer(self):
		self.params = self.model.parameters()

		if self.config.opt == 'adam':
			self.optimizer = optim.Adam(self.params, lr=self.config.lr)
		elif self.config.opt == 'adadelta':
			self.optimizer = optim.Adadelta(self.params, lr=self.config.lr)
		elif self.config.opt == 'asgd':
			self.optimizer = optim.ASGD(self.params, lr=self.config.lr)
		elif self.config.opt =='rmsprop':
			self.optimizer = optim.RMSprop(self.params, lr=self.config.lr)
		else:
			self.optimizer = optim.SGD(self.params, lr=self.config.lr)
			self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', factor=self.config.decay_rate, patience=self.config.decay_patience, verbose=True)


	def trainer(self, source, targets, lengths, hidden, config, device=None ,logger=None):

		self.optimizer.zero_grad()

		if config.model_type == 'RNN':
			output, hidden = self.model(source, hidden, lengths)
		elif config.model_type == 'SAN' or config.model_type == 'SAN-Rel' or config.model_type == 'SAN-Simple':
			output = self.model(source)
		elif config.model_type == 'Mogrify':
			output, hidden = self.model(source, hidden)
		elif config.model_type == 'SARNN':
			output, hidden = self.model(source, hidden)

		#pdb.set_trace()
		mask = (source != 0).float().view(-1)
		loss = self.criterion(output.view(-1,self.voc.noutputs), targets.contiguous().view(-1,self.voc.noutputs)).mean(1)
		loss = (mask * loss).sum()/ mask.sum()
		loss.backward()

		if self.config.max_grad_norm > 0:
			torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

		self.optimizer.step()
		# for p in self.model.parameters():
			# p.data.add_(-self.lr, p.grad.data)

		if config.model_type != 'SAN' and config.model_type != 'SAN-Simple' and config.model_type != 'SAN-Rel':
			hidden = self.repackage_hidden(hidden)

		return loss.item(), hidden

	def evaluator(self, source, targets, lengths, hidden, config, device=None ,logger=None):

		if config.model_type == 'RNN':
			output, hidden = self.model(source, hidden, lengths)
		elif config.model_type == 'SAN' or config.model_type == 'SAN-Rel' or config.model_type == 'SAN-Simple':
			output = self.model(source)
		elif config.model_type == 'Mogrify':
			output, hidden = self.model(source, hidden)
		elif config.model_type == 'SARNN':
			output, hidden = self.model(source, hidden)


		batch_acc = 0.0
		mask = (source!=0).float().unsqueeze(-1)
		masked_output = mask*output
		try:
			out_np= np.int_(masked_output.detach().cpu().numpy() >= self.epsilon)
			target_np = np.int_(targets.detach().cpu().numpy())
		except:
			pdb.set_trace()

		for j in range(out_np.shape[1]):
			out_j = out_np[:,j]
			target_j = target_np[:,j]
			if np.all(np.equal(out_j, target_j)) and (out_j.flatten() == target_j.flatten()).all():
			# If so, set `pred` as one
				batch_acc+=1
		
		batch_acc = batch_acc/source.size(1)


		if config.model_type != 'SAN' and config.model_type != 'SAN-Simple' and config.model_type != 'SAN-Rel':
			hidden = self.repackage_hidden(hidden)

		return  batch_acc, hidden


	def repackage_hidden(self, h):
		"""Wraps hidden states in new Tensors, to detach them from their history."""

		if isinstance(h, torch.Tensor):
			return h.detach()
		else:
			return tuple(self.repackage_hidden(v) for v in h)


def build_model(config, voc, device, logger):
	'''
		Add Docstring
	'''
	model = LanguageModel(config, voc, device, logger)
	model = model.to(device)

	return model



def train_model(model, train_loader, val_loader_bins, voc, device, config, logger, epoch_offset= 0, max_val_acc=0.0, writer= None):
	'''
		Add Docstring
	'''

	if config.histogram and writer:
		for name, param in model.named_parameters():
			writer.add_histogram(name, param, epoch_offset)
	
	estop_count=0
	n_bins = len(val_loader_bins)
	max_train_acc = max_val_acc
	best_train_epoch = 0
	max_val_acc_bins = [max_val_acc for i in range(n_bins)]
	max_val_acc_bin1 = -1
	#max_val_gen_acc = max_val_acc
	#max_train_gen_acc = max_val_acc
	best_epoch_bins = [0 for i in range(n_bins)]
	iters = 0
	viz_every = int(train_loader.num_batches // 4)
	#best_gen_epoch = 0
	#val_acc_epoch_bins = [run_validation(config, model, val_loader_bin, voc, device, logger) for val_loader_bin in val_loader_bins]
	#pdb.set_trace()
	try:
		for epoch in range(1, config.epochs + 1):
			od = OrderedDict()
			od['Epoch'] = epoch + epoch_offset
			print_log(logger, od)

			
			train_loss_epoch = 0.0
			train_acc_epoch = 0.0
			val_acc_epoch = 0.0


			# Train Mode
			model.train()

			start_time= time()
			# Batch-wise Training

			lr_epoch =  model.optimizer.state_dict()['param_groups'][0]['lr']
			for batch, i in enumerate(range(0, len(train_loader), config.batch_size)):
				if config.viz and config.model_type == 'SAN' and batch % viz_every == 0:
					#if val_acc_bin1 > max_val_acc_bin1:
					#	max_val_acc_bin1 = val_acc_bin1
					val_acc_bins = [run_validation(config, model, val_loader_bins[i], voc, device, logger) for i in range(1,4)]
					generate_visualizations(model, config, voc, run_name = config.run_name, iteration = iters, score = sum(val_acc_bins)/3, device = device)
					if writer:
						bin_acc_dict = {'bin{}_score'.format(i+1) : val_acc_bins[i] for i in range(3)}
						writer.add_scalars('acc/val_acc_iters', bin_acc_dict, iters)

				if config.model_type != 'SAN' and config.model_type != 'SAN-Simple' and config.model_type != 'SAN-Rel':
					hidden = model.model.init_hidden(config.batch_size)
				else:
					hidden = None
				source, targets, word_lens = train_loader.get_batch(i)
				source, targets, word_lens = source.to(device), targets.to(device), word_lens.to(device)

				loss, hidden = model.trainer(source, targets, word_lens, hidden, config)
				# if batch % config.display_freq==0:
				# 	od = OrderedDict()
				# 	od['Batch'] = batch
				# 	od['Loss'] = loss
				# 	print_log(logger, od)

				train_loss_epoch += loss #* len(source)
				iters += 1

			train_loss_epoch = train_loss_epoch / train_loader.num_batches

			time_taken = (time() - start_time)/60.0

			if writer:
				writer.add_scalar('loss/train_loss', train_loss_epoch, epoch + epoch_offset)

			logger.debug('Training for epoch {} completed...\nTime Taken: {}'.format(epoch, time_taken))
			logger.debug('Starting Validation')

			val_acc_epoch_bins = [run_validation(config, model, val_loader_bin, voc, device, logger) for val_loader_bin in val_loader_bins]
			train_acc_epoch = run_validation(config, model, train_loader, voc, device, logger)
			if train_acc_epoch ==  max(max_train_acc, train_acc_epoch):
				best_train_epoch = epoch
				max_train_acc = train_acc_epoch
			'''
			if config.generalize:
				val_gen_acc_epoch = run_validation(config, model, val_gen_loader, voc, device, logger)
			'''
			val_acc_epoch = np.mean(val_acc_epoch_bins)
			if config.opt == 'sgd' and model.scheduler:
				model.scheduler.step(val_acc_epoch)

			save_flag = False
			for i in range(n_bins):
				if val_acc_epoch_bins[i] > max_val_acc_bins[i]:
					save_flag = True
					max_val_acc_bins[i] = val_acc_epoch_bins[i]
					#max_train_acc = train_acc_epoch
					best_epoch_bins[i] = epoch

					logger.debug('Validation Accuracy bin{} : {}'.format(i, val_acc_epoch_bins[i]))

					#estop_count=0
			
			if save_flag:
				state = {
							'epoch' : epoch + epoch_offset,
							'model_state_dict': model.state_dict(),
							'voc': model.voc,
							'optimizer_state_dict': model.optimizer.state_dict(),
							'train_loss' : train_loss_epoch,
							'lr' : lr_epoch
						}
				for i in range(n_bins):
					state['val_acc_epoch_bin{}'.format(i)] = val_acc_epoch_bins[i]
					state['max_val_acc_bin{}'.format(i)] = max_val_acc_bins[i]

				save_checkpoint(state, epoch + epoch_offset, logger, config.model_path, config.ckpt)

			'''
			else:
				estop_count+=1
			'''
			'''
			if config.generalize:
				if val_gen_acc_epoch > max_val_gen_acc:
					max_val_gen_acc = val_gen_acc_epoch
					max_train_gen_acc = train_acc_epoch
					best_gen_epoch = epoch
					state = {
						'epoch' : epoch + epoch_offset,
						'model_state_dict': model.state_dict(),
						'voc': model.voc,
						'optimizer_state_dict': model.optimizer.state_dict(),
						'train_loss' : train_loss_epoch,
						'val_acc_epoch' : val_acc_epoch,
						'val_gen_acc_epoch' : val_gen_acc_epoch,
						'max_val_acc': max_val_acc,
						'max_val_gen_acc': max_val_gen_acc,
						'max_train_acc' : max_train_acc,
						'max_train_gen_acc': max_train_gen_acc,
						'lr' : lr_epoch
					}
					logger.debug('Validation Accuracy: {}'.format(val_acc_epoch))

					save_checkpoint(state, epoch + epoch_offset, logger, config.model_path, config.ckpt)
					estop_count=0
			'''
			if writer:
				writer.add_scalar('acc/train_acc', train_acc_epoch, epoch + epoch_offset)
				for i in range(n_bins):
					writer.add_scalar('acc/val_acc_bin{}'.format(i), val_acc_epoch_bins[i], epoch + epoch_offset)
				#writer.add_scalar('acc/val_gen_acc', val_gen_acc_epoch, epoch + epoch_offset)

			od = OrderedDict()
			od['Epoch'] = epoch + epoch_offset
			od['train_loss'] = train_loss_epoch
			od['train_acc'] = train_acc_epoch
			#od['val_acc_epoch']= val_acc_epoch
			#od['val_gen_acc_epoch'] = val_gen_acc_epoch
			#od['max_val_acc'] = max_val_acc
			#od['max_val_gen_acc'] = max_val_gen_acc
			od['lr_epoch'] = lr_epoch
			for i in range(n_bins):
				od['val_acc_epoch_bin{}'.format(i)] = val_acc_epoch_bins[i]
				od['max_val_acc_bin{}'.format(i)] = max_val_acc_bins[i]

			print_log(logger, od)

			if config.histogram and writer:
				# pdb.set_trace()
				for name, param in model.named_parameters():
					writer.add_histogram(name, param, epoch + epoch_offset)

			# if estop_count > 10:
			# 	logger.debug('Early Stopping at Epoch: {} after no improvement in {} epochs'.format(epoch, estop_count))
			# 	break
			
			if np.mean(val_acc_epoch_bins) >= 0.999:
				logger.info('Reached optimum performance!')
				break
			
		writer.export_scalars_to_json(os.path.join(config.board_path, 'all_scalars.json'))
		writer.close()

		logger.info('Training Completed for {} epochs'.format(config.epochs))

		if config.results:
			store_results(config, max_val_acc_bins, max_train_acc, best_train_epoch, train_loss_epoch, best_epoch_bins)
			logger.info('Scores saved at {}'.format(config.result_path))
	
	except KeyboardInterrupt:
		logger.info('Exiting Early....')
		if config.results:
			store_results(config, max_val_acc_bins, max_train_acc, best_train_epoch, train_loss_epoch, best_epoch_bins)
			logger.info('Scores saved at {}'.format(config.result_path))		



def run_validation(config, model, val_loader, voc, device, logger):
	batch_num = 0
	val_acc_epoch =0.0
	model.eval()



	with torch.no_grad():
		for batch, i in enumerate(range(0, len(val_loader), val_loader.batch_size)):
			if config.model_type != 'SAN' and config.model_type != 'SAN-Simple' and config.model_type != 'SAN-Rel':
				hidden = model.model.init_hidden(config.batch_size)
			else:
				hidden = None

			source, targets, word_lens = val_loader.get_batch(i)
			source, targets, word_lens = source.to(device), targets.to(device), word_lens.to(device)
			acc, hidden = model.evaluator(source, targets, word_lens, hidden, config)
			val_acc_epoch += acc 
			batch_num += 1
	
	if batch_num != val_loader.num_batches:
		pdb.set_trace()

	val_acc_epoch = val_acc_epoch / val_loader.num_batches

	return val_acc_epoch

def run_test(config, model, test_loader, voc, device, logger):
	batch_num =1
	test_acc_epoch =0.0
	strings = []
	correct_or_not = []
	lengths = []
	depths = []
	model.eval()

	with torch.no_grad():
		for batch, i in enumerate(range(len(test_loader) - 1)):
			if config.model_type != 'SAN' and config.model_type != 'SAN-Simple' and config.model_type != 'SAN-Rel':
				hidden = model.model.init_hidden(config.batch_size)
			else:
				hidden = None
			try:
				source, targets, word_lens = test_loader.get_batch(i)
			except Exception as e:
				pdb.set_trace()
			source, targets, word_lens = source.to(device), targets.to(device), word_lens.to(device)
			acc, hidden = model.evaluator(source, targets, word_lens, hidden, config)
			test_acc_epoch += acc
			
			source_str = test_loader.data[i]
			source_len = len(source_str)
			source_depth = test_loader.Lang.depth_counter(source_str).sum(1).max()
			strings.append(source_str)
			lengths.append(source_len)
			depths.append(source_depth)
			correct_or_not.append(acc)

			print("Completed {}/{}...".format(i+1, len(test_loader) - 1), end = '\r', flush = True)

	test_acc_epoch = test_acc_epoch / (len(test_loader) - 1)
	test_analysis_df = pd.DataFrame(
						{
							'String' : strings,
							'Length' : lengths,
							'Depth'  : depths,
							'Score'	 : correct_or_not
						}
					)

	return test_acc_epoch, test_analysis_df	


