import logging
import pdb
import torch
from glob import glob
from torch.autograd import Variable
import numpy as np
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def sent_to_idx(voc, sent, max_length=-1):
	idx_vec = []
	for w in sent:
		try:
			idx= voc.get_id(w)
			idx_vec.append(idx)
		except:
			pdb.set_trace()
	idx_vec.append(voc.get_id('T'))
	idx_vec= pad_seq(idx_vec, max_length+1, voc)
	return idx_vec



def sents_to_idx(voc, sents):
	max_length = max([len(s) for s in sents])
	all_indexes = []
	for sent in sents:
		all_indexes.append(sent_to_idx(voc, sent, max_length))

	all_indexes= torch.tensor(all_indexes, dtype= torch.long)
	return all_indexes


def pad_seq(seq, max_length, voc):
	seq += [voc.get_id('T') for i in range(max_length - len(seq))]
	return seq


def sent_to_tensor(voc, sentence, device, max_length):
	indexes = sent_to_idx(voc, sentence, max_length)
	return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def batch_to_tensor(voc, sents, device, max_length):
	batch_sent = []
	# batch_label = []
	for sent in sents:
		sent_id = sent_to_tensor(voc, sent, device, max_length)
		batch_sent.append(sent_id)

	return batch_sent


def idx_to_sent(voc, tensor, no_eos=False):
	sent_word_list = []
	for idx in tensor:
		word = voc.get_word(idx.item())
		if no_eos:
			if word != 'T':
				sent_word_list.append(word)
			# else:
			# 	break
		else:
			sent_word_list.append(word)
	return sentence_word_list


def idx_to_sents(voc, tensors, no_eos=False):
	tensors = tensors.transpose(0, 1)
	batch_word_list = []
	for tensor in tensors:
		batch_word_list.append(idx_to_sent(voc, tensor, no_eos))

	return batch_word_list




def process_batch(sents, labels, voc, device):
	# len_sents = [len(s) for s in sents]
	# max_length = max(len_sents)
	# sents_padded = [pad_seq(s, max_length, voc) for s in sents]
	# labels_padded = [pad_label(s, max_length, tag) for s in labels]
	# pdb.set_trace()
	one_hot = torch.eye(voc.nwords, dtype= torch.float)
	one_hot_input = torch.tensor(one_hot[sents[0]]).unsqueeze(1)

	# Convert to [Max_len X Batch]
	sent_var = Variable(torch.FloatTensor(one_hot_input))#.transpose(0, 1)
	label_var = Variable(torch.FloatTensor(labels))#.transpose(0, 1)
	# label_var = torch.LongTensor(labels)

	sent_var = sent_var.to(device)
	label_var = label_var.to(device)

	return sent_var, label_var
