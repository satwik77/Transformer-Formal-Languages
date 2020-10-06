import sys
import numpy as np
import torch
from collections import defaultdict, Counter
import random
import ipdb as pdb
class RDyck1Language():
	def __init__(self, p, q):
		self.pair_nums = 1
		self.pairs = ['()']
		self.vocabulary = ['(', ')']
		self.n_letters = len (self.vocabulary)

		self.openpar= [elt[0] for elt in self.pairs]
		self.closepar = [elt[1] for elt in self.pairs]

		self.p = p
		self.q = q

	def generate_dyck (self, current_size, max_size, max_depth = -1):
		# Houston, we have a problem here. (Limit exceeded.)
		if current_size >= max_size: 
			return ''
		
		prob = random.random()
		# Grammar: S -> (_i S )_i with prob p | SS with prob q | empty with prob 1 - (p+q)
		if prob < self.p:
			chosen_pair = np.random.choice (self.pairs) # randomly pick one of the pairs.
			sample = chosen_pair[0] + self.generate_dyck (current_size + 2, max_size, max_depth) + chosen_pair [1]
			if len (sample) <= max_size:
				if (max_depth == -1) or (len(sample)==0) or (self.depth_counter(sample).sum(1).max() <= max_depth):
					return sample
		elif prob < self.p + self.q:
			sample = self.generate_dyck (current_size, max_size, max_depth) + self.generate_dyck (current_size, max_size, max_depth)
			if len (sample) <= max_size:
				if (max_depth == -1) or (len(sample)==0) or (self.depth_counter(sample).sum(1).max() <= max_depth):
					return sample
		else:
			return ''

		return ''

	def generate_reset_dyck (self, max_size):
		str1 = self.generate_dyck(0, max_size)
		str2 = self.generate_dyck(0, max_size)
		
		#Select a substring from str1
		rand_idx = np.random.randint(0, len(str1)+1)
		substr = str1[:rand_idx]
		string = substr + '1' + str2
		return string

	def output_generator (self, seq):
		start_id = 0
		out_seq = ''
		for i,ch in enumerate(seq):
			if ch != '1':
				substr = seq[start_id:i+1]
				counter = Counter(substr)
				if counter['('] - counter[')'] > 0:
					out_seq += '1'
				else:
					out_seq += '0'
			else:
				out_seq += '0'
				start_id = i+1
		return out_seq

	def generate_list (self, num, min_size, max_size):
		arr = []
		size_info = defaultdict (list)
		counter = 0
		while counter < num:
			sample = self.generate_reset_dyck (max_size)
			if sample not in arr and len(sample) >= min_size:
				counter += 1
				arr.append (sample)
				size_info [len(sample)].append(sample)
				print ('{} samples generated.'.format(counter), end = '\r', flush = True)
		print()
		return arr, size_info

	def depth_counter (self, seq):
		depths = np.zeros((len(seq), 1))
		start_id = 0
		for i,ch in enumerate(seq):
			if ch != '1':
				substr = seq[start_id:i+1]
				counter = Counter(substr)
				depth = counter['('] - counter[')']
				depths[i,0] = depth
			else:
				depths[i,0] = 0
				start_id = i+1

		return depths

	def training_set_generator (self, num, min_size, max_size):
		input_arr, input_size_arr = self.generate_list (num, min_size, max_size)
		output_arr = []
		for seq in input_arr:
			output_arr.append (self.output_generator (seq))
		return input_arr, output_arr, input_size_arr
	
	def lineToTensorOutput(self, line):
		tensor = torch.zeros(len(line), self.n_letters)
		for li, letter in enumerate(line):
			tensor[li][0] = 1.0
			if letter == '1':
				tensor[li][1] = 1.0
		return tensor