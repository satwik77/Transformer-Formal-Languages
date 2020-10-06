import sys
import numpy as np
import torch
from collections import defaultdict, Counter
import random
import ipdb as pdb

class ParityLanguage():

	def __init__(self, p, q):
		self.p = p
		self.q = q
		self.vocabulary = ['0', '1']
		self.pos_symbol = '1'
		self.n_letters = len(self.vocabulary)

	def check_parity(self, w):
		if w == '':
			return True
		counter = Counter(w)
		return counter[self.pos_symbol] % 2 == 0
	'''
	def generate_strings(self, length, w = '', strings = []):

		if len(w) == length:
			if self.check_parity(w):
				strings.append(w)
			return

		for symbol in self.vocabulary:
			self.generate_strings(length, w + symbol, strings)
	'''

	def generate_string(self, max_length):
		string = ''
		while len(string) < max_length:
			symbol = np.random.choice(3, p = [self.p, self.q, 1-(self.p + self.q)])
			if symbol == 2:
				break
			else:
				string += str(symbol)

		return string

	def generate_list(self, num, min_length, max_length):
		'''
		arr = []
		for l in range(min_length, max_length + 1):
			strings = []
			self.generate_strings(l,w = '', strings = strings)
			arr += strings
		arr = list(np.random.choice(arr, size = num, replace = False))

		return arr
		'''
		arr = []
		while len(arr) < num:
			string = self.generate_string(max_length)
			if string in arr:
				continue
			if len(string) >= min_length and len(string) <= max_length:
				if self.check_parity(string):
					arr.append(string)
				else:
					new_string = string + '1'
					if len(new_string) <= max_length:
						arr.append(new_string)
		return arr


	def output_generator(self, seq):
		output_seq = ''
		for i in range(1, len(seq)+1):
			part_seq = seq[:i]
			if self.check_parity(part_seq):
				output_seq += '1'
			else:
				output_seq += '0'
		return output_seq

	def depth_counter(self, seq):
		max_depth = np.zeros ((len(seq),1))
		for i in range(1, len(seq)+1):
			max_depth[i-1] = Counter(seq[:i])[self.pos_symbol]

		return max_depth

	def training_set_generator(self, num, min_size, max_size):
		input_arr = self.generate_list (num, min_size, max_size)
		output_arr = []
		for seq in input_arr:
			output_arr.append (self.output_generator (seq))
		return input_arr, output_arr

	def lineToTensorOutput(self, line):
		tensor = torch.zeros(len(line), 2)
		for li, letter in enumerate(line):
			tensor[li][int(letter)] = 1.0
		return tensor

