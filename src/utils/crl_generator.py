import numpy as np
import textwrap
import torch
from abc import ABC, abstractmethod 
import ipdb as pdb
import torch

class DFA(object):
	def __init__(self, sigma, Q, delta, q0, F):
		self.sigma = sigma
		self.Q = Q
		self.delta = delta
		self.q0 = q0
		self.F = F

	def __call__(self, string):
		qt = self.q0
		for symbol in string:
			qt = self.delta(qt, symbol)
		if qt in self.F:
			return True
		else:
			return False

class CyclicRegularLang(ABC):
	def __init__(self, p, q):
		self.p = p
		self.q = q
		self.sigma = []
		self.Q = {}
		self.delta = self.transition_function
		self.q0 = None
		self.F = {}
		self.n_letters = len(self.sigma)
		self.dfa = DFA(self.sigma, self.Q, self.delta, self.q0, self.F)

	@abstractmethod
	def transition_function(self, q, s):
		pass

	def belongs_to_lang(self, string):
		if self.dfa(string):
			#pdb.set_trace()
			return True
		else:
			return False
	'''
	def generate_strings(self, length, w = '', strings = []):

		if len(w) == length:
			if self.belongs_to_lang(w):
				strings.append(w)
			return

		for symbol in self.sigma:
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
			self.generate_strings(l, w = '', strings = strings)
			arr += strings
		arr = list(np.random.choice(arr, size = min(num, len(arr)), replace = False))

		return arr
		'''
		arr = []
		while len(arr) < num:
			string = self.generate_string(max_length)
			if string in arr:
				continue
			if len(string) >= min_length and len(string) <= max_length:
				if self.belongs_to_lang(string):
					arr.append(string)
					print("Generated {}/{} samples".format(len(arr), num), end = '\r', flush = True)
		print()
		return arr

	def output_generator(self, seq):
		output_seq = ''
		for i in range(1, len(seq)+1):
			part_seq = seq[:i]
			if self.belongs_to_lang(part_seq):
				output_seq += '1'
			else:
				output_seq += '0'
		return output_seq

	@abstractmethod
	def depth_counter(self, seq):
		pass

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

class CyclicRegularLang1(CyclicRegularLang):

	def __init__(self, p, q):
		super(CyclicRegularLang1, self).__init__(p,q)
		self.sigma = ['0', '1']
		self.Q = ['q0', 'q1']
		self.delta = self.transition_function
		self.q0 = 'q0'
		self.F = {'q0'}
		self.dfa = DFA(self.sigma, self.Q, self.delta, self.q0, self.F)
		self.n_letters = len(self.sigma)


	def transition_function(self, q, s):
		if q == 'q0':
			return 'q1'
		else:
			return 'q0'

	def depth_counter(self, seq):
		## To Do. The current implementation is not right, just a placeholder
		return np.ones((len(seq), 1))

class CyclicRegularLang2(CyclicRegularLang):

	def __init__(self, p, q):
		super(CyclicRegularLang2, self).__init__(p,q)
		self.sigma = ['0', '1']
		self.Q = ['q0', 'q1', 'q2']
		self.delta = self.transition_function
		self.q0 = 'q0'
		self.F = {'q0'}
		self.dfa = DFA(self.sigma, self.Q, self.delta, self.q0, self.F)
		self.n_letters = len(self.sigma)


	def transition_function(self, q, s):
		if q == 'q0':
			return 'q1'
		elif q == 'q1':
			return 'q2'
		else:
			return 'q0'

	def depth_counter(self, seq):
		## To Do. The current implementation is not right, just a placeholder
		return np.ones((len(seq), 1))

class CyclicRegularLang3(CyclicRegularLang):

	def __init__(self, p, q):
		super(CyclicRegularLang3, self).__init__(p,q)
		self.sigma = ['0', '1']
		self.Q = ['q0', 'q1']
		self.delta = self.transition_function
		self.q0 = 'q0'
		self.F = {'q0'}
		self.dfa = DFA(self.sigma, self.Q, self.delta, self.q0, self.F)
		self.n_letters = len(self.sigma)


	def transition_function(self, q, s):
		if q == 'q0' and s == '0':
			return 'q0'

		if q == 'q0' and s == '1':
			return 'q1'

		if q == 'q1' and s == '0':
			return 'q1'

		if q == 'q1' and s == '1':
			return 'q0'

	def depth_counter(self, seq):
		## To Do. The current implementation is not right, just a placeholder
		return np.ones((len(seq), 1))

class CyclicRegularLang4(CyclicRegularLang):

	def __init__(self, p, q):
		super(CyclicRegularLang4, self).__init__(p,q)
		self.sigma = ['0', '1']
		self.Q = ['q0', 'q1','q2','q3','q4']
		self.delta = self.transition_function
		self.q0 = 'q0'
		self.F = {'q0'}
		self.dfa = DFA(self.sigma, self.Q, self.delta, self.q0, self.F)
		self.n_letters = len(self.sigma)


	def transition_function(self, q, s):
		if q == 'q0' and s == '0':
			return 'q1'

		if q == 'q0' and s == '1':
			return 'q0'

		if q == 'q1' and s == '0':
			return 'q2'

		if q == 'q1' and s == '1':
			return 'q0'
		
		if q == 'q2' and s == '0':
			return 'q3'
		
		if q == 'q2' and s == '1':
			return 'q1'

		if q == 'q3' and s == '0':
			return 'q4'
		
		if q == 'q3' and s == '1':
			return 'q2'
		
		if q == 'q4' and s == '0':
			return 'q4'
		
		if q == 'q4' and s == '1':
			return 'q0'

	def depth_counter(self, seq):
		## To Do. The current implementation is not right, just a placeholder
		return np.ones((len(seq), 1))

class CyclicRegularLang5(CyclicRegularLang):

	def __init__(self, p, q):
		super(CyclicRegularLang5, self).__init__(p,q)
		self.sigma = ['0', '1']
		self.Q = ['q0', 'q1','q2','q3','q4', 'q5']
		self.delta = self.transition_function
		self.q0 = 'q0'
		self.F = {'q0'}
		self.dfa = DFA(self.sigma, self.Q, self.delta, self.q0, self.F)
		self.n_letters = len(self.sigma)


	def transition_function(self, q, s):
		
		if q == 'q0' and s == '0':
			return 'q1'

		if q == 'q0' and s == '1':
			return 'q2'

		if q == 'q1' and s == '0':
			return 'q0'

		if q == 'q1' and s == '1':
			return 'q0'
		
		if q == 'q2' and s == '0':
			return 'q3'
		
		if q == 'q2' and s == '1':
			return 'q3'

		if q == 'q3' and s == '0':
			return 'q4'
		
		if q == 'q3' and s == '1':
			return 'q3'
		
		if q == 'q4' and s == '0':
			return 'q4'
		
		if q == 'q4' and s == '1':
			return 'q5'

		if q == 'q5' and s == '0':
			return 'q1'

		if q == 'q5' and s == '1':
			return 'q4'


	def depth_counter(self, seq):
		## To Do. The current implementation is not right, just a placeholder
		return np.ones((len(seq), 1))