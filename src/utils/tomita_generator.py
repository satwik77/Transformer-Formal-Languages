import numpy as np
import torch
from collections import Counter
import ipdb as pdb
from abc import ABC, abstractmethod 
from src.utils.crl_generator import DFA

class TomitaLanguage(ABC):

	def __init__(self, p, q):
		self.p = p
		self.q = q
		self.sigma = ['0', '1']
		self.n_letters = len(self.sigma)

	@abstractmethod
	def belongs_to_lang(self, seq):
		pass

	def generate_string(self,min_length, max_length):
		string = ''
		symbols = self.sigma + ['T']
		while len(string) < max_length:
			symbol = np.random.choice(symbols, p = [self.p, self.q, 1-(self.p + self.q)])
			if symbol == 'T':
				break
			else:
				string += str(symbol)

		return string

	def generate_list(self, num, min_length, max_length, leak):

		arr = []
		while len(arr) < num:
			string = self.generate_string(min_length, max_length)
			if not leak and string in arr:
				continue
			if len(string) >= min_length and len(string) <= max_length:
				if self.belongs_to_lang(string):
					arr.append(string)
					print("Generated {}/{} samples".format(len(arr), num), end = '\r', flush = True)
		print()	
		return (arr)

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

	def training_set_generator(self, num, min_size, max_size, leak):
		input_arr = self.generate_list (num, min_size, max_size, leak)
		output_arr = []
		for seq in input_arr:
			output_arr.append (self.output_generator (seq))
		return input_arr, output_arr
	
	def lineToTensorOutput(self, line):
		tensor = torch.zeros(len(line), 2)
		for li, letter in enumerate(line):
			tensor[li][int(letter)] = 1.0
		return tensor

class Tomita1Language(TomitaLanguage):

	def __init__(self, p, q):
		super(Tomita1Language, self).__init__(p, q)
		self.sigma = ['0', '1']
		self.Q = ['q0', 'q1']
		self.delta = self.transition_function
		self.q0 = 'q0'
		self.F = {'q0'}
		self.dead_states = {'q1'}
		self.dfa = DFA(self.sigma, self.Q, self.delta, self.q0, self.F)
		self.n_letters = len(self.sigma)

	def transition_function(self, q, s):
		if q == 'q0':
			if s == '0':
				return 'q1'
			if s == '1':
				return 'q0'
		if q == 'q1':
			return 'q1'

	def get_final_state(self, seq):
		q = self.q0
		for s in seq:
			q = self.transition_function(q, s)
		return q


	def belongs_to_lang(self, seq):
		return self.dfa(seq)

	def get_legal_characters(self, seq):
		q = 'q0'
		leg_chars = []

		for i,s in enumerate(seq):
			leg_char = []
			q_f_0 =  self.get_final_state(seq[:i+1] + '0')
			q_f_1 = self.get_final_state(seq[:i+1] + '1')
			if q_f_0 not in self.dead_states:
				leg_char.append('0')
			if q_f_1 not in self.dead_states:
				leg_char.append('1')
			leg_chars.append(leg_char)
			q = self.dfa(s)

		return leg_chars

	def generate_string(self,min_length, max_length):
		length = np.random.randint(min_length, max_length + 1)
		string = ''.join(['1' for i in range(length)])
		return string

	def output_generator(self, seq):
		output_seq = ''
		legal_chars = self.get_legal_characters(seq)
		for legal_char in legal_chars:
			if '0' in legal_char:
				output_seq += '1'
			else:
				output_seq += '0'
			
			if '1' in legal_char:
				output_seq += '1'
			else:
				output_seq += '0'
		return output_seq

	def lineToTensorOutput(self, line):
		tensor = torch.zeros(len(line)//2, 2)
		for li,i in enumerate(range(0, len(line), 2)):
			l1, l2 = line[i], line[i+1]
			tensor[li][0] = float(l1)
			tensor[li][1] = float(l2)
		return tensor

	def depth_counter(self, seq):
		## To Do. The current implementation is not right, just a placeholder
		return np.ones((len(seq), 1))

class Tomita2Language(Tomita1Language):

	def __init__(self, p, q):
		super(Tomita2Language, self).__init__(p, q)
		self.sigma = ['0', '1']
		self.Q = ['q0', 'q1', 'q2']
		self.delta = self.transition_function
		self.q0 = 'q0'
		self.F = {'q0'}
		self.dead_states = {'q2'}
		self.dfa = DFA(self.sigma, self.Q, self.delta, self.q0, self.F)
		self.n_letters = len(self.sigma)

	def transition_function(self, q, s):
		if q == 'q0':
			if s == '0':
				return 'q2'
			if s == '1':
				return 'q1'
		if q == 'q1':
			if s == '0':
				return 'q0'
			if s == '1':
				return 'q2'
		if q == 'q2':
			return 'q2'

	def generate_string(self,min_length, max_length):
		length = (np.random.randint(min_length, max_length) + 1)//2
		string = ''.join(['10' for i in range(length)])
		return string

	def depth_counter(self, seq):
		## To Do. The current implementation is not right, just a placeholder
		return np.ones((len(seq), 1))

class Tomita3Language(Tomita1Language):

	def __init__(self, p, q):
		super(Tomita3Language, self).__init__(p, q)
		self.sigma = ['0', '1']
		self.Q = ['q0', 'q1', 'q2', 'q3', 'q4']
		self.delta = self.transition_function
		self.q0 = 'q0'
		self.F = {'q0', 'q1', 'q2'}
		self.dead_states = {'q3','q4'}
		self.dfa = DFA(self.sigma, self.Q, self.delta, self.q0, self.F)
		self.n_letters = len(self.sigma)

	def transition_function(self, q, s):
		if q == 'q0':
			if s == '0':
				return 'q0'
			if s == '1':
				return 'q1'
		if q == 'q1':
			if s == '0':
				return 'q3'
			if s == '1':
				return 'q0'
		if q == 'q2':
			if s== '0':
				return 'q3'
			if s == '1':
				return 'q1'
		if q == 'q3':
			if s == '0':
				return 'q2'
			if s == '1':
				return 'q4'

		if q == 'q4':
			return 'q4'

	def generate_string(self,min_length, max_length):
		length = np.random.randint(min_length, max_length+1)
		string = ''
		last_toss = None
		last_one_count = 0
		while len(string) != length:
			toss = np.random.choice(['0','1'])
			if toss == '1':
				char_count = np.random.randint(length - len(string) + 1)
				string += ''.join([toss for _ in range(char_count)])
				if last_toss == '0' and char_count != 0:
					last_one_count = char_count
				else:
					last_one_count += char_count
			else:
				if last_toss is None or last_one_count%2 == 0:
					char_count = np.random.randint(length - len(string) + 1)
					string += ''.join([toss for _ in range(char_count)])
				else:
					choices = np.arange(0, length - len(string) + 1, 2)
					char_count = np.random.choice(choices)
					string += ''.join([toss for _ in range(char_count)])
			if char_count != 0:
				last_toss = toss

		if not self.dfa(string):
			pdb.set_trace()
		
		return string

	def depth_counter(self, seq):
		## To Do. The current implementation is not right, just a placeholder
		return np.ones((len(seq), 1))


class Tomita4Language(Tomita3Language):

	def __init__(self, p, q):
		super(Tomita4Language, self).__init__(p, q)
		self.sigma = ['0', '1']
		self.Q = ['q0', 'q1', 'q2', 'q3']
		self.delta = self.transition_function
		self.q0 = 'q0'
		self.F = {'q0', 'q1', 'q2'}
		self.dead_states = {'q3'}
		self.dfa = DFA(self.sigma, self.Q, self.delta, self.q0, self.F)
		self.n_letters = len(self.sigma)

	def transition_function(self, q, s):
		if q == 'q0':
			if s == '0':
				return 'q1'
			if s == '1':
				return 'q0'
		if q == 'q1':
			if s == '0':
				return 'q2'
			if s == '1':
				return 'q0'
		if q == 'q2':
			if s== '0':
				return 'q3'
			if s == '1':
				return 'q0'
		if q == 'q3':
			return 'q3'

	def belongs_to_lang(self, seq):
		return self.dfa(seq)

	def generate_string(self, min_length, max_length):
		length = np.random.randint(min_length, max_length+1)
		string = ''
		while len(string) < length:
			toss = np.random.choice(['0', '1'])
			if toss == '0':
				if len(string) >=2 and string[-1] == '0' and string[-2] == '0':
					continue
				else:
					string += toss
			else:
				string += toss
		if not self.dfa(string):
			pdb.set_trace()
		return string

	def depth_counter(self, seq):
		## To Do. The current implementation is not right, just a placeholder
		return np.ones((len(seq), 1))


	

class Tomita5Language(TomitaLanguage):

	def __init__(self, p, q):
		super(Tomita5Language, self).__init__(p, q)

	def belongs_to_lang(self, seq):
		if seq == '':
			return True
		counter = Counter(seq)
		if counter['0'] % 2 == 0 and counter['1'] % 2 == 0:
			return True
		return False

	def depth_counter(self, seq):
		## To Do. The current implementation is not right, just a placeholder
		return np.ones((len(seq), 1))


class Tomita6Language(TomitaLanguage):

	def __init__(self, p, q):
		super(Tomita6Language, self).__init__(p, q)

	def belongs_to_lang(self, seq):
		if seq == '':
			return True
		counter = Counter(seq)
		if abs(counter['0'] - counter['1']) % 3 == 0:
			return True
		return False

	def depth_counter(self, seq):
		## To Do. The current implementation is not right, just a placeholder
		depths = []
		for i in range(1, len(seq)+1):
			subseq = seq[:i]
			counter = Counter(subseq)
			n = abs(counter['0'] - counter['1'])//3
			depths.append(n)
		return np.array(depths)[:,np.newaxis].astype(float)

class Tomita7Language(Tomita3Language):

	def __init__(self, p, q):
		super(Tomita3Language, self).__init__(p, q)
		self.sigma = ['0', '1']
		self.Q = ['q0', 'q1', 'q2', 'q3', 'q4']
		self.delta = self.transition_function
		self.q0 = 'q0'
		self.F = {'q0', 'q1', 'q2', 'q3'}
		self.dead_states = {'q4'}
		self.dfa = DFA(self.sigma, self.Q, self.delta, self.q0, self.F)
		self.n_letters = len(self.sigma)

	def transition_function(self, q, s):
		if q == 'q0':
			if s == '0':
				return 'q0'
			if s == '1':
				return 'q1'
		if q == 'q1':
			if s == '0':
				return 'q2'
			if s == '1':
				return 'q1'
		if q == 'q2':
			if s== '0':
				return 'q2'
			if s == '1':
				return 'q3'
		if q == 'q3':
			if s == '0':
				return 'q4'
			else:
				return 'q3'
		if q == 'q4':
			return 'q4'

	def belongs_to_lang(self, seq):
		return self.dfa(seq)

	def check_string(self, string, max_length):
		if not self.dfa(string):
			pdb.set_trace()
		if len(string) == max_length:
			return True
		else:
			return False

	def generate_string(self,min_length, max_length):
		string = ''
		#length = np.random.randint(min_length, max_length+1)
		length = max_length
		num_zeros = np.random.randint(0, length+1)
		string += ''.join(['0' for _ in range(num_zeros)])
		if self.check_string(string, length):
			return string
		num_ones = np.random.randint(0, length - len(string) + 1)
		string += ''.join(['1' for _ in range(num_ones)])
		if self.check_string(string, length):
			return string
		num_zeros = np.random.randint(0, length - len(string) + 1)
		string += ''.join(['0' for _ in range(num_zeros)])
		if self.check_string(string, length):
			return string
		num_ones = np.random.randint(0, length - len(string) + 1)
		string += ''.join(['1' for _ in range(num_ones)])
		self.check_string(string, length)
		return string

	def depth_counter(self, seq):
		## To Do. The current implementation is not right, just a placeholder
		return np.ones((len(seq), 1))

