import logging
import pdb
import pandas as pd
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
import json

'''Logging Modules'''

#log_format='%(asctime)s | %(levelname)s | %(filename)s:%(lineno)s - %(funcName)5s() ] | %(message)s'
def get_logger(name, log_file_path='./logs/temp.log', logging_level=logging.INFO, log_format='%(asctime)s | %(levelname)s | %(filename)s: %(lineno)s : %(funcName)s() ::\t %(message)s'):
	logger = logging.getLogger(name)
	logger.setLevel(logging_level)
	formatter = logging.Formatter(log_format)

	file_handler = logging.FileHandler(log_file_path, mode='w')
	file_handler.setLevel(logging_level)
	file_handler.setFormatter(formatter)

	stream_handler = logging.StreamHandler()
	stream_handler.setLevel(logging_level)
	stream_handler.setFormatter(formatter)

	logger.addHandler(file_handler)
	logger.addHandler(stream_handler)

	# logger.addFilter(ContextFilter(expt_name))

	return logger


def print_log(logger, dict):
	string = ''
	for key, value in dict.items():
		string += '\n {}: {}\t'.format(key.replace('_', ' '), value)
	# string = string.strip()
	logger.info(string)



def store_results(config, max_val_acc_bins, train_acc, best_train_epoch, train_loss, best_epoch_bins):
	try:
		with open(config.result_path) as f:
			res_data =json.load(f)
	except:
		res_data = {}

	data= {'run_name' : config.run_name
	#, 'max_val_acc' : max_val_acc
	#, 'max_val_gen_acc': max_val_gen_acc
	, 'train_acc' : train_acc
	#, 'train_gen_acc' : train_gen_acc
	, 'best_train_epoch' : best_train_epoch
	, 'train_loss' : train_loss
	, 'lang' : config.lang
	, 'use_emb' : config.use_emb
	, 'emb_size': config.emb_size
	, 'model_type': config.model_type
	, 'cell_type' : config.cell_type
	#, 'best_epoch': best_epoch
	#, 'best_gen_epoch' : best_gen_epoch
	, 'hidden_size' : config.hidden_size
	, 'd_model' : config.d_model
	, 'heads' : config.heads
	, 'depth' : config.depth
	, 'dropout' : config.dropout
	, 'lr' : config.lr
	, 'batch_size' : config.batch_size
	, 'epochs' : config.epochs
	, 'opt' : config.opt
	, 'Dyck-n' : config.num_par
	, 'train_size': config.training_size
	, 'test_size': config.test_size
	, 'window' : [config.lower_window, config.upper_window]
	, 'pq' : [config.p_val, config.q_val]
	, 'generalize' : config.generalize
	, 'dataset' : config.dataset
	, 'pos_encode' : config.pos_encode
	, 'seed' : config.seed
	, 'pos_encode_type' : config.pos_encode_type
	, 'max_period' : config.max_period
	# , 'memory_size' : config.memory_size
	# , 'memory_dim' : config.memory_dim
	}
	for i,max_val_acc_bin in enumerate(max_val_acc_bins):
		data['max_val_acc_bin{}'.format(i)] = max_val_acc_bin
		data['best_epoch_bin{}'.format(i)] = best_epoch_bins[i]

	# res_data.update(data)
	res_data[str(config.run_name)] = data

	with open(config.result_path, 'w', encoding='utf-8') as f:
		json.dump(res_data, f, ensure_ascii= False, indent= 4)

