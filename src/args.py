import argparse

### Add Early Stopping ###

def build_parser():
	# Data loading parameters
	parser = argparse.ArgumentParser(description='Run Single sequence model')


	# Mode specifications
	parser.add_argument('-mode', type=str, default='train', choices=['train', 'test'], help='Modes: train, test')
	# parser.add_argument('-debug', action='store_true', help='Operate on debug mode')
	parser.add_argument('-debug', dest='debug', action='store_true', help='Operate in debug mode')
	parser.add_argument('-no-debug', dest='debug', action='store_false', help='Operate in normal mode')
	parser.set_defaults(debug=False)
	parser.add_argument('-load_model', dest='load_model', action='store_true', help='load_model')
	parser.add_argument('-no-load_model', dest='load_model', action='store_false', help='Dont')
	parser.set_defaults(load_model=False)
	parser.add_argument('-results', dest='results', action='store_true', help='Store results')
	parser.add_argument('-no-results', dest='results', action='store_false', help='Do not store results')
	parser.set_defaults(results=True)

	# Run name should just be alphabetical word (no special characters to be included)
	parser.add_argument('-run_name', type=str, default='debug', help='run name for logs')
	parser.add_argument('-display_freq', type=int, default=35, help='number of batches after which to display loss')
	parser.add_argument('-dataset', type=str, default='Dyck-2-Depthv1', help='Dataset')


	# Input files
	parser.add_argument('-vocab_size', type=int, default=50000, help='Vocabulary size to consider')
	# parser.add_argument('-res_file', type=str, default='generations.txt', help='File name to save results in')
	# parser.add_argument('-res_folder', type=str, default='Generations', help='Folder name to save results in')
	# parser.add_argument('-out_dir', type=str, default='out', help='Out Dir')
	# parser.add_argument('-len_sort', action="store_true", help='Sort based on length')
	parser.add_argument('-histogram', dest='histogram', action='store_true', help='Operate in debug mode')
	parser.add_argument('-no-histogram', dest='histogram', action='store_false', help='Operate in normal mode')
	parser.set_defaults(histogram=True)


	# Device Configuration
	parser.add_argument('-gpu', type=int, default=1, help='Specify the gpu to use')
	parser.add_argument('-seed', type=int, default=1729, help='Default seed to set')
	parser.add_argument('-logging', type=int, default=1, help='Set to 0 if you do not require logging')
	parser.add_argument('-ckpt', type=str, default='model', help='Checkpoint file name')
	# parser.add_argument('-log_fmt', type=str, default='%(asctime)s | %(levelname)s | %(name)s | %(message)s', help='Specify format of the logger')

	# Dont modify ckpt_file
	# If you really want to then assign it a name like abc_0.pth.tar (You may only modify the abc part and don't fill in any special symbol. Only alphabets allowed
	# parser.add_argument('-date_fmt', type=str, default='%Y-%m-%d-%H:%M:%S', help='Format of the date')


	# LSTM parameters
	parser.add_argument('-emb_size', type=int, default=64, help='Embedding dimensions of inputs')
	parser.add_argument('-model_type', type=str, default='SAN', choices= ['RNN', 'SAN','SAN-Rel', 'Mogrify', 'SARNN', 'SAN-Simple'],  help='Model Type: RNN or Transformer or Mogrifier or SARNN')
	parser.add_argument('-cell_type', type=str, default='LSTM', choices= ['LSTM', 'GRU', 'RNN'],  help='RNN cell type, default: lstm')
	# parser.add_argument('-use_attn', action='store_true', help='To use attention mechanism?')
	# parser.add_argument('-attn_type', type=str, default='general', help='Attention mechanism: (general, concat), default: general')
	parser.add_argument('-hidden_size', type=int, default=64, help='Number of hidden units in each layer')
	parser.add_argument('-depth', type=int, default=1, help='Number of layers in each encoder and decoder')
	parser.add_argument('-dropout', type=float, default=0.0, help= 'Dropout probability for input/output/state units (0.0: no dropout)')
	# parser.add_argument('-emb_size', type=int, default=256, help='Embedding dimensions of encoder and decoder inputs')
	# parser.add_argument('-beam_width', type=int, default=10, help='Specify the beam width for decoder')
	parser.add_argument('-max_length', type=int, default=35, help='Specify max decode steps: Max length string to output')
	parser.add_argument('-bptt', type=int, default=35, help='Specify bptt length')

	parser.add_argument('-use_emb', dest='use_emb', action='store_true', help='use_emb Weights ')
	parser.add_argument('-no-use_emb', dest='use_emb', action='store_false', help='use_emb Weights')
	parser.set_defaults(use_emb=False)

	parser.add_argument('-init_range', type=float, default=0.08, help='Initialization range for seq2seq model')
	parser.add_argument('-tied', dest='tied', action='store_true', help='Tied Weights in input and output embeddings')
	parser.add_argument('-no-tied', dest='tied', action='store_false', help='Tied Weights in input and output embeddings')
	parser.set_defaults(tied=False)

	parser.add_argument('-generalize', dest='generalize', action='store_true', help='Whether to test on disjoint windows as train')
	parser.add_argument('-no-generalize', dest='generalize', action='store_false', help='Whether to test on disjoint windows as train')
	parser.set_defaults(generalize=True)

	# parser.add_argument('-bidirectional', dest='bidirectional', action='store_true', help='Bidirectionality in LSTMs')
	# parser.add_argument('-no-bidirectional', dest='bidirectional', action='store_false', help='Bidirectionality in LSTMs')
	# parser.set_defaults(bidirectional=True)



	''' Transformer '''
	parser.add_argument('-d_model', type=int, default=32, help='Embedding size in Transformer')
	parser.add_argument('-d_ffn', type=int, default=64, help='Hidden size of FFN in Transformer')
	parser.add_argument('-heads', type=int, default=1, help='Number of Attention heads in each layer')
	parser.add_argument('-pos_encode', dest='pos_encode', action='store_true', help='Whether to use position encodings')
	parser.add_argument('-no-pos_encode', dest='pos_encode', action='store_false', help='Whether to use position encodings')
	parser.set_defaults(pos_encode=False)	
	parser.add_argument('-max_period', type = float, default = 10000.0)
	parser.add_argument('-pos_encode_type', type = str, default = 'absolute', choices = ['absolute', 'cosine_npi','learnable'])
	parser.add_argument('-posffn', dest='posffn', action='store_true', help='Whether to use position encodings')
	parser.add_argument('-no-posffn', dest='posffn', action='store_false', help='Whether to use position encodings')
	parser.set_defaults(posffn=True)
	parser.add_argument('-bias', dest='bias', action='store_true', help='Whether to use position encodings')
	parser.add_argument('-no-bias', dest='bias', action='store_false', help='Whether to use position encodings')
	parser.set_defaults(bias=True)
	parser.add_argument('-viz', dest='viz', action='store_true', help='Whether to visualize representations of transformer')
	parser.add_argument('-no-viz', dest='viz', action='store_false', help='Whether to visualize representations of transformer')
	parser.set_defaults(viz=False)
	parser.add_argument('-freeze_emb', dest='freeze_emb', action='store_true', help='Whether to fix embedding layer')
	parser.add_argument('-freeze_q', dest='freeze_q', action='store_true', help='Whether to fix query layer')
	parser.add_argument('-freeze_k', dest='freeze_k', action='store_true', help='Whether to fix key layer')
	parser.add_argument('-freeze_v', dest='freeze_v', action='store_true', help='Whether to fix value layer')
	parser.add_argument('-freeze_f', dest='freeze_f', action='store_true', help='Whether to fix linear layer after attention')
	parser.add_argument('-zero_k', dest='zero_k', action='store_true', help='Whether to fix key matrix as null')


	# Training parameters
	parser.add_argument('-lr', type=float, default=0.005, help='Learning rate')
	parser.add_argument('-decay_patience', type=int, default=3, help='Wait before decaying learning rate')
	parser.add_argument('-decay_rate', type=float, default=0.1, help='Amount by which to decay learning rate on plateu')
	parser.add_argument('-max_grad_norm', type=float, default=-0.25, help='Clip gradients to this norm')
	parser.add_argument('-batch_size', type=int, default=32, help='Batch size')
	parser.add_argument('-epochs', type=int, default=25, help='Maximum # of training epochs')
	parser.add_argument('-opt', type=str, default='rmsprop', choices=['adam', 'rmsprop', 'adadelta', 'sgd', 'asgd'], help='Optimizer for training')


	# FL parameters
	parser.add_argument('-lang', type=str, default='Dyck', choices= ['Dyck', 'Counter', 'Shuffle','Parity','CRL', 'AAStarBBStar','ABStar','ABABStar', 'AAStar','Tomita','Boolean','CStarAnCStar','CStarAnCStarBnCStar','CStarAnCStarv2','RDyck','CAB_n_ABD','AnStarA2','D_n'], help='Formal Language')
	parser.add_argument('-lower_window', type=int, default=2, help='Lower Length Window')
	parser.add_argument('-upper_window', type=int, default=100, help='Upper Length Window')

	parser.add_argument('-lower_depth', type=int, default=0, help='Lower Length Window')
	parser.add_argument('-upper_depth', type=int, default=-1, help='Upper Length Window')

	parser.add_argument('-val_lower_window', type=int, default=52, help='Lower Length Window')
	parser.add_argument('-val_upper_window', type=int, default=100, help='Upper Length Window')

	parser.add_argument('-training_size', type=int, default=10000, help='Training data size')
	parser.add_argument('-test_size', type=int, default=500, help='Test data size')

	parser.add_argument('-memory_size', type=int, default=50, help='Size of memory/stack')
	parser.add_argument('-memory_dim', type=int, default=5, help='Dimension of memory')

	parser.add_argument('-num_par', type=int, default=2, help='Dyck-n or abc..')
	parser.add_argument('-p_val', type=float, default=0.5, help='P val of CFG for Dyck')
	parser.add_argument('-q_val', type=float, default=0.25, help='Q val of CFG for Dyck')

	parser.add_argument('-crl_n', type=int, default=1, help='CRL-n')

	parser.add_argument('-generate', dest='generate', action='store_true', help='Generate Data')
	parser.add_argument('-no-generate', dest='generate', action='store_false', help='load data')
	parser.set_defaults(generate=False)

	parser.add_argument('-leak', dest='leak', action='store_true', help='leak Data')


	parser.add_argument('-bins', type=int, default=2, help='Number of validation bins')

	## Generate Data parameters
	#parser.add_argument('-bins', type=int, default=2, help='Number of validation bins')
	parser.add_argument('-bin1_lower_window', type = int, default = 52)
	parser.add_argument('-bin1_upper_window', type = int, default = 100)
	parser.add_argument('-bin1_lower_depth', type = int, default = 0)
	parser.add_argument('-bin1_upper_depth', type = int, default = -1)
	parser.add_argument('-len_incr', type = int, default = 50)
	parser.add_argument('-depth_incr', type = int, default = 5)
	parser.add_argument('-vary_len', action = 'store_true')
	parser.add_argument('-vary_depth', action = 'store_true')



	return parser
