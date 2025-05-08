import sys

init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
num_samples = 1 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 1e-5 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 1 # retain only the top_k most likely tokens, clamp others to have 0 probability
compile = True # use PyTorch 2.0 to compile the model to be faster