# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

#data
dataset = 'iGSM-med'
batch_size = 64
block_size = 768
gradient_accumulation_steps = 1 * 8 # used to simulate larger batch sizes

# model
n_layer = 12
n_head = 12
n_embd = 768

# instantly checkpoint
eval_interval = 1
eval_iters = 1
out_dir = 'out'
max_iters = 1