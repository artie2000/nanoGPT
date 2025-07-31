# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
from tree_generalisation_lib import gen_train_tokens, eval_model

# data
def data_gen_on_demand():
    while True:
        yield from gen_train_tokens(prop_correct_fn = lambda: 1)

out_name = "tree-tiny-accurate"
eval_fn = lambda model: eval_model(model, prop_correct = 1)
eval_stop_token = ord("$")
data_iter = data_gen_on_demand
batch_size = 16
block_size = 768
gradient_accumulation_steps = 1 * 8 # used to simulate larger batch sizes

# model
n_layer = 4
n_head = 4
n_embd = 256

# eval stuff
eval_interval = 100
eval_iters = 100
log_interval = 10
out_dir = 'out' # checkpointing

# optimizer
learning_rate = 0.002 # max learning rate
max_iters = 10_000 # total number of training iterations
weight_decay = 0.05
beta1 = 0.9
beta2 = 0.98

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 100 # how many steps to warm up for
lr_decay_iters = 10_000 # should be ~= max_iters per Chinchilla
min_lr = 0.0002 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla