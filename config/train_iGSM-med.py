# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

import sys
sys.path.insert(0, '/home/s2668504/iGSM')

from data_gen.pretrain.id_gen import IdGen
from tools.tools import tokenizer, fix_seed
from typing import Literal

def get_prob_sol_ans_triple(tpy: Literal["med", "hard"]):
    assert tpy in ["med", "hard"], "Invalid type: Choose 'med' or 'hard'"
    # Set parameters based on difficulty
    max_op = 15 if tpy == "med" else 21
    max_edge = 20 if tpy == "med" else 28

    id_gen = IdGen(
        max_op=max_op,        # Maximum # of operations
        max_edge=max_edge,    # Maximum # of edges (instance parameters) in the structure graph
        perm_level=5,         # Random shuffle level for problem description
        detail_level=0        # Most detailed solution format
    )

    id_gen.gen_prob([i for i in range(23)], p_format="pq")

    return id_gen

# data
def data_gen():
    id_gen = get_prob_sol_ans_triple("med")
    yield from id_gen.token_id

data_iter = data_gen
batch_size = 64
block_size = 768
gradient_accumulation_steps = 1 * 8 # used to simulate larger batch sizes

# model
n_layer = 12
n_head = 12
n_embd = 768

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10
out_dir = 'out' # checkpointing

# optimizer
learning_rate = 0.002 # max learning rate
max_iters = 10000 # total number of training iterations -- should be 100000
weight_decay = 0.05
beta1 = 0.9
beta2 = 0.98

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 1000 # how many steps to warm up for
lr_decay_iters = 100000 # should be ~= max_iters per Chinchilla
min_lr = 0.00002 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla