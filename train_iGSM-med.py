# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py

from iGSM_link import gen_problem

# TODO : save location in file for restart from checkpoint

# data
def data_gen():
    with open('iGSM-med_data_tokenised.txt', 'r') as reader:
        for line in reader:
            yield int(line)

'''import time
t0 = time.time()
for i in range(10):
    def foo():
        while True:
            yield from data_gen()
    it = foo()
    for i in range(768 * 64):
        next(it)
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    print(dt)'''

data_iter = data_gen
batch_size = 64
block_size = 768
gradient_accumulation_steps = 1 * 8 # used to simulate larger batch sizes

# model
n_layer = 6 # should be 12
n_head = 6 # should be 12
n_embd = 384 # should be 768

# eval stuff
eval_interval = 1000
eval_iters = 200
log_interval = 10
out_dir = 'out' # checkpointing

# optimizer
learning_rate = 0.002 # max learning rate
max_iters = 1000 # total number of training iterations -- should be 100_000
weight_decay = 0.05
beta1 = 0.9
beta2 = 0.98

# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 1000 # how many steps to warm up for
lr_decay_iters = 100000 # should be ~= max_iters per Chinchilla
min_lr = 0.00002 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla