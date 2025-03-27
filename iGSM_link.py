import sys
sys.path.insert(0, '/home/s2668504/iGSM')

from math_gen.problem_gen import Problem
from data_gen.prototype.id_gen import IdGen_PT
from const.params import test_bin, train_bin
from tools.tools_test import true_correct
from tools.tools import tokenizer, fix_seed
from typing import Literal

class IdGen(IdGen_PT):
    def __init__(self, max_op=10, max_edge=15, op=None, perm_level: str = None, detail_level: str = None, be_shortest: bool=True, op_style="light") -> None:
        super().__init__('light', op_style, max_op, max_edge, op, perm_level, detail_level, be_shortest)
    
    def gen_prob(self, ava_hash, p_format: str, problem: Problem=None):
        super().gen_prob(ava_hash, p_format, problem=problem)

def gen_problem(tpy: Literal["med", "hard"], mode: Literal["train","eval"]):
    assert tpy in ["med", "hard"], "Invalid type: Choose 'med' or 'hard'"
    assert mode in ["train", "eval"], "Invalid type: Choose 'train' or 'eval'"

    # Set parameters based on args
    max_op = 15 if tpy == "med" else 21
    max_edge = 20 if tpy == "med" else 28
    op = max_op if mode == "eval" else None
    hashes = test_bin if mode == "eval" else train_bin

    id_gen = IdGen(
        max_op=max_op,        # Maximum # of operations
        op=op,                # Fix number of operations (if not None)
        max_edge=max_edge,    # Maximum # of edges (instance parameters) in the structure graph
        perm_level=5,         # Random shuffle level for problem description
        detail_level=0        # Most detailed solution format
    )

    id_gen.gen_prob(hashes, p_format="pq")

    return id_gen