import sys
from sample import *

test_file_name = "eval"

with open(test_file_name+"-custom-in.txt",'r') as inp_file:
    with open(test_file_name+"-custom-out.txt",'w') as out_file:
        while line := inp_file.readline():
            out = "".join([chr(c) for c in generate([ord(c) for c in line], stop_token=eval_stop_token)])
            print(out)
            out_file.write(out+"\n")