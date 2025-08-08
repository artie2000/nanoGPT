from sample import *

with open("eval-"+out_name+".txt", 'w') as out_file:
    for length in range(1,40):
        bench_data_str = eval_fn(lambda inp : generate(inp, stop_token=eval_stop_token), 
            eval_iters=1000, length=length)
        out_text = str(length) + " " + bench_data_str
        print(out_text)
        out_file.write(out_text+"\n")