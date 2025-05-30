from sample import *

bench_data_str = eval_fn(lambda inp : generate(inp, stop_token=eval_stop_token),eval_iters=100)
        
with open("eval-"+out_name+".txt", 'w') as out_file:
    out_text = bench_data_str
    print(out_text)
    out_file.write(out_text+"\n")