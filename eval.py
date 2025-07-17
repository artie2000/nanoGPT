from sample import *

with open("eval-"+out_name+".txt", 'w') as out_file:
    for prop_correct in [0.5,0.6,0.7,0.8,0.9,1]:
        bench_data_str = eval_fn(lambda inp : generate(inp, stop_token=eval_stop_token), 
            eval_iters=1000, prop_correct=prop_correct)
        out_text = str(prop_correct) + " " + bench_data_str
        print(out_text)
        out_file.write(out_text+"\n")