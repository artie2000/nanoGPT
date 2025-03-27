from iGSM_link import gen_problem
import os.path
from multiprocessing import Process

os.environ["OPENBLAS_MAIN_FREE"] = "1"

num_processes = 8 # number of parallel processes to start

def problem_token_stream():
    while True:
        yield gen_problem("med", "train").token_id

def write_data(file_id):
    path = "iGSM-med_data_tokenised_" + file_id + ".txt"
    if os.path.isfile(path):
        print("File " + path + " already exists, aborting.")
        return

    with open(path, 'w') as writer:
        gen = problem_token_stream()
        for token_list in gen:
            try:
                output = "".join((str(t)+"\n" for t in token_list))
                writer.writelines(output)
            finally:
                pass
                # TODO: ensure correct state (ie delete excess data)

print(__name__)
if __name__ == '__main__':
    processes = [Process(target=write_data, args=(str(i),)) for i in range(num_processes)]

    for p in processes:
        print("hi" + str(p))
        p.start()
    
    try:
        for p in processes:
            p.join()
        
    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()

        for p in processes:
            if p.is_alive():
                p.join()