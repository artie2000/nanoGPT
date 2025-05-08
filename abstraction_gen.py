from abstraction_lib import gen_train_tokens
import os.path
import sys

def problem_token_stream():
    while True:
        yield gen_train_tokens()

file_id = sys.argv[1]
print(file_id)
path = "abstraction_data_tokenised_" + file_id + ".txt"
print(path)

if os.path.isfile(path):
    print("File " + path + " already exists, aborting.")
else:
    try:
        with open(path, 'w') as writer:
            gen = problem_token_stream()
            for token_list in gen:
                output = "".join((str(t)+"\n" for t in token_list))
                writer.writelines(output)
    except Exception as e: print(e)
    finally:
        pass
        # TODO: ensure correct state (ie delete excess data)
