from iGSM_link import *
from sample import generate

iters = 1000
c = 0

for i in range(iters):
    id_gen = gen_problem("med", "eval")
    prompt = [222] + id_gen.prob_token + [223]
    response = tokenizer.decode(generate(prompt))
    response = response[len(tokenizer.decode(prompt)):-1] # clip to solution only
    correct, error, _ = true_correct(response, id_gen.problem)
    if correct:
        c += 1
    else:
        error.display()
    print(str(c) + " out of " + str(i))

print("Accuracy: " + str(c//iters))