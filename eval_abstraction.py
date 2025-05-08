from sample import generate
from abstraction_lib import *

iters = 1000
total_count = [0,0,0]

for i in range(iters):
    summands = gen_eval_summands(length = 2)
    eqn_prob, eqn_full = gen_equation(summands = summands)
    _, permuted_eqn_full = gen_equation(summands = [summands[1],summands[0]])
    _, extra_eqn_full = gen_eval_problem(length = 2)
    
    responses = [generate(eqn_prob,stop_token=ord(";")), generate(permuted_eqn_full + eqn_prob,stop_token=ord(";")), generate(extra_eqn_full + eqn_prob,stop_token=ord(";"))]
    expected_responses = [eqn_full, permuted_eqn_full + eqn_full, extra_eqn_full + eqn_full]

    for i in range(len(responses)):
        if detokenise(responses[i]) == expected_responses[i]:
            total_count[i] += 1

print("Accuracy: " + "".join([str(c/float(iters)) + "; " for c in total_count]))