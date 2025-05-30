import random

def tokenise(text):
    return [ord(c) for c in text]

def detokenise(tokens):
    return "".join([chr(c) for c in tokens])

def gen_summands(length):
    return [random.randint(1,9999) for i in range(length)]

def gen_order(length):
    return [(random.random() < 0.5) for i in range(length-2)]

def gen_equation(length = None, summands = None, order = None):
    if length == None:
        length = random.randint(2,4)

    if summands == None:
        summands = gen_summands(length)
    
    if order == None:
        order = gen_order(len(summands))

    text = str(summands[0]) + "+" + str(summands[1])
    for i in range(2,len(summands)):
        text = "(" + text + ")"
        if order[i-2]:
            text = text + "+" + str(summands[i])
        else:
            text = str(summands[i]) + "+" + text
    
    question = ":" + text + "="
    full = question + str(sum(summands)) + ";"

    return question, full

def hash_summands(summands):
    return (sum([sum([int(c) for c in str(num)]) for num in summands]) + sum([int(c) for c in str(sum(summands))])) % 7

def hash_equation(text):
    return sum([int(c) for c in text if c.isdigit()]) % 7

def gen_train_tokens():
    while True:
        _, text = gen_equation()

        if hash_equation(text) != 0:
            break

    yield from tokenise(text)

def gen_eval_summands(length = None):
    if length == None:
        length = random.randint(2,4)

    while True:
        summands = gen_summands(length)
        if hash_summands(summands) == 0:
            break

    return summands

def gen_eval_problem(length = None, summands = None, order = None):
    summands = gen_eval_summands(length = length)
    return gen_equation(summands = summands, order = order)

# model: function completing problem string -> solution string
def eval_model(model, eval_iters=1000):
    total_count = [0,0,0]

    for i in range(eval_iters):
        summands = gen_eval_summands(length = 2)
        eqn_prob, eqn_full = gen_equation(summands = summands)
        _, permuted_eqn_full = gen_equation(summands = [summands[1],summands[0]])
        _, extra_eqn_full = gen_eval_problem(length = 2)
        
        inputs = [eqn_prob, permuted_eqn_full + eqn_prob, extra_eqn_full + eqn_prob]
        responses = [detokenise(model(tokenise(inp))) for inp in inputs]
        expected_responses = [eqn_full, permuted_eqn_full + eqn_full, extra_eqn_full + eqn_full]

        for j in range(len(responses)):
            if responses[j] == expected_responses[j]:
                total_count[j] += 1
    
    return "".join([" " + str(c) for c in total_count])[1:]