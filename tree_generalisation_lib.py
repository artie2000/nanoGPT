import random
import re
import hashlib

def tokenise(text):
    return [ord(c) for c in text]

def detokenise(tokens):
    return "".join([chr(c) for c in tokens])

def tree_to_text(tree):
    if (data := tree["var"]) != None:
        return str(data)
    else:
        left = tree["left"]
        left_txt = tree_to_text(left)
        if left["var"] == None:
            left_txt = "(" + left_txt + ")"
        
        right = tree["right"]
        right_txt = tree_to_text(right)
        if right["var"] == None:
            right_txt = "(" + right_txt + ")"

        return left_txt + ">" + right_txt

def text_to_tree(text):
    b = 0
    key_pos = None
    for i in range(len(text)):
        c = text[i]
        if c == "(":
            b += 1
        elif c == ")":
            b -= 1
        if b == 0 and c == ">":
            key_pos = i

    if text[0] == "(" and key_pos == None: # expression is bracketed
        return text_to_tree(text[1:-1])

    if not ">" in text:
        return {"var" : int(text), "left" : None, "right" : None}

    return {"var" : None,
            "left" : text_to_tree(text[:key_pos]),
            "right" : text_to_tree(text[key_pos+1:])}

def hash_tree(tree):
    text = tree_to_text(tree)

    # regularise variables
    vars = list(dict.fromkeys(re.findall(r"[0-9]+", text))) # deduplicate
    var_dict = {vars[i]:str(i) for i in range(len(vars))}
    reg_text = re.sub(r"[0-9]+", lambda m: var_dict[m.group()], text)

    # hash
    return int(hashlib.md5(reg_text.encode("utf-8")).hexdigest(), 16) % 10

def get_tree_vars(tree):
    if tree["var"] != None:
        return {tree["var"]}
    return get_tree_vars(tree["left"]) | get_tree_vars(tree["right"])

def geom(p):
    k = 0
    while random.random() < p:
        k += 1
    return k

def bal_brackets_to_tree(text, var_depth_mult = 0.5):
    if text == "": return {"var" : geom(var_depth_mult), "left" : None, "right" : None}

    b = 0
    key_pos = None
    for i in range(len(text)):
        c = text[i]
        if c == "(":
            b += 1
        elif c == ")":
            b -= 1
        if b == 0:
            key_pos = i
            break

    return {"var" : None,
            "left" : bal_brackets_to_tree(text[1:key_pos],var_depth_mult=var_depth_mult),
            "right" : bal_brackets_to_tree(text[key_pos+1:],var_depth_mult=var_depth_mult)}

# generate a uniformly random binary tree with n non-leaf nodes
# variable labels are generated with a geometric distibution, parameter var_depth_mult
def gen_tree_with_nodes(n, var_depth_mult = 0.5):
    left_set = set(random.sample(range(2*n), n))
    brackets = "".join(["(" if i in left_set else ")" for i in range(2*n)])

    i = 0
    start_idx = 0
    end_idx = 2*n
    bal = 0

    while i < end_idx:
        if brackets[i] == "(": bal += 1
        else: bal -= 1

        if bal == 0:
            if brackets[start_idx] == "(":
                start_idx = i + 1
            else:
                brackets = (
                    brackets[:start_idx] + 
                    "(" + brackets[i+1:end_idx] + ")" + 
                    "".join([")" if brackets[j] == "(" else "(" for j in range(start_idx+1,i)]) +
                    brackets[end_idx:])
                end_idx = end_idx - i + start_idx
                i = start_idx
                start_idx += 1
        
        i += 1
    
    return bal_brackets_to_tree(brackets, var_depth_mult=var_depth_mult)

def gen_tree(n=None,n_min=2,n_max=10):
    if n == None:
        n = random.randrange(n_min, n_max+1)
    return gen_tree_with_nodes(n-1)


# substitute base using the given variable -> tree dict
def substitute(base, vars):
    if base["var"] in vars.keys():
        return vars[base["var"]]
    if base["var"] != None:
        return base
    return {"var" : None,
            "left" : substitute(base["left"], vars),
            "right" : substitute(base["right"], vars)}

# determine if g is a substitution of f
# return (true or false, dict of substitutions)
def get_substitution(f, g):
    if f["var"] != None: # f leaf
        return True, {f["var"] : g}

    if g["var"] != None: # f not leaf, g leaf
        return False, None

    same_l, subs_l = get_substitution(f["left"], g["left"])
    same_r, subs_r = get_substitution(f["right"], g["right"])

    if same_l and same_r:
        compat = all([subs_l[var] == subs_r[var] for var in subs_l.keys() & subs_r.keys()])
        subs = subs_l | subs_r if compat else None
        return compat, subs

    return False, None

# return: tree, substitutions, indices of correct substitutions
def gen_tree_with_subs(k = 10, prop_correct = 1):
    n = random.randrange(2, 21)
    tree = gen_tree(n)
    vars = get_tree_vars(tree)
    num_correct = int(k * prop_correct)

    # correct substitutions
    derived = [substitute(tree, {var: gen_tree(n_min=1,n_max=7) for var in vars})
               for _ in range(num_correct)]
    # fake substitutions (length is roughly correct)
    derived.extend([gen_tree(n_min=2*n,n_max=5*n) for _ in range(k - num_correct)])
    
    # shuffle
    indices = list(range(k))
    random.shuffle(indices)
    derived = [derived[indices[i]] for i in range(k)]
    return tree, derived, {i for i in range(k) if indices[i] < num_correct}

def gen_text(tree, derived):
    text_start = "^" + "".join([tree_to_text(der) + "," for der in derived])[:-1] + ";"
    text_full = text_start + tree_to_text(tree) + "$"
    return text_start, text_full

# parse and check (see gen_text for format)
# tree: the base tree, derived: the substitutions of the base tree, correct_inds: the indices to check
def check_generalisation(tree, derived, correct_inds, text_full):
    try:
        gen = text_to_tree(text_full[1:-1].split(";")[1])
    except Exception as e: # parsing error
        return False, str(e)

    if get_substitution(tree, gen)[0]:
        if all([get_substitution(gen, derived[i])[0] for i in correct_inds]):
            return True, None
        else:
            return False, "Not a generalisation of the derived trees"
    else:
        return False, "Not a substitution of the base tree"

def gen_train_tokens(prop_correct_fn = lambda: random.uniform(0.5,1)):
    while True:
        prop_correct = prop_correct_fn()
        tree, derived, _ = gen_tree_with_subs(prop_correct = prop_correct)
        _, text_full = gen_text(tree, derived)

        if hash_tree(tree) != 7:
            break

    yield from tokenise(text_full)

def gen_eval_problem(prop_correct = 1):
    while True:
        tree, derived, correct_inds = gen_tree_with_subs(prop_correct = prop_correct)
        text_start, _ = gen_text(tree, derived)

        if hash_tree(tree) == 7:
            break

    return tree, derived, correct_inds, text_start

# model: function completing problem string -> solution string
def eval_model(model, eval_iters=1000, prop_correct = 0.6):
    count = 0

    for i in range(eval_iters):
        tree, derived, correct_inds, text_start = gen_eval_problem(prop_correct = prop_correct)
        if check_generalisation(tree, derived, correct_inds, detokenise(model(tokenise(text_start))))[0]:
            count += 1

    return str(count)