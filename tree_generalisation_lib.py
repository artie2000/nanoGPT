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

def gen_tree(depth = 0, depth_probs = [1,0.8,0.6,0.4,0.2,0.1,0], var_depth_mult = 0.5):
    if random.random() < depth_probs[depth]:
        return {"var" : None,
                "left" : gen_tree(depth + 1, depth_probs=depth_probs,var_depth_mult=var_depth_mult),
                "right" : gen_tree(depth + 1, depth_probs=depth_probs,var_depth_mult=var_depth_mult)}
    else:
        k = 0
        while random.random() < var_depth_mult:
            k += 1
        return {"var" : k, "left" : None, "right" : None}

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

def gen_tree_with_subs(k = 10):
    tree = gen_tree()
    vars = get_tree_vars(tree)
    derived = [substitute(tree, {var: gen_tree(depth_probs = [0.75,0.5,0.25,0]) for var in vars})
               for _ in range(k)]
    return tree, derived

def gen_text(tree, derived):
    text_start = "^" + "".join([tree_to_text(der) + "," for der in derived])[:-1] + ";"
    text_full = text_start + tree_to_text(tree) + "$"
    return text_start, text_full

# parse and check (see gen_text for format)
# tree: the base tree, derived: the substitutions of the base tree
def check_generalisation(tree, derived, text_full):
    try:
        gen = text_to_tree(text_full[1:-1].split(";")[1])
    except Exception as e: # parsing error
        return False, str(e)

    if get_substitution(tree, gen)[0]:
        if all([get_substitution(gen, der)[0] for der in derived]):
            return True, None
        else:
            return False, "Not a generalisation of the derived trees"
    else:
        return False, "Not a substitution of the base tree"

def gen_train_tokens():
    while True:
        tree, derived = gen_tree_with_subs()
        _, text_full = gen_text(tree, derived)

        if hash_tree(tree) != 7:
            break

    yield from tokenise(text_full)

def gen_eval_problem():
    while True:
        tree, derived = gen_tree_with_subs()
        text_start, _ = gen_text(tree, derived)

        if hash_tree(tree) == 7:
            break

    return tree, derived, text_start

# model: function completing problem string -> solution string
def eval_model(model, eval_iters=1000):
    count = 0

    for i in range(eval_iters):
        tree, derived, text_start = gen_eval_problem()
        if check_generalisation(tree, derived, detokenise(model(tokenise(text_start))))[0]:
            count += 1

    return str(count)