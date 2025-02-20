
import random as r
import math
from collections import namedtuple
import json
import string
import time

f = open("nanoGPT/data_gen/world_knowledge.json")
world_knowledge = list(json.load(f).items())

# structure graph : 2D array (layer, entry -> set of child vertices)

# e : number of edges
# d : number of layers
# w0, w1 : min, max vertices per layer
# return : structure digraph
# d,w0,w1 in {2,3,4}
# w0 <= w1
# (d-1)*w0 <= e <= (d-1)*(w1**2)
def DrawStructure(e, d, w0, w1):
    l = [w0 for x in range(d)]
    p = r.uniform(0,1)

    # generate vertex counts
    while min(l) != w1:
        if sum([l[i] * l[i+1] for i in range(d-1)]) < e:
            l[r.randrange(d)] += 1
        elif e == sum([l[i] for i in range(d-1)]): break
        elif r.uniform(0,1) < p:
            l[r.choice([i for i in range(d) if l[i] < w1])] += 1
        else: break

    # generate minimal graph
    G = [[set() for j in range(l[i])] for i in range(d)]
    [G[i-1][r.randrange(l[i-1])].add(j) for i in range(1,d) for j in range(l[i])]

    # add extra edges
    e_cur = sum([l[i] for i in range(1,d)])
    while e_cur < e:
        i = r.randrange(0,d-1)
        a = r.randrange(l[i])
        bs = [b for b in range(l[i+1]) if not (b in G[i][a])]
        if len(bs)>0 :
            e_cur += 1
            G[i][a].add(r.choice(bs))
    
    return G

# given structure graph G_s, parameters encoded as follows:
# instance parameter : (layer l, index i, child j) [number of (l+1,j) in parent (l,i)]
# abstract parameter : (layer l, index i, diff k) [number of layer l+k in (l,i)]
# val, op_type, bin_op_type params are for word problem construction

InstParam = namedtuple("InstParam", ["layer", "index", "child", "type"], defaults = ("inst",))
AbsParam = namedtuple("AbsParam", ["layer", "index", "diff", "type"], defaults = ("abs",))
RNGParam = namedtuple("RNGParam", ["type"], defaults = ("rng",))

# dependency graph : dictionary (parameter -> set of parents)

# G_n : dependency graph
# return : operation count of G_n
def OperationCount(G_n):
    return sum([max(len(G_n[p])-1,1) for p in G_n if p.type != "rng"])

# G_n : dependency graph
# G_s : structure graph
# p : AbsParam
# op_rem : remaining operations budget
# limit_ops : whether to fail upon running out of operations
# augments G_n with all parameters necessary for p as well as their dependency edges
# return : remaining operations budget (or -1 on failure)
def AddAbstractParam(G_n, G_s, p, op_rem = 0, limit_ops = True):
    if p in G_n :
        return op_rem

    G_n[p] = set()
    children = G_s[p.layer][p.index]

    # update operations budget
    w = len(children)
    if w == 0: return op_rem - 1 # trivial case
    op_rem -= w + max(1, w + (0 if p.diff == 1 else w) - 1) # count (instance param ops -- always 1) + (p ops)

    # failure check
    if limit_ops and op_rem < 0:
        return -1

    for child_index in children:
        # add instance parameters
        inst_child = InstParam(p.layer, p.index, child_index)
        G_n[p].add(inst_child)
        if inst_child not in G_n:
            G_n[inst_child] = set()
        
        # recursively add abstract parameters
        if p.diff > 1:
            abs_child = AbsParam(p.layer + 1, child_index, p.diff - 1)
            G_n[p].add(abs_child)
            op_rem = AddAbstractParam(G_n, G_s, abs_child, op_rem=op_rem, limit_ops=limit_ops)
            
            # recursive failure check
            if limit_ops and op_rem == -1:
                return -1
    
    return op_rem
        

# G_s : structure graph
# n : operation bound for abstract step
# m : operation count for instance step
# return : dependency graph with minimal edge set (or None on failure)
def DrawNecessary1(G_s, n, m):
    d = len(G_s)
    G_n = {}

    # enumerate parameters
    abs = [[AbsParam(l, j, i) for l in range(d-i) for j in range(len(G_s[l]))] for i in range(1,d)]
    inst = [InstParam(l, j, c) for l in range(d) for j in range(len(G_s[l])) for c in G_s[l][j]]

    # add abstract parameters and their edges
    updated = True
    while updated:
        updated = False
        for i in range(d-1,0,-1):
            abs_i = [e for e in abs[i-1] if not e in G_n] # abstract parameters of difficulty i not yet added
            if len(abs_i)>0:
                G_n_copy = G_n.copy()
                op_rem = AddAbstractParam(G_n_copy, G_s, r.choice(abs_i), op_rem=n)
                if op_rem >= 0:
                    n = op_rem
                    updated = True
                    G_n = G_n_copy
                    break

    # add extra instance parameters
    inst_rem = [p for p in inst if not p in G_n]
    diff = m-OperationCount(G_n)
    if len(inst_rem) < diff: return None # failure
    G_n.update([(p, set()) for p in r.sample(inst_rem, diff)])

    return G_n

# params : list of parameters
# s : "special" set
# return : random element of params biased towards abstract parameters and parameters in s
def PickBiasedRandom(params, s):
    w = [math.exp(((1 if p.type == "abs" else 0) + (1 if p in s else 0)) * abs(r.gauss(0,1))) for p in params]
    return r.choices(params,
        weights = w)[0]

# G_n : dependency graph
# return : reverse topological ordering of graph
# modifies G_n to add extra dependency edges between instance parameters
def DrawNecessary2(G_n):
    topo = []
    next_params = set()
    avail_params = {p for p in G_n if all(p not in G_n[q] for q in G_n)}

    next_param = r.choice(tuple(avail_params))
    topo.append(next_param)

    for i in range(len(G_n) - 1):
        # update parameter sets
        avail_params.remove(next_param)
        avail_params.update({p for p in G_n[next_param] if all(p not in G_n[q] for q in G_n if q not in topo)})
        next_params.update(G_n[next_param])

        # add extra dependencies
        if len(avail_params & next_params) == 0:
            if next_param.type == "abs": return None # fail
            p = PickBiasedRandom(list(avail_params), set())
            G_n[next_param].add(p)
            next_params.add(p)
        elif next_param.type == "inst" and r.uniform(0,1) > r.uniform(0,1):
            p = PickBiasedRandom(list(avail_params), next_params)
            G_n[next_param].add(p)
            next_params.add(p)

        # choose next parameter
        next_param = r.choice(tuple(avail_params & next_params))
        next_params.remove(next_param)
        topo.append(next_param)
        
    return topo

# pool : set of parameters
# k : number of parameters to choose
# returns a set of k random parameters from the pool possibly including RNG
def ChooseParamsFromPool(pool, k):
    ret = set()
    if len(pool) + 1 == k:
        ret.update(pool)
        ret.add(RNGParam())
    elif k > 0:
        if r.uniform(0,1) > 0.5:
            ret.add(RNGParam())
            k -= 1
        ret.update(r.sample(list(pool), k))
    return ret


# G_n : dependency graph
# topo : topological order on G_n
# s : final operation count
# adds dependency edges to G_n so that it has operation count s
def DrawNecessary3(G_n, topo, s):
    cur_op = [max(len(G_n[topo[i]])-1,1) for i in range(len(topo))]
    max_op = [min(3,max(1,len(topo)-i-1)) for i in range(len(topo))]
    while sum(cur_op) < s:
        avail_inds = [i for i in range(len(topo)) if topo[i].type == "inst" and cur_op[i] < max_op[i]]
        if len(avail_inds) == 0:
            G_n = None
            return # failure
        cur_op[r.choice(avail_inds)] += 1

    for i in range(len(topo)):
        if topo[i].type == "inst":
            pool = set(topo[i+1:])
            if cur_op[i] == 1:
                dep_num = r.choice((1,2))
            else:
                dep_num = cur_op[i] + 1
            dep_num = min(len(pool) + 1, dep_num)
            for p in G_n[topo[i]]:
                if p in pool:
                    pool.remove(p)
                    dep_num -= 1
            G_n[topo[i]].update(ChooseParamsFromPool(pool, dep_num))

# G_s : structure graph
# params : set of parameters
# abs : abstract parameter
# return : whether abs is computable from the instance parameters in params
def IsComputableAbs(G_s, params, abs):
    ret = True
    if abs.diff > 1:
        ret = all([IsComputableAbs(G_s, params, AbsParam(abs.layer + 1, j, abs.diff - 1)) for j in G_s[abs.layer][abs.index]])
    return ret and all([InstParam(abs.layer, abs.index, j) in params for j in G_s[abs.layer][abs.index]])

# G_s : structure graph
# G_n : dependency graph
# return : G_n with unneccesary dependency edges added
def DrawUnnecessary(G_s, G_n):
    G_u = G_n.copy()

    # enumerate parameters
    d = len(G_s)
    abs = [AbsParam(l, j, i) for i in range(1,d) for l in range(d-i) for j in range(len(G_s[l]))]
    inst_rem = [InstParam(l, j, c) for l in range(d) for j in range(len(G_s[l])) for c in G_s[l][j]]

    new_params = set()

    while True:
        # choose new instance parameter to add
        abs_poss = [p for p in abs if p in G_u or IsComputableAbs(G_s, G_u.keys(), p)]
        inst_rem = [p for p in inst_rem if p not in G_u]
        if len(inst_rem) == 0: break
        new_param = r.choice(inst_rem)

        if r.uniform(0,1) > 0.5:
            pool = new_params.copy()
            new_params.add(new_param)
        else:
            pool = set(abs_poss).union(G_u.keys())
                
        dep_num = 1
        while dep_num < min(4, len(pool)+1):
            if r.uniform(0,1) > 0.5:
                dep_num += 1
            else:
                break
        
        selected = ChooseParamsFromPool(pool, dep_num)
        G_u[new_param] = selected
        for p in selected:
            if p not in G_u:
                if p.type == "inst":
                    G_u[p] = set()
                elif p.type == "abs":
                    AddAbstractParam(G_u, G_s, p, limit_ops = False)
    
    return G_u

# G_s : structure graph
# world_knowledge = world knowledge (4x4x5x20 array of strings)
# return : array (layer in G_s) -> (name reference), array (entry in G_s) -> (name reference)
def AttachNameData(G_s, world_knowledge):
    d = len(G_s)
    category = r.randint(0,3)
    start_layer = r.randint(0,4-d)
    layer_names = [k for (k,v) in world_knowledge[4*category+start_layer : 4*category+start_layer+d]]
    
    names = [None] * d
    
    for i in range(d):
        idx = 4*category+start_layer+i
        name_list = r.choice(list(world_knowledge[idx][1].values()))
        names[i] = r.sample(name_list, len(G_s[i]))

    return layer_names, names


# labels : data from AttachNameData
# p : parameter
# return : English name of parameter
def ParamName(labels, p):
    if p.type == "rng" : return ""

    first = labels[1][p.layer][p.index]
    if p.type == "inst":
        second = labels[1][p.layer + 1][p.child]
    elif p.type == "abs":
        second = labels[0][p.layer + p.diff]
    return first + "'s " + second

# G_s : structure graph
# G : dependency graph
# labels : data from AttachNameData
# return : dict (parameter -> English name)
def AttachNames(G_s, G, world_knowledge):
    data = AttachNameData(G_s, world_knowledge)
    return {p : ParamName(data, p) for p in G}

# names : data from AttachNames
# G : dependency graph
# p : parameter
# return : English sentence defining p, data about random choices made
def GenParamProblemText(names, G, p):
    val, op, bin_op, pool = None, None, None, G[p].copy()

    out = "The number of each " + names[p] + " equals"
    for q in pool:
        if q.type == "rng":
            val = r.randrange(23)
            op = r.getrandbits(1)
            out += " " + str(val)
            if len(G[p]) > 1:
                out += (" more than" if op else " times")
            pool.remove(q)
            break
    
    pool = list(pool)
    r.shuffle(pool)
    n = len(pool)
    if n == 1:
        out += " each " + names[pool[0]]
    elif n == 2:
        bin_op = r.getrandbits(1)
        out += " the " + ("sum" if bin_op else "difference") + " of each " + names[pool[0]] + " and each " + names[pool[1]]
    elif n > 2:
        out += " the sum of "
        for i in range(n-1):
            out += "each " + names[pool[i]] + ", "
        out += "and each " + names[pool[n-1]]
    out += "."

    return out, (val, op, bin_op, pool)

# names : data from AttachName
# G : dependency graph
# query : query parameter
# return : text for the problem of finding query, dict (parameter -> data about random choices made)
def GenProblemText(names, G, query):
    out = {p: GenParamProblemText(names, G, p) for p in G if p.type == "inst"}
    sentences = [s for s, _ in out.values()]
    rand_data = ({p: d[i] for p, (_, d) in out.items()} for i in range(4))
    r.shuffle(sentences)

    # parse query name into question
    first, second = names[query].split("'s ")
    sentences.append("How many " + second + " does " + first + " have?")

    return " ".join(sentences), rand_data

# sum_var : the variable for the sum
# summand_vars : list of summand variables
# values : list of values of summed variables
# avail_vars : shuffled list of available variables
# return : integer sum mod 23, list of clauses for sum computation by binary ops, remaining variables
def GenSumText(sum_var, summand_vars, values, avail_vars):
    l = len(summand_vars)
    clauses = []

    if l == 0:
        final_val = 0
        clauses.append(sum_var + " = 0")
    elif l == 1:
        final_val = values[0]
        clauses.append(sum_var + " = " + summand_vars[0] + " = " + str(final_val))
    else:
        temp_var = (avail_vars.pop() if l > 2 else sum_var)
        temp_val = (values[0] + values[1]) % 23
        clauses.append(temp_var + " = " + summand_vars[0] + " + " + summand_vars[1] + " = " + str(values[0]) + " + " + str(values[1]) + " = " + str(temp_val))

        for i in range(2,l):
            old_temp_var = temp_var
            old_temp_val = temp_val
            temp_var = (avail_vars.pop() if i < l-1  else sum_var)
            temp_val = (temp_val + values[i]) % 23
            clauses.append(temp_var + " = " + old_temp_var + " + " + summand_vars[i] + " = " + str(old_temp_val) + " + " + str(values[i]) + " = " + str(temp_val))
        final_val = temp_val

    return final_val, clauses, avail_vars

# names : data from AttachName
# rand_data : random data from GenProblemText
# G : dependency graph
# topo : list encoding solution as a topological order (query / root is first)
# return : text for the solution to finding the query
def GenSolutionText(names, rand_data, G, topo):
    rng_val, op, bin_op, pool = rand_data
    avail_vars = [k for k in string.ascii_lowercase + string.ascii_uppercase] # pool of variable names
    r.shuffle(avail_vars)
    var_vals = {}
    param_vars = {}

    l = len(topo)
    output = ""

    for p in reversed(topo):
        param_vars[p] = avail_vars.pop()
        clauses = []

        if p.type == "abs":
            deps = list(G[p])
            r.shuffle(deps) # ensure order children are summed over is random

            if p.diff == 1:
                summand_vars = [param_vars[q] for q in deps]
            else:
                summand_vars = []
                for q in deps:
                    if q.type == "inst":
                        temp_var = avail_vars.pop()
                        summand_vars.append(temp_var)
                        var0, var1 = param_vars[q], param_vars[AbsParam(q.layer + 1, q.child, p.diff - 1)]
                        var_vals[temp_var] = (var_vals[var0] * var_vals[var1]) % 23
                        clauses.append(temp_var + " = " + var0 + " * " + var1 + " = " + str(var_vals[var0]) + " * " + str(var_vals[var1]) + " = " + str(var_vals[temp_var]))
            
            var_vals[param_vars[p]], temp_clauses, avail_vars = GenSumText(param_vars[p], summand_vars, [var_vals[v] for v in summand_vars], avail_vars)
            clauses.extend(temp_clauses)
        
        elif p.type == "inst":
            deps = pool[p]
            n = len(deps)

            has_rng = (n < len(G[p])) # whether or not p depends on RNG
            if has_rng and n > 1:
                final_var = avail_vars.pop()
            elif has_rng and n == 1:
                final_var = param_vars[deps[0]]
            else:
                final_var = param_vars[p]

            if n == 2 and not bin_op[p]:
                val0, val1 = var_vals[param_vars[deps[0]]], var_vals[param_vars[deps[1]]]
                var_vals[final_var] = (val0 - val1) % 23
                clauses.append(
                    final_var + " = " + param_vars[deps[0]] + " - " + param_vars[deps[1]] + " = " + str(val0) + " - " + str(val1) + " = " + str(var_vals[final_var]))
            elif n > 1 or not has_rng:
                summand_vars = [param_vars[q] for q in deps]
                var_vals[final_var], temp_clauses, avail_vars = GenSumText(final_var, summand_vars, [var_vals[v] for v in summand_vars], avail_vars)
                clauses.extend(temp_clauses)
            
            if has_rng:
                if n == 0:
                    var_vals[param_vars[p]] = rng_val[p]
                    clauses.append(final_var + " = " + str(rng_val[p]))
                else:
                    if op[p]:
                        var_vals[param_vars[p]] = (rng_val[p] + var_vals[final_var]) % 23
                        clauses.append(
                            param_vars[p] + " = " + str(rng_val[p]) + " + " + final_var + " = " + str(rng_val[p]) + " + " + str(var_vals[final_var]) + " = " + str(var_vals[param_vars[p]]))
                    else:
                        var_vals[param_vars[p]] = (rng_val[p] * var_vals[final_var]) % 23
                        clauses.append(
                            param_vars[p] + " = " + str(rng_val[p]) + " * " + final_var + " = " + str(rng_val[p]) + " * " + str(var_vals[final_var]) + " = " + str(var_vals[param_vars[p]]))

        output += "Define " + names[p] + " as " + param_vars[p] + "; "

        for clause in clauses[:-1]:
            output += clause + "; "
        output += "so " + clauses[-1] + ". "
    
    output += "Answer: " + str(var_vals[param_vars[topo[0]]])
    return output

# op_max : maximum number of operations in solution
# ip_max : maximum number of instance parameters
# force : if true, forces number of operations to equal op_max
# return : sample problem in the form (full dependency graph, minimal dependency graph, topological ordering of minimal graph)
def DrawAll(op_max, ip_max, force):
    success = False

    while True:
        s = op_max if force else min(r.randint(1,op_max), r.randint(1,op_max))
        n = max(r.randint(1,s), r.randint(1,s))
        m = r.randint(n,s)
        rel = (s-1) / (ip_max-1)
        weights = [math.exp(-(rel-0.2)**2), math.exp(-(rel-0.5)**2), math.exp(-(rel-0.8)**2)]
        d, = r.choices([2,3,4], weights, k=1)
        t_0, t_1 = r.choices([2,3,4], weights, k=2)
        w_0, w_1 = min(t_0, t_1), max(t_0, t_1)
        e = min(r.randint((d-1)*w_0, ip_max), r.randint((d-1)*w_0, ip_max), (d-1)*(w_1**2))
        G_s = DrawStructure(e,d,w_0,w_1)

        topo = None
        for i in range(1000):
            G_n = DrawNecessary1(G_s,n,m)
            if G_n == None : continue
            topo = DrawNecessary2(G_n)
            if topo == None : continue
        
        if topo == None : continue # failed 1000 times
        DrawNecessary3(G_n, topo, s)
        if G_n != None : break # success
    
    G_u = DrawUnnecessary(G_s, G_n)
    names = AttachNames(G_s, G_u, world_knowledge)
    problem_text, rand_data = GenProblemText(names, G_u, topo[0])
    soln_text = GenSolutionText(names, rand_data, G_u, topo)
    return problem_text, soln_text

def iGSM_med_train_gen():
    # TODO: implement hash splitting?
    while True:
        problem_text, soln_text = DrawAll(15, 20, False)
        yield from problem_text
        yield '\n'
        yield from soln_text
        yield '\n'

# test - iGSM-med training data
for c in iGSM_med_train_gen():
    pass#print(c, end="")