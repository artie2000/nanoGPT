
import random as r
import math
from collections import namedtuple
import json

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
    while max(l) != w1:
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
    e_cur = sum([l[i] for i in range(d-1)])
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

InstParam = namedtuple('InstParam', ['layer', 'index', 'child', 'type'], defaults = ('inst',))
AbsParam = namedtuple('AbsParam', ['layer', 'index', 'diff', 'type'], defaults = ('abs',))
RNGParam = namedtuple('RNGParam', ['type'], defaults = ('rng',))

# dependency graph : dictionary (parameter -> set of parents)

# G_n : dependency graph
# return : operation count of G_n
def OperationCount(G_n):
    return sum([max(len(G_n[p])-1,1) for p in G_n if p.type != 'rng'])

# G_n : dependency graph
# G_s : structure graph
# p : AbsParam
# op_rem : remaining operations budget
# limit_ops : whether to fail upon running out of operations
# augments G_n with all parameters necessary for p as well as their dependency edges
# return : remaining operations budget (or -1 on failure)
def AddAbstractParam(G_n, G_s, p, op_rem = 0, limit_ops = True):
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
    updated = False
    while not updated:
        updated = False
        for i in range(d-1,0,-1):
            abs_i = [e for e in abs[i-1] if not e in G_n] # abstract parameters of difficulty i not yet added
            if len(abs_i)>0:
                G_n_copy = G_n.copy()
                success = AddAbstractParam(G_n_copy, G_s, r.choice(abs_i), op_rem=n)
                if success >= 0:
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
            if next_param.type == 'abs': return None # fail
            p = PickBiasedRandom(list(avail_params), set())
            G_n[next_param].add(p)
            next_params.add(p)
        elif next_param.type == 'inst' and r.uniform(0,1) > r.uniform(0,1):
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

# G_n : dependency graph
# G_s : structure graph
# return : G_n with unneccesary dependency edges added
def DrawUnnecessary(G_n, G_s):
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
def AttachName(G_s, world_knowledge):
    d = len(G_s)
    category = r.randint(0,3)
    start_layer = r.randint(0,4-d)
    layer_names = [k for (k,v) in world_knowledge[4*category+start_layer : 4*category+start_layer+d]]
    
    names = [None] * d
    print(names)
    for i in range(d):
        idx = 4*category+start_layer+i
        name_list = r.choice(list(world_knowledge[idx][1].values()))
        names[i] = r.sample(name_list, len(G_s[i]))

    return layer_names, names

# TODO : figure out structure for problem data (for instance params)

def GenProblemData(G_u): return

# labels : data from attach_name

def GenProblemText(labels, G_u, data): return

def GenSolutionText(labels, G_n, data): return

# test
G_s = DrawStructure(9,3,2,4)
print(G_s)
G_n = DrawNecessary1(G_s,10,10)
print(G_n)
topo = DrawNecessary2(G_n)
print(G_n)
print(topo)
DrawNecessary3(G_n, topo, 10)
print(G_n)
G_u = DrawUnnecessary(G_n, G_s)
print(G_u)
print(AttachName(G_s, world_knowledge))