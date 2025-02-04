
import random as r

# d : number of layers
# e : number of edges
# w0, w1 : min, max vertices per layer
# return : digraph given by 2D array (layer, vertex) -> (set of child vertices)
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
    G = [[{r.randrange(l[i+1])} if i<d-1 else {} for j in range(l[i])] for i in range(d)]

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

# given structure graph G of depth d, parameters encoded as follows:
# instance parameter : (l < d, i < G[l], j in G[l][i]) [so (l+1,j) is a child of (l,i)]
# abstract parameter of difficulty i : (l, l + i, j < G[l]) [encoding total at layer l+i for (l,j)]

# G_s : structure graph
# n : operation bound for abstract step
# m : operation bound for instance step
# return : digraph given by set (parameter, set of parent vertices)
def DrawNecessary1(G_s, n, m):
    d = len(G_s)
    G_n = {}

    # enumerate abstract parameters by difficulty
    abs = [[(l,l+i,j) for l in range(1,d-i) for j in range(len(G_s[l]))] for i in range(1,d)]

    updated = True
    while not updated:
        updated = False
        for i in range(d-1,0,-1):
            abs_i = [e for e in abs[i] if not e in G_n]
            if len(abs_i)>0:
                AddAbstractParam(G_s, r.choice(abs_i))