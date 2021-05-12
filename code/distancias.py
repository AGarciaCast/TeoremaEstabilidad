# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:45:56 2021.

@author: Alejandro
"""

# import complejosSimpliciales
import numpy as np
import math
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import gudhi



def distanciaEuclidea(p1, p2):
    """
    Distancia euclídea entre los puntos p1 y p2.

    p1: list.
    p2: list.
    """
    p1Np = np.array(p1)
    p2Np = np.array(p2)
    return np.sqrt(np.dot(p1Np - p2Np, p1Np - p2Np))

def distanciaInf(x, y):
    """
    Calcular la distancia infiniro entre los puntos x e y.

    x: numpy.ndarray.
    y: numpy.ndarray.
    """
    res = 0
    if x[1] == float('inf') and y[1] == float('inf'):
        res = abs(x[0]-y[0])
    elif x[1] ==  float('inf') and y[1] != float('inf'):
        res = float('inf')
    elif x[1] != float('inf') and y[1] == float('inf'):
        res = float('inf')
    else:
        res = max(abs(x[0]-y[0]), abs(x[1]-y[1]))

    return res


def hausdorffDir(A, B):
    """
    Calcular la distancia Hausdorff directa entre los conjuntos de puntos A y B.

    A: numpy.array.
    B: numpy.array.
    """
    cmax = 0
    for x in A:
        cmin = float('inf')
        for y in B:
            d = distanciaInf(x, y)
            if d < cmin:
                cmin = d

        if cmin > cmax:
            cmax = cmin

    return cmax


def hausdorff(A, B):
    """
    Calcular la distancia Hausdorff entre los conjuntos de puntos A y B.

    A: numpy.array.
    B: numpy.array.
    """
    return max(hausdorffDir(A, B),  hausdorffDir(B, A))


def grafoBottleneck(X, Y):
    X0 = list()
    X0_inf = list()

    Y0 = list()
    Y0_inf = list()
    for x in X:
        if x[0] != x[1]:
            if x[1] == float("inf"):
                X0_inf.append(x[0])
            else:
                X0.append((x[0], x[1]))

    for y in Y:
        if y[0] != y[1]:
            if y[1] == float("inf"):
                Y0_inf.append(y[0])
            else:
                Y0.append((y[0], y[1]))

    X0_ = [((x[0]+x[1])/2, (x[0]+x[1])/2) for x in X0]
    Y0_ = [((y[0]+y[1])/2, (y[0]+y[1])/2) for y in Y0]
    U = X0 + Y0_
    V = Y0 + X0_
    n = len(U)
    G = nx.Graph()
    G.add_nodes_from([(f"u{i}", {'coord': U[i]}) for i in range(0, n)], bipartite=0)
    G.add_nodes_from([(f"v{i}", {'coord': V[i]}) for i in range(0, n)], bipartite=1)
    edges = list()
    for i in range(0, n):
        for j in range(0, n):
            d = 0
            if U[i] in X0 or V[j] in Y0:
                d = distanciaInf(U[i], V[j])

            # if d != float('inf'):

            edges.append((f"u{i}", f"v{j}",d))

    G.add_weighted_edges_from(edges)

    distPinf = 0

    if len(X0_inf) != len(Y0_inf):
        distPinf = float("inf")
    else:
        distPinf = max([abs(x-y) for x,y in zip(sorted(X0_inf), sorted(Y0_inf))]+[0])

    return G, U, V, distPinf

def subgrafoG(G, i):
    G0 = nx.Graph([(u, v, d) for u, v, d in G.edges(data=True) if d['weight'] <= i])
    for n, d in G0.nodes(data=True):
        d["bipartite"] = G.nodes[n]["bipartite"]

    return G0

def cambiarPesos(G, Gi, dU, dV):
    for u, v, d in Gi.edges(data=True):
        d['weight'] = G[u][v]['weight'] - dU[u] - dV[v]


def digrafoAsociado(G, M, U = None, V = None):
    if U is None or V is None:
        U, V = bipartite.sets(G)

    D = nx.DiGraph()
    keysM = M.keys()
    D.add_weighted_edges_from([(u, v, d) if u not in keysM or M[u] != v
                               else (v, u, d)
                               for u, v, d in G.edges(data='weight')])


    D.add_weighted_edges_from([("s", u, 0) for u in U if u not in keysM])
    D.add_weighted_edges_from([(v, "t", 0) for v in V if v not in keysM])

    return D

def plotMatching(G, M, U):
    plt.figure(figsize=(8, 6))
    pos = nx.drawing.layout.bipartite_layout(G, U)
    nx.draw_networkx(G, pos=pos)
    edgelist=[(u,M[u]) for u in U if u in M.keys()]
    nx.draw_networkx_edges(G, pos, edgelist=edgelist, width=2.5, edge_color='blue')


def bottleneck(X, Y):
    G, _, _, distPinf = grafoBottleneck(X, Y)
    if distPinf == float("inf"):
        return float("inf")
    else:
        U, V = bipartite.sets(G)
        dU = {u: 0 for u in U}
        dV = {v: 0 for v in V}
        Gi = G.copy()
        Gi_0 = subgrafoG(Gi, 0)
        Mi = dict()

        plotMatching(Gi, Mi, U)

        if Gi_0.nodes():
            Mi = nx.bipartite.maximum_matching(Gi_0)

        plotMatching(Gi, Mi, U)

        while not nx.is_perfect_matching(Gi, Mi):
            Di = digrafoAsociado(Gi, Mi, U, V)
            length, path = nx.single_source_dijkstra(Di, "s")
            caminoAumento = path["t"][1:-1]

            for i in range(0, len(caminoAumento), 1):
                x = caminoAumento[i]
                # Mi.pop(x, None)
                if i%2==0:
                    Mi[x] = caminoAumento[i+1]
                else:
                    Mi[x] = caminoAumento[i-1]

            plotMatching(Gi, Mi, U)

            for u in U:
                dU[u] -= length[u]

            for v in V:
                dV[v] += length[v]

            cambiarPesos(G, Gi, dU, dV)


        return max([G[u][Mi[u]]["weight"] for u in U if u in Mi.keys()]+[distPinf])

if __name__ == "__main__":
    A = np.array([(2, 4), (3, 2), (0, 0), (0, 0.8), (4, float('inf'))])
    B = np.array([(2.8, 4), (3, 3), (4.2, float('inf'))])
    print("Hausdorff: ", hausdorffDir(A, B), "\n")

    C = np.array([(2, 4), (4, 2), (0, 0)])
    D = np.array([(2.8, 4), (4.8, 2.8), (0, 0.8)])

    diag1 = np.array([(2.7, 3.7),(9.6, 14.),(34.2, 34.974), (3., float('inf'))])
    diag2 = np.array([(2.8, 4.45),(9.5, 14.1),(3.2, float('inf'))])

    E = np.array([(0, 0)])
    F = np.array([(0, 13)])

    G, _, _, d = grafoBottleneck(diag1, diag2)

    print("DISTANCIA P INF", d)
    print(G.nodes(data=True), "\n")
    print(G.edges(data=True), "\n")

    print(bottleneck(diag1, diag2), "=", gudhi.bottleneck_distance(diag1, diag2))

