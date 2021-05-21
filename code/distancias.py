# -*- coding: utf-8 -*-
"""
Created on Tue May  4 15:45:56 2021.

@author: Alejandro
"""

import numpy as np
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
import sympy as sy
import gudhi

# variable curvas
t = sy.symbols('t', real=True)


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
    elif x[1] == float('inf') and y[1] != float('inf'):
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
    """
    Obtener el grafo bipartido de para el emparejamiento de la distancia bottleneck.

    X: numpy.array.
    y: numpy.array.
    """
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
            d = 0.0
            if U[i] in X0 or V[j] in Y0:
                d = distanciaInf(U[i], V[j])

            edges.append((f"u{i}", f"v{j}", d))

    G.add_weighted_edges_from(edges)

    distPinf = 0

    if len(X0_inf) != len(Y0_inf):
        distPinf = float("inf")
    else:
        distPinf = max([abs(x-y) for x, y in zip(sorted(X0_inf), sorted(Y0_inf))]+[0])

    return G, U, V, distPinf


def subgrafoG(G, i):
    """
    Obtener el subgrafo de G formado por las aristas que tienen peso menor o igual a i.

    G: networkx.Graph.
    i: float.
    """
    G0 = nx.Graph([(u, v, d) for u, v, d in G.edges(data=True) if d['weight'] <= i])
    for n, d in G0.nodes(data=True):
        d["bipartite"] = G.nodes[n]["bipartite"]

    return G0


def subgrafoG_todosV(G, i):
    """
    Obtener el subgrafo generador de G formado por las aristas que tienen peso menor o igual a i.

    G: networkx.Graph.
    i: float.
    """
    G0 = nx.Graph()
    G0.add_nodes_from(G.nodes(data=True))
    G0.add_edges_from([(u, v, d) for u, v, d in G.edges(data=True) if d['weight'] <= i])

    return G0


def cambiarPesos(Gi, length):
    """
    Obtener los pesos de la nueva iteración a partir del diccionario de distancia obtenido de Dijkstra.

    Gi: networkx.Graph.
    length: dict.
    """
    for u, v, d in Gi.edges(data=True):
        d['weight'] += length[u] - length[v]
        if d['weight'] < 0:
            d['weight'] = 0

    return 0


def digrafoAsociado(G, M, U=None, V=None):
    """
    Obtener el digrafo asociado al emparejamiento M y el grafo bipartido M con vertices U+V.

    Gi: networkx.Graph.
    M: dict.
    U: list.
    V: list.
    """
    if U is None or V is None:
        U, V = bipartite.sets(G)

    D = nx.DiGraph()
    keysM = M.keys()

    aristas = list()
    for u, v, d in G.edges(data='weight'):
        if u in U:
            if u not in keysM or M[u] != v:
                aristas.append((u, v, d))
            else:
                aristas.append((v, u, d))
        else:
            if u not in keysM or M[u] != v:
                aristas.append((v, u, d))
            else:
                aristas.append((u, v, d))

    D.add_weighted_edges_from(aristas)
    D.add_weighted_edges_from([("s", u, 0) for u in U if u not in keysM])
    D.add_weighted_edges_from([(v, "t", 0) for v in V if v not in keysM])

    return D


def plotMatching(G, M, U):
    """
    Representar el emparejamiento M en el grafo bipartido G con vertices superiores en U.

    G: networkx.Graph.
    M: dict.
    U: list.
    """
    plt.figure(figsize=(8, 6))
    pos = nx.drawing.layout.bipartite_layout(G, U)
    nx.draw_networkx(G, pos=pos)
    edgelist = [(u, M[u]) for u in U if u in M.keys()]
    nx.draw_networkx_edges(G, pos, edgelist=edgelist, width=2.5, edge_color='blue')


def bottleneck(X, Y, plot=False, debug=False):
    """
    Calcular la distancia bottleneck entre los conjuntos de puntos X y Y.

    Si se quiere que se muestren los emparejamientos poner plot=True.
    Si se quiere información de la traza debug=True.

    A: numpy.array.
    B: numpy.array.
    plot: boolean.
    debug: boolean.
    """
    G, _, _, distPinf = grafoBottleneck(X, Y)
    if distPinf == float("inf"):
        return float("inf")
    else:
        U, V = bipartite.sets(G)
        Gi = G.copy()
        Mi = dict()

        if plot:
            plotMatching(Gi, Mi, U)

        while not nx.is_perfect_matching(G, Mi):
            Di = digrafoAsociado(Gi, Mi, U, V)
            length, path = nx.single_source_dijkstra(Di, "s")
            caminoAumento = path["t"][1:-1]

            for i in range(0, len(caminoAumento)):
                x = caminoAumento[i]

                if i % 2 == 0:
                    Mi[x] = caminoAumento[i+1]
                else:
                    Mi[x] = caminoAumento[i-1]

            if plot:
                plotMatching(Gi, Mi, U)

            if debug:
                print("\n", {u: Mi[u] for u in U if u in Mi}, "\n")
                print(nx.is_perfect_matching(subgrafoG_todosV(Gi, 0), Mi), "\n")
                print("MAX", max([G[u][Mi[u]]["weight"] for u in U if u in Mi.keys()]+[distPinf]), "\n")
                print(caminoAumento, "\n")
                print([(u, v, d) for u, v, d in Gi.edges(data=True) if d["weight"] < 0], "\n")

            cambiarPesos(Gi, length)

        return max([G[u][Mi[u]]["weight"] for u in U if u in Mi.keys()]+[distPinf])


if __name__ == "__main__":

    A = np.array([(2, 4), (3, 2), (0, 0), (0, 0.8), (4, 5.2)])
    B = np.array([(2.8, 4), (3, 3), (4.2, 5.8)])
    C = np.array([(2, 4), (4, 2), (0, 0)])
    D = np.array([(2.8, 4), (4.8, 2.8), (0, 0.8)])

    diag1 = np.array([(2.7, 3.7), (9.6, 14.), (34.2, 34.974), (3., float('inf'))])
    diag2 = np.array([(2.8, 4.45), (9.5, 14.1), (3.2, float('inf'))])

    G, _, _, d = grafoBottleneck(diag1, diag2)

    print("Hausdorff: ", hausdorffDir(diag1, diag2), "\n")
    # print(G.nodes(data=True), "\n")
    # print(G.edges(data=True), "\n")
    print("Bottleneck: ", bottleneck(diag1, diag2), "=", gudhi.bottleneck_distance(diag1, diag2))
