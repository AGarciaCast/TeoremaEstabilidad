# -*- coding: utf-8 -*-
"""
Created on Fri May 14 23:34:04 2021.

@author: Alejandro
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import gudhi
import folium
import sympy as sy
import matplotlib.colors
from folium.map import Layer, FeatureGroup,LayerControl,Marker
import distancias
from scipy.spatial.distance import directed_hausdorff

# variable curvas
t = sy.symbols('t', real=True)

coloresPartidos = {
                      "PSOE": "#DD5C47",
                      "PP": "#5C95DA",
                      "ERC": "#E6C410",
                      "VOX": "#5CB451",
                      "JxC": "#B6194D",
                      "Bildu": "#6BE955",
                      "PNV": "#4F954A",
                      "NA+": "#A83B2A",
                      "UP": "#8961AA",
                      "PRC": "#269A1B",
                      "CC": "#FFEF4B",
                      "Cs": "#DF9B07",
                      "TEX": "#037252"
                      }


def plotalpha(ac, st, k, ax):
    """
    Representar el alpha complejo del complejo K.

    puntos: np.array.
    K: Complejo.
    """
    puntos = np.array([ac.get_point(i) for i in range(st.num_vertices())])

    triangulos = [s[0] for s in st.get_skeleton(2) if len(s[0]) == 3 and s[1] <= k]
    if triangulos:
        c = np.ones(len(puntos))
        cmap = matplotlib.colors.ListedColormap("limegreen")
        ax.tripcolor(puntos[:, 0], puntos[:, 1], [s[0] for s in st.get_skeleton(2) if len(s[0]) == 3 and s[1] <= k],
                     c, edgecolor="k", lw=2, cmap=cmap)

    ax.plot(puntos[:, 0], puntos[:, 1], 'ko')

    aristas = [s[0] for s in st.get_skeleton(2) if len(s[0]) == 2 and s[1] <= k]
    if aristas:
        for arista in aristas:
            p1 = puntos[arista[0]]
            p2 = puntos[arista[1]]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'k')


def puntosCurvaRuido(curva, t, t0, t1, numPuntos=10, mu=0, sigma=0.1):
    """
    Obtener conjunto discretos de puntos de una curva con ruido.

    curva: list.
    t: t sympy symbol
    t0: float
        Inicio intervalo.
    t1: float
        Final intervalo.
    numPuntos: int. Por defecto 10.
    mu: float. Por defecto 0
        Media para la distribución normal.
    sigma: float. Por defecto 0.1
        Desviación típica para la distribución normal.
    """
    valores = np.linspace(t0, t1, num=numPuntos)
    puntosCurva = np.array([[x.subs(t, v) for x in curva] for v in valores], dtype=np.float64)
    ruido = np.random.normal(mu, sigma, [numPuntos, len(curva)])

    return puntosCurva + ruido


def ejemploPlano(points1, points2, indice, mapa=None):
    if mapa is None:
        fig, axs = plt.subplots(2, 2, dpi=100)
        fig.set_size_inches(15, 10)
    else:
        fig, axs = plt.subplots(1, 2, dpi=100)
        fig.set_size_inches(15, 5)
        sub_group1 = folium.FeatureGroup(name="Alpha Complejo Wikipedia", control=True, show=False)
        sub_group2 = folium.FeatureGroup(name="Alpha Complejo Random", control=True, show=True)

    alpha_complex1 = gudhi.AlphaComplex(points=points1)
    simplex_tree1 = alpha_complex1.create_simplex_tree()

    result_str = 'Alpha complex is of dimension ' + repr(simplex_tree1.dimension()) + ' - ' + \
                 repr(simplex_tree1.num_simplices()) + ' simplices - ' + \
                 repr(simplex_tree1.num_vertices()) + ' vertices.'
    print(result_str)

    value = max(set([filtered_value[1] for filtered_value in simplex_tree1.get_filtration()]))

    if mapa is None:
        plotalpha(alpha_complex1, simplex_tree1, value, axs[0, 0])
        axs[0, 0].title.set_text(r"$r={}$".format(str(value)))
    else:
        plotAlphaMapa(alpha_complex1, simplex_tree1, value, mapa, sub_group1)

    diag1 = simplex_tree1.persistence()

    if mapa is None:
        gudhi.plot_persistence_diagram(diag1, axes=axs[0, 1])
    else:
        gudhi.plot_persistence_diagram(diag1, axes=axs[0])

    alpha_complex2 = gudhi.AlphaComplex(points=points2)
    simplex_tree2 = alpha_complex2.create_simplex_tree()

    result_str = 'Alpha complex is of dimension ' + repr(simplex_tree2.dimension()) + ' - ' + \
                 repr(simplex_tree2.num_simplices()) + ' simplices - ' + \
                 repr(simplex_tree2.num_vertices()) + ' vertices.'
    print(result_str)

    value = max(set([filtered_value[1] for filtered_value in simplex_tree1.get_filtration()]))

    if mapa is None:
        plotalpha(alpha_complex2, simplex_tree2, value, axs[1, 0])
        axs[1, 0].title.set_text(r"$r={}$".format(str(value)))
    else:
        plotAlphaMapa(alpha_complex2, simplex_tree2, value, mapa, sub_group2)

    diag2 = simplex_tree2.persistence()
    if mapa is None:
        gudhi.plot_persistence_diagram(diag2, axes=axs[1, 1])
    else:
        gudhi.plot_persistence_diagram(diag2, axes=axs[1])

    fig.tight_layout()
    fig.savefig(f'ejemplo{indice}.png', dpi=100)
    plt.show()

    diag1_0 = np.array([s[1] for s in diag1 if s[0] == 0])
    diag2_0 = np.array([s[1] for s in diag2 if s[0] == 0])

    print("\nDistancia Hausdorff: ", round(distancias.hausdorffDir(points1, points2), 5))

    print("Distancia bottleneck dimensión 0:", round(distancias.bottleneck(diag1_0, diag2_0), 5),
          "(Implementación GUDHI:", str(round(gudhi.bottleneck_distance(diag1_0, diag2_0), 5)) + ")")

    diag1_1 = np.array([s[1] for s in diag1 if s[0] == 1])
    diag2_1 = np.array([s[1] for s in diag2 if s[0] == 1])
    print("Distancia bottleneck dimensión 1:", round(distancias.bottleneck(diag1_1, diag2_1), 5),
          "(Implementación GUDHI:", str(round(gudhi.bottleneck_distance(diag1_1, diag2_1), 5)) + ")", "\n")


def ejemplo1(backup=True):
    if backup:
        with open('puntos1_1.npy', 'rb') as f:
            points1 = np.load(f)

        with open('puntos1_2.npy', 'rb') as f:
            points2 = np.load(f)
    else:
        curva = [4 * sy.sin(t), 9 * sy.cos(t)]
        points1 = puntosCurvaRuido(curva, t, 0, 2*np.pi, numPuntos=30)
        points2 = puntosCurvaRuido(curva, t, 0, 2*np.pi, numPuntos=30, mu=2, sigma=0.3)

    print("-------------Ejemplo 1-------------\n")
    ejemploPlano(points1, points2, 1)


def ejemplo2(backup=True):
    if backup:
        with open('puntos2_1.npy', 'rb') as f:
            points1 = np.load(f)

        with open('puntos2_2.npy', 'rb') as f:
            points2 = np.load(f)
    else:
        curva = [10 * sy.cos(2 * t) * sy.cos(t),
                 10 * sy.cos(2 * t) * sy.sin(t)]
        points1 = puntosCurvaRuido(curva, t, 0, 2*np.pi, numPuntos=50, sigma=0.00)
        points2 = puntosCurvaRuido(curva, t, 0, 2*np.pi, numPuntos=50, mu=2, sigma=0.3)

    print("-------------Ejemplo 2-------------\n")
    ejemploPlano(points1, points2, 2)


def estiloMapa(feature, elecciones):
    provincia = feature['properties']['texto']
    partido = elecciones.loc[elecciones["Provincia"] == provincia, "Partido"].values[0]
    provincia_style = {
                        'weight': 1,
                        'fillOpacity': 0.8,
                        'fillColor': coloresPartidos[partido],
                        'color': '#000000'
                      }

    return provincia_style


def plotAlphaMapa(ac, st, k, m, sub_group):
    """
    Representar el alpha complejo del complejo K.

    puntos: np.array.
    K: Complejo.
    """
    puntos = np.array([ac.get_point(i) for i in range(st.num_vertices())])
    aristas = [s[0] for s in st.get_skeleton(2) if len(s[0]) == 2 and s[1] <= k]

    if aristas:
        for arista in aristas:
            p1 = puntos[arista[0]].tolist()
            p2 = puntos[arista[1]].tolist()
            folium.PolyLine([p1, p2], weight=3, color='black').add_to(sub_group)

    for p in puntos:
        folium.CircleMarker(p.tolist(), radius=4, fill=True, fill_opacity=1,
                            fill_color='white', color='black').add_to(sub_group)
    sub_group.add_to(m)


def ejemploMapa():
    m = folium.Map(
        location=[36.2, -4],
        tiles="CartoDB Positron",
        zoom_start=6,
    )

    elecciones = pd.read_csv('elecciones2019Nov.csv')
    eleccionesPSOE = elecciones[elecciones["Partido"] == "PSOE"]
    coordenadasWiki = np.array([tuple(elem) for elem in eleccionesPSOE[["Latitud", "Longitud"]].values.tolist()])
    coordenadasRandom = np.array([tuple(elem) for elem in eleccionesPSOE[["LatitudR", "LongitudR"]].values.tolist()])

    folium.GeoJson(r"spain_provincias.geojson", name="Partidos Ganadores", control=False,
                   style_function=(lambda feature: estiloMapa(feature, elecciones))).add_to(m)

    print("-------------Ejemplo 3-------------\n")
    ejemploPlano(coordenadasWiki, coordenadasRandom, "Mapa", mapa=m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save("mapaEspaña.html")
    return elecciones, coordenadasRandom


if __name__ == "__main__":
    ejemplo1()
    ejemplo2()
    elecciones, coordenadasRandom = ejemploMapa()
