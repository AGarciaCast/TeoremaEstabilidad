# -*- coding: utf-8 -*-
"""
Created on Fri May 14 23:34:04 2021.

@author: Alejandro
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gudhi
import folium
import sympy as sy
import matplotlib.colors
import distancias

# variable curvas
t = sy.symbols('t', real=True)

# Colores partidos España
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
# Directorios datos
pathIn = r"input/"
pathOut = r"output/"


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
    """
    Calcular la distancia Hausdorff entre los conjuntos de puntos y la distancia bottleneck entre sus diagramas de persistencia.

    Si se va a usar el mapa se utilizarán complejos de Vietoris-Rips y en otro caso alfa complejos

    points1: numpy.array.
    points2: numpy.array.
    incice: String.
        String indicativo del ejemplo a ejecutar.
    mapa = folium.map.
    """
    # ------------Primer conjunto de puntos------------

    if mapa is None:
        # Inicializamos la figura
        fig, axs = plt.subplots(2, 2, dpi=100)
        fig.set_size_inches(15, 10)

        # Calculamos el alfa complejo
        complex1 = gudhi.AlphaComplex(points=points1)
        simplex_tree1 = complex1.create_simplex_tree()
        nameComp = "Alpha complex"
    else:
        # Inicializamos la figura
        fig, axs = plt.subplots(1, 2, dpi=100)
        fig.set_size_inches(15, 5)
        fig2, axs2 = plt.subplots(1, 2, dpi=100)
        fig2.set_size_inches(15, 5)

        # Calculamos el complejo de Vietoris-Rips
        complex1 = gudhi.RipsComplex(points=points1)
        simplex_tree1 = complex1.create_simplex_tree(max_dimension=2)
        nameComp = "Vietoris-Rips complex"

    print(nameComp + 'is of dimension ' + repr(simplex_tree1.dimension()) + ' - ' +
          repr(simplex_tree1.num_simplices()) + ' simplices - ' +
          repr(simplex_tree1.num_vertices()) + ' vertices.'
          )

    # Obtenemos el diagrama de persistencia
    diag1 = simplex_tree1.persistence()
    if mapa is None:
        # Obtener valor filtración
        value = max(set([filtered_value[1] for filtered_value in simplex_tree1.get_filtration()]))

        # Representar alfa complejo de dicha filtración
        plotalpha(complex1, simplex_tree1, value, axs[0, 0])
        axs[0, 0].title.set_text(r"Alfa complejo $(r = {})$".format(str(value)))

        # Representar el diagrama de persistencia
        gudhi.plot_persistence_diagram(diag1, axes=axs[0, 1])
    else:
        # Obtener valor filtración
        valueM = max([s[1][1] for s in diag1 if s[0] == 1])
        value = max([s[1][0] for s in diag1 if s[1][1] == valueM])

        # Representar complejo de Vietoris-Rips de dicha filtración sobre el mapa
        sub_group1 = folium.FeatureGroup(name=f"Complejo VR Wikipedia (r = {value})", control=True, show=True)
        plotAlphaMapa(points1, simplex_tree1, value, mapa, sub_group1)

        # Representar el diagrama de persistencia y codigo de barras
        fig.suptitle(r"Complejo VR Wikipedia $(r = {})$".format(str(value)), fontsize=16)
        gudhi.plot_persistence_diagram(diag1, axes=axs[0])
        gudhi.plot_persistence_barcode(diag1, axes=axs[1])

    # ------------Segundo conjunto de puntos------------

    if mapa is None:
        # Calculamos el alfa complejo
        complex2 = gudhi.AlphaComplex(points=points2)
        simplex_tree2 = complex2.create_simplex_tree()
    else:
        # Calculamos el complejo de Vietoris-Rips
        complex2 = gudhi.RipsComplex(points=points2)
        simplex_tree2 = complex2.create_simplex_tree(max_dimension=2)

    print(nameComp + 'is of dimension ' + repr(simplex_tree2.dimension()) + ' - ' +
          repr(simplex_tree2.num_simplices()) + ' simplices - ' +
          repr(simplex_tree2.num_vertices()) + ' vertices.'
          )

    # Obtenemos el diagrama de persistencia
    diag2 = simplex_tree2.persistence()
    if mapa is None:
        # Obtener valor filtración
        value = max(set([filtered_value[1] for filtered_value in simplex_tree1.get_filtration()]))

        # Representar alfa complejo de dicha filtración
        plotalpha(complex2, simplex_tree2, value, axs[1, 0])
        axs[1, 0].title.set_text(r"Alfa complejo $+$ Ruido $(r = {})$".format(str(value)))

        # Representar el diaggrama de persistencia
        gudhi.plot_persistence_diagram(diag2, axes=axs[1, 1])
    else:
        # Obtener valor filtración
        valueM = max([s[1][1] for s in diag2 if s[0] == 1])
        value = max([s[1][0] for s in diag2 if s[1][1] == valueM])

        # Representar complejo de Vietoris-Rips de dicha filtración sobre el mapa
        sub_group2 = folium.FeatureGroup(name=f"Complejo VR Random (r = {value})", control=True, show=False)
        plotAlphaMapa(points2, simplex_tree2, value, mapa, sub_group2)

        # Representar el diagrama de persistencia y codigo de barras
        fig2.suptitle(r"Complejo VR Random $(r = {})$".format(str(value)), fontsize=16)
        gudhi.plot_persistence_diagram(diag2, axes=axs2[0])
        gudhi.plot_persistence_barcode(diag2, axes=axs2[1])

    # Ajustar figura y guardarla en directorio de trabajo
    fig.tight_layout()
    if mapa is not None:
        fig.subplots_adjust(top=0.88)
        fig2.tight_layout()
        fig2.subplots_adjust(top=0.88)
        fig2.savefig(pathOut + f'ejemplo{indice}2.png', dpi=100)

    fig.savefig(pathOut + f'ejemplo{indice}.png', dpi=100)
    plt.show()

    # ------------Calculo de las distancias------------

    # Obtener los puntos de dimensión 0 de los diagramas
    diag1_0 = np.array([s[1] for s in diag1 if s[0] == 0])
    diag2_0 = np.array([s[1] for s in diag2 if s[0] == 0])

    print("\nDistancia Hausdorff: ", round(distancias.hausdorffDir(points1, points2), 5))

    print("Distancia bottleneck dimensión 0:", round(distancias.bottleneck(diag1_0, diag2_0), 5),
          "(Implementación GUDHI:", str(round(gudhi.bottleneck_distance(diag1_0, diag2_0), 5)) + ")")

    # Obtener los puntos de dimensión 1 de los diagramas
    diag1_1 = np.array([s[1] for s in diag1 if s[0] == 1])
    diag2_1 = np.array([s[1] for s in diag2 if s[0] == 1])
    print("Distancia bottleneck dimensión 1:", round(distancias.bottleneck(diag1_1, diag2_1), 5),
          "(Implementación GUDHI:", str(round(gudhi.bottleneck_distance(diag1_1, diag2_1), 5)) + ")", "\n")


def ejemplo1(backup=True):
    """
    Ejecutar el ejemplo 1: Estudio de la estabilidad en una elipse.

    backup: boolean.
        True si quieres usar los puntos guardados en los ficheros puntos1_1.npy y puntos1_2.npy.
    """
    if backup:
        with open(pathIn + 'puntos1_1.npy', 'rb') as f:
            points1 = np.load(f)

        with open(pathIn + 'puntos1_2.npy', 'rb') as f:
            points2 = np.load(f)
    else:
        curva = [4 * sy.sin(t), 9 * sy.cos(t)]
        points1 = puntosCurvaRuido(curva, t, 0, 2*np.pi, numPuntos=30)
        points2 = puntosCurvaRuido(curva, t, 0, 2*np.pi, numPuntos=30, mu=2, sigma=0.3)

    print("-------------Ejemplo 1-------------\n")
    ejemploPlano(points1, points2, 1)


def ejemplo2(backup=True):
    """
    Ejecutar el ejemplo 2: Estudio de la estabilidad en una rosa polar.

    backup: boolean.
        True si quieres usar los puntos guardados en los ficheros puntos2_1.npy y puntos2_2.npy.
    """
    if backup:
        with open(pathIn + 'puntos2_1.npy', 'rb') as f:
            points1 = np.load(f)

        with open(pathIn + 'puntos2_2.npy', 'rb') as f:
            points2 = np.load(f)
    else:
        curva = [10 * sy.cos(2 * t) * sy.cos(t),
                 10 * sy.cos(2 * t) * sy.sin(t)]
        points1 = puntosCurvaRuido(curva, t, 0, 2*np.pi, numPuntos=50, sigma=0.00)
        points2 = puntosCurvaRuido(curva, t, 0, 2*np.pi, numPuntos=50, mu=2, sigma=0.3)

    print("-------------Ejemplo 2-------------\n")
    ejemploPlano(points1, points2, 2)


def estiloMapa(feature, elecciones):
    """
    Representación de las provincias segun el color del partido politico ganador.

    feature: GeoJson feature.
    elecciones: pandas.DataFrame.
    """
    provincia = feature['properties']['texto']
    partido = elecciones.loc[elecciones["Provincia"] == provincia, "Partido"].values[0]
    provincia_style = {
                        'weight': 1,
                        'fillOpacity': 0.8,
                        'fillColor': coloresPartidos[partido],
                        'color': '#000000'
                      }

    return provincia_style


def plotAlphaMapa(puntos, st, k, m, sub_group):
    """
    Representar el alpha complejo del complejo K.

    puntos: np.array.
    K: Complejo.
    """
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
    """Ejecutar el ejemplo 3: Estudio de la estabilidad en el mapa de España de las elecciones nacionales de Nov 2019."""
    m = folium.Map(
        location=[36.2, -4],
        tiles="CartoDB Positron",
        zoom_start=6,
    )

    elecciones = pd.read_csv(pathIn + "elecciones2019Nov.csv", index_col="id_provincia")
    eleccionesPSOE = elecciones[elecciones["Partido"] == "PSOE"]
    coordenadasWiki = np.array([tuple(elem) for elem in eleccionesPSOE[["Latitud", "Longitud"]].values.tolist()])
    coordenadasRandom = np.array([tuple(elem) for elem in eleccionesPSOE[["LatitudR", "LongitudR"]].values.tolist()])

    folium.GeoJson(pathIn + r"spain_provincias.geojson", name="Partidos Ganadores", control=False,
                   style_function=(lambda feature: estiloMapa(feature, elecciones))).add_to(m)

    print("-------------Ejemplo 3-------------\n")
    ejemploPlano(coordenadasWiki, coordenadasRandom, "Mapa", mapa=m)

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(pathOut + "mapSpain.html")
    return elecciones, coordenadasRandom


if __name__ == "__main__":
    ejemplo1()
    ejemplo2()
    elecciones, coordenadasRandom = ejemploMapa()
