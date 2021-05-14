# -*- coding: utf-8 -*-
"""
Created on Fri May 14 23:34:04 2021

@author: Alejandro
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import datetime

import folium
from folium.plugins import TimeSliderChoropleth



m = folium.Map(
    location=[40.416775, -3.703790],
    tiles="cartodbpositron",
    zoom_start=6,
)



provincias_geo = r"spain_provincias.geojson"

with open(provincias_geo) as provincias_file:
    provincias_json = json.load(provincias_file)

provincias = [provincias_json['features'][i]['properties']['texto']
              for i in range(len(provincias_json['features']))]

print(sorted(provincias))


coloresPartidos={
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

elecciones = pd.read_csv('elecciones2019Nov.csv')



def estilo(feature):

    provincia=feature['properties']['texto']
    print(provincia)
    partido = elecciones.loc[elecciones["Provincia"]==provincia, "Partido"].values[0]
    provincia_style = {
             'fillOpacity': 1,
             'weight': 1,
             'fillOpacity': 0.8,
             'fillColor': coloresPartidos[partido],
             'color': '#000000'}

    return provincia_style

folium.GeoJson(provincias_geo, name="geojson", style_function=estilo).add_to(m)

folium.CircleMarker([41.583333, -4.666667], radius=4, fill=True, fill_opacity=1, fill_color='white',
    color = 'black').add_to(m)

m.save("mapaEspaña.html")


