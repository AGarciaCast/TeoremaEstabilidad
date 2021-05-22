# -*- coding: utf-8 -*-
"""
Created on Sat May 22 16:54:05 2021.

@author: Alejandro
"""

from numpy import arange
from pylab import meshgrid, cm, imshow, contour, clabel, colorbar, title
import math
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt

# Directorios datos
pathOut = r"output/"


def z_func(x, y, A):
    """
    Calcular la distancia entre el punto (x, y) y el conjunto A.

    x: float.
    y: float.
    A: list.
    """
    p1 = (x, y)
    return min([math.sqrt(sum([(a - b) ** 2 for a, b in zip(p1, p2)])) for p2 in A])


mpl.rcParams['figure.dpi'] = 100

A = [(-1.0, 0.0), (1.0, 0.0), (1.0, 1.0), (-1.0, -1.0)]
x = arange(-3.0, 3.0, 0.1)
y = arange(-3.0, 3.0, 0.1)
# grid of point
X, Y = meshgrid(x, y)
# evaluation of the function on the grid
Z = np.array([[z_func(x1, y1, A) for x1 in x] for y1 in y])
# drawing the function
im = imshow(Z, cmap=cm.RdBu)
# adding the Contour lines with labels
cset = contour(Z, arange(-1, 1.5, 0.2), linewidths=2, cmap=cm.Set2)
clabel(cset, inline=True, fmt='%1.1f', fontsize=10)
# adding the colobar on the right
colorbar(im)
# latex fashion title
title('$z=d((x,y), A)$')
plt.savefig(pathOut + 'subnivelDistancias.png', dpi=200, bbox_inches='tight')
plt.show()

fig = plt.figure(dpi=200)
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                       cmap=cm.RdBu, linewidth=0, antialiased=True)

ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
fig.colorbar(surf, shrink=0.8, aspect=10, pad=0.1)

fig.savefig(pathOut + 'grafoDistancias.png', dpi=200, bbox_inches='tight')
plt.show()
