import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

def display_circles_min(pcs, n_comp, pca, axis_ranks, labels=None, label_rotation=0, lims=None, minimum=None, displayArrows=True, radius=1, zoom=6, target=None):
    for d1, d2 in axis_ranks:
        if d2 < n_comp:
            
            # initialisation de la figure
            fig, ax = plt.subplots(figsize=(zoom,zoom))

            xmin, xmax, ymin, ymax = -radius, radius, -radius, radius

            tab = pcs[[d1,d2]].T
            if minimum is not None:
                tabFiltered = np.empty([0, 2], dtype=float)
                for iTab in range(len(tab)):
                    if (abs(tab[iTab][0]) > minimum or abs(tab[iTab][1]) > minimum or labels[iTab]==target):
                        newrow = tab[iTab]
                        tabFiltered = np.vstack([tabFiltered, newrow])
                tab = tabFiltered

            # affichage des flèches
            # s'il y a plus de 30 flèches, on n'affiche pas le triangle à leur extrémité
            if displayArrows:
                plt.quiver(np.zeros(len(tab)), np.zeros(len(tab)),
                   tab[:,0], tab[:,1], 
                   angles='xy', scale_units='xy', scale=1, color="grey")
                # (voir la doc : https://matplotlib.org/api/_as_gen/matplotlib.pyplot.quiver.html)
            else:
                tab = pcs[[d1,d2]].T
                # Si une limite est définie on ne filtre que les valeurs supérieurs :
                if minimum is not None:
                    tabFiltered = []
                    for iTab in range(len(tab)):
                        if (abs(tab[iTab][0]) > minimum or abs(tab[iTab][1]) > minimum):
                            tabFiltered.append(tab[iTab])
                    tab = tabFiltered
                lines = [[[0,0],[x,y]] for x,y in tab]
                ax.add_collection(LineCollection(lines, axes=ax, alpha=.1, color='black'))
            
            # affichage des noms des variables  
            if labels is not None:  
                for i,(x, y) in enumerate(pcs[[d1,d2]].T):
                    if x >= xmin and x <= xmax and y >= ymin and y <= ymax :
                        if (minimum is not None) and ((abs(x) > minimum) or (abs(y) > minimum) or (labels[i]==target)):
                            plt.text(x, y, labels[i], fontsize='10', ha='center', va='center', rotation=label_rotation, color="blue", alpha=0.5)
            
            # affichage du cercle
            circle = plt.Circle((0,0), 1, facecolor='none', edgecolor='b')
            plt.gca().add_artist(circle)

            # définition des limites du graphique
            plt.xlim(xmin, xmax)
            plt.ylim(ymin, ymax)
        
            # affichage des lignes horizontales et verticales
            plt.plot([-1, 1], [0, 0], color='grey', ls='--')
            plt.plot([0, 0], [-1, 1], color='grey', ls='--')

            # nom des axes, avec le pourcentage d'inertie expliqué
            plt.xlabel('F{} ({}%)'.format(d1+1, round(100*pca.explained_variance_ratio_[d1],1)))
            plt.ylabel('F{} ({}%)'.format(d2+1, round(100*pca.explained_variance_ratio_[d2],1)))

            plt.title("Cercle des corrélations (F{} et F{})".format(d1+1, d2+1))
            plt.show(block=False)

def display_variables(pcs, axis_ranks, labels=None, minimum=None, target=None):
    for d1, d2 in axis_ranks: 
        f1 = {}
        f2 = {}
        for i,(x, y) in enumerate(pcs[[d1,d2]].T):
            if abs(x) > minimum or (labels[i]==target) :
                f1[labels[i]] = x
            if abs(y) > minimum or (labels[i]==target) :
                f2[labels[i]] = y
    print('Représentation des features sur F1 : ')
    for key, value in f1.items():
        print('    %s : value = %f' % (key, value))
    print('Représentation des features sur F2 : ')
    for key, value in f2.items():
        print('    %s : value = %f' % (key, value))

def display_variable(pcs, axis_rank, labels=None, minimum=None, target=None):
    f = {}
    for i, x in enumerate(pcs[axis_rank].T):
        if (abs(x) > minimum) or (labels[i]==target) :
            f[labels[i]] = x
    print('Représentation des features sur F : ')
    for key, value in f.items():
        print('    %s : value = %f' % (key, value))

