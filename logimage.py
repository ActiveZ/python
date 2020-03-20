import itertools
import tkinter as tk
import timeit
# import cartesian as cartesian
import numpy as np
import scipy.special
from sympy.utilities.iterables import multiset_permutations


# https://openclassrooms.com/fr/courses/4452741-decouvrez-les-librairies-python-pour-la-data-science/4740941-plongez-en-detail-dans-la-librairie-numpy


def affiche_damier():
    cell_size = 20
    board_largeur = 15  # ou 8 pour un échiquier
    board_hauteur = 15

    colors = ["white", "black"]

    root = tk.Tk()
    root.title("Logimage")

    canvas = tk.Canvas(root, width=board_largeur * cell_size, height=board_hauteur * cell_size)
    canvas.pack()

    for x in range(board_hauteur):
        for y in range(board_largeur):
            color = colors[(x + y) % 2]
            canvas.create_rectangle(
                y * cell_size,
                x * cell_size,
                y * cell_size + cell_size,
                x * cell_size + cell_size,
                fill=color, outline=color
            )

    root.mainloop()


def saisie():
    nbLigne = 15
    nbColonne = 15
    # nbLigne = int(input("Nombre de ligne: "))
    # nbColonne = int(input("Nombre de colonne: "))
    csv = np.genfromtxt('gourmande.csv', delimiter=",")
    np.nan_to_num(csv, copy=False)  # remplace NaN par 0
    # print(csv)
    print("Total ligne: ", csv.shape[0])
    print("Total colonne: ", csv.shape[1])

    tabloColonnes = csv[:csv.shape[0] - nbLigne, csv.shape[1] - nbColonne:]
    print("Colonnes: ", tabloColonnes)
    print("Colonne 0: ", tabloColonnes[0])
    tabloLigne = csv[csv.shape[0] - nbLigne:, :csv.shape[1] - nbColonne]
    print("Lignes: ", tabloLigne)


class Ligne:
    def __init__(self, ligne):
        self.ligne = ligne

    def list_ligne(self):
        # [f(x) for x in another_list]  qui crée une liste en appliquant la fonction  f  à chaque membre de  another_list
        [print("ligne", i, obj_ligne.ligne[i]) for i in range(nbLigne)]


class Colonne:
    def __init__(self, colonne):
        self.colonne = colonne

    def list_colonne(self):
        [print("colonne", i, obj_colonne.colonne[i]) for i in range(nbColonne)]


def replace(tablo):  # remplace [2,3] par [1,1,0,1,1,1,] ok
    print("Replace Tablo: ", tablo)
    x = np.array([], dtype=np.byte)
    for i in tablo:
        x = np.concatenate((x, np.ones(i, dtype=np.byte), [0]))
    x = np.delete(x, -1)  # efface la derniere colonne
    print("x: ", x)


def test_combi(tablo):
    print("Tablo: ", tablo)
    n = np.zeros(5, dtype=np.byte)
    print("n: ", n)
    # print("x: ", x)


def factorielleAddition(n):
    total = 0
    for x in range(n + 1):
        total += x
    # print("n:", n, "somme:", total)
    return total


def decoupeTablo():
    test0 = np.eye(6, dtype=np.byte)
    print("test0: \n", test0, "\nnb elts:", test0.size)
    print(test0[3, :])


def permutationItertool(tabloK, longueurLigne):
    print("------------- permutations ----------------")
    result = np.array([], dtype=int)
    longueurLigne = longueurLigne - tabloK.sum() + len(tabloK)
    a = np.array(np.append(tabloK, np.zeros(longueurLigne - len(tabloK))),
                 dtype=int)  # exemple [6, 7, 8, 0, 0, 0] si tabloK=[6,7,8] et longueurLigne=6
    a=np.copy(tabloK) # en chantier !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print("aaaa:",a)
    a = np.array(list(set(itertools.permutations(a))))
    for x in a:
        if np.array_equal(x[x != 0], tabloK):  # si la permutation sans les zeros est égale au tabloK
            result = np.append(result, x)  # alors je la garde dans result
    result = np.reshape(result, (-1, longueurLigne))
    print("result: permut\n", result, "\nshape:", result.shape)
    return result


def transformationBinaire(tablo):  # cf def replace
    # print("tablo TB:", tablo)
    result = np.array([], dtype=bool)
    for x in tablo:
        result = np.append(result, np.ones(x, dtype=bool))
        result = np.append(result, [False])
    result = result[:-1]  # suppression du dernier element 0
    # print("result:", result)
    return result


def sommeColonnes(tabloData, nbCol):
    print("------------------- somme colonnes ---------------------------------")
    z = np.array([[]], dtype=bool).reshape(-1, nbCol)
    for x in permutationItertool(tabloData, nbCol):
        z = np.vstack([z, transformationBinaire(x)])
    # print("result z:\n", z)
    print("taille z:", z.shape)
    print("Somme des colonnes de z: ", np.logical_or(z.sum(axis=0) == 0, z.sum(axis=0) == z.shape[0]))
    # true signifie 0 ou 1 !


# ----------------maim-----------------------
# affiche_damier()
np.random.seed()
# ----------------------saisie--------------------------------
nbLigne = 15
nbColonne = 15
# nbLigne = int(input("Nombre de ligne: "))
# nbColonne = int(input("Nombre de colonne: "))
csv = np.genfromtxt('gourmande.csv', delimiter=",", dtype=int, filling_values=0)
# print(csv)

# print("Total ligne: ", csv.shape[0])
# print("Total colonne: ", csv.shape[1])

tabloColonne = np.transpose(csv[:csv.shape[0] - nbLigne, csv.shape[1] - nbColonne:])
tabloLigne = csv[csv.shape[0] - nbLigne:, :csv.shape[1] - nbColonne]
# -------------------------------------------------------------

obj_ligne = Ligne(tabloLigne)
# obj_ligne.list_ligne()

obj_colonne = Colonne(tabloColonne)
# obj_colonne.list_colonne()
# print("-------------------------------------------------------------")

k = np.array([1,2,3])
nbColonne=10


print("tablo K:", k, "  Longueur ligne:", nbColonne)

debut = timeit.default_timer()
sommeColonnes(k, nbColonne)
print("duree: ", timeit.default_timer() - debut)
