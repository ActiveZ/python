import itertools
import tkinter as tk

# import cartesian as cartesian
import numpy as np
import scipy.special


def en_test():
    x1 = np.random.randint(10, size=(3, 4))  # Tableau de dimension 2
    x1 = np.random.randint(10, size=(6))  # Tableau de dimension 1
    print("nombre de dimensions de x1: ", x1.ndim)
    print("forme de x1: ", x1.shape)
    print("taille de x1: ", x1.size)
    print("type de x1: ", x1.dtype)
    print(x1)
    print(x1[::-1])

    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    z = np.concatenate([x, y])
    print(x)
    print(y)
    print(z)

    x = np.array([1, 2, 3])
    grid = np.array([[9, 8, 7],
                     [6, 5, 4]])

    print(np.vstack([x, grid]))

    x = np.arange(10)
    print(x)
    print("x + 5 =", x + 5)

    " --------------------------------------------------"
    M = np.random.randint(6, size=(3, 4))
    print(M)
    # Notez la syntax variable.fonction au lieu de
    # np.fonction(variable). Les deux sont possibles si
    # la variable est un tableau Numpy.
    print("La somme de tous les éléments de M: ", M.sum())
    print("Les sommes des colonnes de M: ", M.sum(axis=0))
    print("Les sommes des lignes de M: ", M.sum(axis=1))

    x = np.random.randint(2, size=(3, 10))
    print(x)
    print("Nombre de colonne de x: ", x.shape[0])

    # y=x.sum(axis=0)
    # print("Les sommes des colonnes de M: ", np.logical_or(y==0, y==3))

    print("Les sommes des colonnes de x: ", x.sum(axis=0))
    print("Les sommes des colonnes de M ok !: ", np.logical_or(x.sum(axis=0) == 0, x.sum(axis=0) == x.shape[0]))


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
    # tabloColonnes = csv[:4,5:]
    print("Colonnes: ", tabloColonnes)
    print("Colonne 0: ", tabloColonnes[0])
    tabloLigne = csv[csv.shape[0] - nbLigne:, :csv.shape[1] - nbColonne]
    print("Lignes: ", tabloLigne)


def cartesien2():
    # https://stackoverflow.com/questions/1208118/using-numpy-to-build-an-array-of-all-combinations-of-two-arrays
    # print("Tablo: ", tablo)
    # n = np.zeros(5, dtype=np.byte)
    # print("n: ",n)
    print("meshgrid")
    # x = (np.array(np.meshgrid([1,2,3], [4,5], [6,7])).T.reshape(-1, 3))
    # x = (np.array(np.meshgrid([1, 2, 3], [0, 0, 0, 0, 0, 0]))) # .T.reshape(-1, 6))
    # x = (np.array(np.meshgrid([1, 2, 3], [0, 0, 0, 0, 0, 0])).T.reshape(-1, 6))
    # print(x)
    # cartes=cartesian(([1, 2, 3], [0,0,0,0,0,0]))
    x = [1, 2, 3]
    y = [0, 0, 0, 0, 0, 0]
    cartes = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    print("cart: ", cartes)

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

print("Total ligne: ", csv.shape[0])
print("Total colonne: ", csv.shape[1])

tabloColonne = np.transpose(csv[:csv.shape[0] - nbLigne, csv.shape[1] - nbColonne:])
tabloLigne = csv[csv.shape[0] - nbLigne:, :csv.shape[1] - nbColonne]
# -------------------------------------------------------------

obj_ligne = Ligne(tabloLigne)
obj_ligne.list_ligne()
print("-------------------------------------------------------------")
obj_colonne = Colonne(tabloColonne)
obj_colonne.list_colonne()

# en_test()
# a1 + x1 + a2 + x2 + a3 + x3 + ... + an + xn + an+1 les extremes peuvent = 0, pas les autres


arra = np.ones((8, 8))
print("Original array:")
print(arra)
result = np.triu(arra, k=1) * 8
print("\nMemory size of each element of result")
print(result.itemsize)
print("\nResult1:")
print(result)
result = np.triu(arra, k=2) * 8
print("\nResult2:")
print(result)
print("\nAfter swapping column1 with column4:")
result[:, [0, 3]] = result[:, [3, 0]]
print(result)

# https://www.w3resource.com/python-exercises/numpy/python-numpy-exercise-42.php
x = np.eye(3, dtype=int) * 8
print("\nMemory size pour eye int")
print(x.itemsize)
print("eye:\n", x)

# x = np.diagflat([4, 5, 6, 8, 0, 0, 0, 0])

x = np.diagflat(np.ones(6,dtype=int)*3)
print("diagflat:\n", x)
y=np.ones(6, dtype=int)*8
print("y: ",y)
z=np.ones(6, dtype=int)*5
x=np.insert(x,0,y,axis=1)
x=np.insert(x,0,z,axis=1)
print("diagflat2:\n", x)

x=scipy.special.comb(6,3) # C(n,k)
print("comb: ", x)
