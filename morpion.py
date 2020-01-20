# https://deptinfo-ensip.univ-poitiers.fr/ENS/doku/doku.php/stu:python_gui:pyqt
# Dessiner en utilisant paintEvent
from PySide2 import QtGui, QtCore, QtWidgets
import sys

# Numéro du joueur en cours (1 ou -1)
numero_joueur = 1
# Plateau de jeu
jeu = [[0] * 3 for i in range(3)]


def analyse_partie(gr_jeu):
    # Renvoie 1 ou -1 si un joueur a gagné
    # Renvoie 0 si la partie est nulle
    # Renvoie None dans les autres cas
    for j in range(3):
        s = sum(gr_jeu[j][i] for i in range(3))
        if s == 3 or s == -3: return s // 3
        s = sum(gr_jeu[i][j] for i in range(3))
        if s == 3 or s == -3: return s // 3
    s = sum(gr_jeu[i][i] for i in range(3))
    if s == 3 or s == -3: return s // 3
    s = sum(gr_jeu[i][2 - i] for i in range(3))
    if s == 3 or s == -3: return s // 3
    for i in range(3):
        for j in range(3):
            if gr_jeu[i][j] == 0: return None
    return 0


class ZoneDessin(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def fin_partie(self):
        # S'il y a un gagnant, on affiche un message et on réinitialise le
        # jeu
        g = analyse_partie(jeu)
        if g != None:
            msg = QtWidgets()
            if g == 1 or g == -1:
                msg.setText("Le joueur " + str(g) + " est vainqueur")
            else:
                msg.setText("Partie nulle")
            # La main passe au dialogue
            msg.exec_()
            # Remise de la partie à 0
            for i in range(3):
                for j in range(3): jeu[i][j] = 0
            self.repaint()

    def mousePressEvent(self, e):
        global numero_joueur
        largeur_case = self.width() // 3
        hauteur_case = self.height() // 3
        # Les coordonnées du point cliqué sont e.x() et e.y()

        # Transformation des coordonnées écran en coordonnées dans
        # le plateau de jeu
        j = e.x() // largeur_case
        i = e.y() // hauteur_case
        # Vérification
        print('Vous avez cliqué sur la case : ', (i, j))
        # La case est elle vide ?
        if jeu[i][j] == 0:
            # Si oui, on joue le coup
            jeu[i][j] = numero_joueur
            # Et c'est au tour de l'autre joueur
            numero_joueur = -numero_joueur
        # Si non, rien de particulier à faire. C'est toujours au même
        # joueur

        # On réaffiche
        self.repaint()
        # On analyse le jeu pour savoir s'il y a une fin de partie
        self.fin_partie()

    def paintEvent(self, e):
        p = QtGui.QPainter(self)
        p.setBrush(QtGui.QBrush(QtCore.Qt.SolidPattern))
        # Dessin de la grille
        largeur_case = self.width() // 3
        hauteur_case = self.height() // 3
        for i in range(4):
            p.drawLine(0, i * hauteur_case, self.width(), i * hauteur_case)
            p.drawLine(i * largeur_case, 0, i * largeur_case, self.height())
        # Dessin des pions
        # On parcourt la représentation du jeu et on affiche
        for i in range(3):
            for j in range(3):
                if jeu[i][j] != 0:
                    if jeu[i][j] == 1:
                        p.setBrush(QtGui.QColor(255, 0, 0))
                    else:
                        p.setBrush(QtGui.QColor(255, 255, 0))
                    p.drawEllipse(j * largeur_case, i * hauteur_case,
                                  largeur_case, hauteur_case)


class Fenetre(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.resize(420, 420)
        self.setWindowTitle("Tic Tac Toe")
        dessin = ZoneDessin(self)
        dessin.setGeometry(10, 10, 400, 400)


app = QtWidgets.QApplication(sys.argv)
frame = Fenetre()
frame.show()
sys.exit(app.exec_())

