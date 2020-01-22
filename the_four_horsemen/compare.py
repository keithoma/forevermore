import numpy as np
import scipy.linalg as lg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from pathlib import Path


def qr(matrix):
	"""Diese Funktion gibt die QR-Zerlegung einer Matrix zurück.

	Input
	-----
	matrix : numpy.ndarray
	Die Matrix, deren QR-Zerlegung berechnet werden soll.

	Return
	------
	rounded_Q, rounded_R : tuple
	QR-Zerlegung als Tupel (Q, R)
	"""
	non_rounded = lg.qr(matrix)
	rounded_Q = np.round(non_rounded[0], 6)
	rounded_R = np.round(non_rounded[1], 6)
	return rounded_Q, rounded_R

def get_rank(QR_matrix):
	"""Berechnet den Rang einer Matrix unter Ausnutzung ihrer QR-Zerlegung aus.

	Input
	-----
	QR_matrix : tuple
	Tupel, welches die QR-Zerlegung der Matrix, deren Rang wir berechnen möchten,
	darstellt (Q, R).

	Return
	------
	rank : int
	Rang der Matrix, deren QR-Zerlegung wir durch das Input-Tupel eingesetzt haben.
	"""
	rank = 0
	rounded_QR = np.round(QR_matrix[1], 6)
	for i in range(len(QR_matrix[1][0])):
		if rounded_QR[i][i] != 0:
			rank = rank + 1
	return rank

def full_rank(QR_matrix):
	"""Gibt an, ob eine mxn Matrix mit m>=n die Vollrangbedingung erfüllt.

	Input
	-----
	QR_matrix : tuple
	Tupel, welches die QR-Zerlegung der Matrix, die wir auf Vollrangbedingung prüfen möchten,
	darstellt (Q, R).

	Return
	------
	True/False : boolean
	True, falls Vollrangbedingung erfüllt ist,
	False, falls diese nicht erfüllt ist.
	"""
	rank = get_rank(QR_matrix)
	if rank == len(QR_matrix[1][0]):
		return True
	else:
		return False

def solve_QR(matrix, b):
	"""Löst überbestimmte lineare Gleichungssysteme mittels QR-Zerlegung.

	Input
	-----
	QR_matrix : numpy.ndarray
	Die QR-Zerlegung der Matrix, dessen Normalengleichung wir für den Vektor b lösen möchten.

	b : numpy.ndarray
	Der Vektor auf der rechten Seite der Normalengleichung.

	Return
	------
	x_min : numpy.ndarray
	Lösungsvektor der Normalengleichung bzgl. der Matrix und des Vektors b.
	"""

	QR_matrix = qr(matrix)
	Q_transpose = np.transpose(QR_matrix[0])
	z_value = np.matmul(Q_transpose, b)
	z_1 = z_value[0:matrix.shape[1]:1]
	z_2 = z_value[matrix.shape[1]:]
	R_first_nxn = QR_matrix[1][0:matrix.shape[1]:1]
	x_min = lg.solve_triangular(R_first_nxn, z_1)
	return x_min

def norm(matrix, x, b):
	"""Gibt die Norm des Residuums für einen beliebigen Vektor x bzgl. eines (überbestimmten)
	linearen Gleichungssystems Ax = b aus.

	Input
	-----
	matrix : numpy.ndarray
	Die Matrix in der Formel des Residuums
	x : numpy.ndarray
	Der gewählte Lösungsvektor x für das überbestimmte lineare Gleichungssystem
	b : numpy.ndarray
	Die rechte Seite des überbestimmten linearen Gleichungssystems

	Return
	------
	np.linalg.norm(inside_norm) : numpy.float64
	Die Norm des Residuums für das überbestimmte lineare Gleichungssystem und den Lösungs-
	vektor x.
	"""
	inside_norm = np.matmul(matrix, x) - b
	return np.linalg.norm(inside_norm)

def condition(matrix):
	"""Gibt die Kondition der Matrix A und der Matrix A^T * A aus.

	Input
	-----
	matrix : numpy.ndarray
	Die Matrix, für die wir obiges berechnen möchten

	Return
	------
	condition_matrix, condition_product : tuple
	Tupel mit der Kondition der Matrix A bzw. der Kondition von A^T * A,
	berechnet mit der 2-Norm.
	"""
	matrix_transposed = np.transpose(matrix)
	product_with_transposed = np.matmul(matrix_transposed, matrix)
	condition_matrix = np.linalg.cond(matrix)
	condition_product = np.linalg.cond(product_with_transposed)
	return condition_matrix, condition_product

def get_data(data_folder, file, tupel_length, selection_list = None):
	"""Diese Funktion liest die Daten einer Datei ein und gibt sie
	für unsere Zwecke geordnet als numpy.ndarray wieder aus.

	Input
	-----
	file_data : string
	der Datenpfad zur Textdatei mit den Daten
	tupel_length : int
	Die Daten unserer Datei sind als Datenpunkt (1 eingeben), -paar(2 eingeben),
	-tripel(3 eingeben) gespeichert, welche dieser benutzt wird,
	signalisiert der Nutzer hiermit
	selection_list : list (optional)
	Durch eine Liste an Integern können Zeilen aus der Datei ausgewählt werden,
	für die die Datenpaare betrachtet werden sollen.

	Return
	------
	input_values : numpy.ndarray
	Unsere Daten als Numpy-Array wie in der Textdatei angeordnet
	"""
	data_folder = Path(data_folder)
	file_to_open = data_folder / file
	input_values = open(file_to_open)
	input_values = input_values.read().replace(",", "").split()
	input_values = list(map(np.float64, input_values))
	it = iter(input_values)
	input_values = [i for i in zip(*[iter(input_values)]*tupel_length)]
	input_values = np.asarray(input_values)
	if selection_list != None:
		input_values = input_values[selection_list,:]
	return input_values

def main():
    def demo(A, b):
        # We start by printing A and b.
        print("Matrix A =\n{}\n\nVector b =\n{}".format(A, b))
        print("\n===== ===== =====\n")
        # Proceed by printing the QR-decomposition of A.
        Q, R = qr(A)
        print("For the QR-decomposition, we have\nQ =\n{}\n\nR =\n{}".format(Q, R))
        print("\n===== ===== =====\n")
        # Has the matrix A full column rank?
        print("Has the matrix A full column rank?\n=> {}".format(full_rank(qr(A))))
        print("\n===== ===== =====\n")
        # Solve Ax = b.
        print("We solve the equation Ax = b with the QR-decomposition.\nb =\n{}".format(solve_QR(A, b)))
        print("\n===== ===== =====\n")
        # Norm.
        print("The norm of (Ax - b) is\n{}".format(norm(A, solve_QR(A, b), b)))
        print("\n===== ===== =====\n")
        # Condition.
        cond_A, cond_sqr = condition(A)
        print("The condition of the matrix A is\n{}\n".format(cond_A))
        print("The condition of the matrix A^T * A is\n{}\n".format(cond_sqr))

    A = np.transpose([[2,   3,  5, 7,  11],
                      [13, 17, 19, 23, 29],
                      [31, 37, 41, 43, 47]])

    b = [1, 2, 3, 4, 5]
    demo(A, b)
if __name__ == '__main__':
    main()
