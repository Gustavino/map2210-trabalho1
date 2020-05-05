from math import sqrt
import numpy as np
from scipy import linalg
from sklearn import datasets
 
def cholesky(A, b):
    n = len(A)
    L = np.zeros((n,n))

    # Decomposição de Cholesky 
    for i in range(n):
        for k in range(i+1):
            left_elements_sum = sum(L[i][j] * L[k][j] for j in range(k)) # Nem roda para a primeira iteracao
            
            if (i == k): # Elementos da diagonal
                if A[i][i] - left_elements_sum < 0:
                    print("A[i][i] eh {:.19f}".format(A[i][i]))
                    print("left_elements_sum eh {:.19f}".format(left_elements_sum))
                L[i][k] = sqrt(A[i][i] - left_elements_sum)
            else:
                L[i][k] = (A[i][k] - left_elements_sum) / L[k][k]

    y_vector = np.array(np.zeros(n))

    # STEP 8
    y_vector[0] = b[0] / L[0][0]

    # STEP 9
    for i in range (1, n): # CHECK
        sum_y = sum(L[i][j] * y_vector[j] for j in range(i))
        y_vector[i] = (b[i] - sum_y) / L[i][i]

    # STEP 10
    x_vector = np.array(np.zeros(n))

    x_vector[-1] = y_vector[-1] / L[-1][-1]

    # STEP 11
    for i in range(n-2, -1, -1):
        sum_x = sum(L[j][i] * x_vector[j] for j in range(i+1, n))
        x_vector[i] = (y_vector[i] - sum_x) / L[i][i]



    # STEP 12
    return x_vector
 
# A = np.array([[6, 3, 4, 8], [3, 6, 5, 1], [4, 5, 10, 7], [8, 1, 7, 25]])
A = np.array([[4, 2, 2], [2, 6, 2], [2, 2, 5]])
# B = np.array([[25, 15, -5], 
# [15, 18, 0], 
# [-5, 0, 11]])


dimension = 18
# reference_matrix = A
reference_matrix = linalg.hilbert(dimension)
# reference_matrix = datasets.make_spd_matrix(dimension)

vector_b = np.array([reference_matrix[i].sum() for i in range (0, len(reference_matrix))]) # Generating vector b, the sum of Hilbert matrices rows
# vector_b = [0, 1, 0]
vector_x = cholesky(reference_matrix, vector_b)

# print("Matriz referência:")
# print(augmented_hilbert_matrix)

# print("L:")
# print(L)

# print(linalg.cholesky(reference_matrix))

# print("L transposta:")
# print(np.transpose(L))

print("vector_x:")
print(vector_x)








