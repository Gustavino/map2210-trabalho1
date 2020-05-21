from math import sqrt
import numpy as np
from scipy import linalg
 
def cholesky(A, b):
    n = len(A)
    L = np.zeros((n,n))

    # Decomposição de Cholesky 
    for i in range(n):
        for k in range(i+1):
            left_elements_sum = sum(L[i][j] * L[k][j] for j in range(k)) # Nem roda para a primeira iteracao
            
            if (i == k): # Elementos da diagonal
                L[i][k] = sqrt(A[i][i] - left_elements_sum)
            else:
                L[i][k] = (A[i][k] - left_elements_sum) / L[k][k]

    y_vector = np.array(np.zeros(n))

    # STEP 8
    y_vector[0] = b[0] / L[0][0]

    # STEP 9 - Ly = b
    for i in range (1, n): 
        sum_y = sum(L[i][j] * y_vector[j] for j in range(i))
        y_vector[i] = (b[i] - sum_y) / L[i][i]

    # STEP 10
    x_vector = np.array(np.zeros(n))

    x_vector[-1] = y_vector[-1] / L[-1][-1]

    # STEP 11 - Ltx = y
    for i in range(n-2, -1, -1):
        sum_x = sum(L[j][i] * x_vector[j] for j in range(i+1, n))
        x_vector[i] = (y_vector[i] - sum_x) / L[i][i]

    # STEP 12
    return x_vector