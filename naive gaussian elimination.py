
import sys
import numpy as np
from scipy import linalg
import math

precision = 0.0000000001

def find_first_non_null_pivot(A, start):
    """  Iterate over the first column searching for a bigger than zero pivot  """
    pivot_row_index = start

    for i in range(start, n):
        if abs(A[i][start]) > precision:
            pivot_row_index = i
            break

    return pivot_row_index


def backward_substitution(A):
    n = len(A)
    x = np.ones(shape = n)                                  
                                             
    for i in range(n-1, -1, -1):                                     # STEP 9. Backward substitution.
        x[i] = A[i][n] / A[i][i]                                     # STEP 8
        for j in range(i-1, -1, -1):
            A[j][n] -= A[j][i] * x[i]


    A = np.delete(A, n, axis=1)
    #print(A)
    print("O determinante vale {:.10f}".format(np.linalg.det(A)))
    return x


def gauss(A):                                                   # A is the augmented matrix
    n = len(A)

    for i in range(0, n):                                       # STEP 1. i = i-th row.
        pivot_row_index = find_first_non_null_pivot(A, i)

        if abs(A[pivot_row_index][i]) < precision:          # STEP 2. Checking if the pivot element is equals to zero.
            print("no unique solution exists")
            return np.array(object=[])

        if (pivot_row_index != i):                              # STEP 3. Swaping rows if the pivot isn't in the main diagonal.
            A[[i, pivot_row_index]] = A[[pivot_row_index, i]] #------------------------STEP 4 5 6

        for k in range(i + 1, n):
            m = -A[k][i] / A[i][i]
            for j in range(i, n + 1):
                if i == j:
                    A[k][j] = 0
                else:
                    A[k][j] += m * A[i][j]

    for k in range(0, n):                                       # STEP 7. Verifying if there is any zero in the main diagonal.
        if abs(A[k][k]) < precision:                              # Checking if A[k][k] is zero.
            print("no unique solution exists")
            return np.array(object=[])

    # STARTING BACKWARD SUBSTITUTION
    return backward_substitution(A)


if __name__ == "__main__":

    old_stdout = sys.stdout
    log_file = open("message.log", "w")
    sys.stdout = log_file

    print ("A precisão utilizada é {}".format(precision))

    for n in range(2,16, 2):
        hilbert_matrix = linalg.hilbert(int(n)) # Generating a nxn Hilbert matrix
        
        vector_b = np.array([hilbert_matrix[i].sum() for i in range (0, len(hilbert_matrix))]) # Generating vector b, the sum of Hilbert matrices rows
        vector_b = np.vstack(vector_b)
        
        augmented_hilbert_matrix = np.append(hilbert_matrix, vector_b, axis=1) 

        print("--------------------------------------------------------------------------")
        print("INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM {}".format(n))
        vector_x = gauss(augmented_hilbert_matrix)

        print("O vetor resultante é {}.".format(vector_x, n))
        

        if (vector_x.shape != (0,)):
        
            if (abs(vector_x[0]) > 0.001):
                print("O primeiro \"1.\" mostrado pelo Python, na verdade, é {:.18f}".format(vector_x[0]))

            ones_vector = np.ones(n)
            subtraction = np.subtract(ones_vector, vector_x)
            for i in range(n):
                if abs(subtraction[i]) < precision:
                    subtraction[i] = 0
            print("O vetor diferença é {}".format(subtraction))
    
    sys.stdout = old_stdout
    log_file.close