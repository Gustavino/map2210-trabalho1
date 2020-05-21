import numpy as np
from scipy import linalg

precision = 0.0000000000000001

def find_first_non_null_pivot(A, start):
    """  Iterate over the first column searching for a bigger than zero pivot  """
    pivot_row_index = start
    n = len(A)

    for i in range(start, n):
        if abs(A[i][start]) > precision:
            pivot_row_index = i
            break

    return pivot_row_index

def backward_substitution(A):
    n = len(A)
    x = np.ones(shape = n)                                  
                                             
    for i in range(n-1, -1, -1):                                # STEPS 8 and 9.
        x[i] = A[i][n] / A[i][i]                                     
        for j in range(i-1, -1, -1):
            A[j][n] -= A[j][i] * x[i]

    return x                                                    # STEP 10.

def gauss(A):                                                   # A é a matriz aumentada de Ax = B
    # n é a dimensão da matriz quadrada A, nxn, definida fora da função. Ou seja, n é o tamanho da matriz A não aumentada.
    n = len(A)

    for i in range(0, n):                                       # STEP 1. 
        pivot_row_index = find_first_non_null_pivot(A, i)

        # Checando se o pivô é igual a zero. 
        if abs(A[pivot_row_index][i]) < precision:              # STEP 2.
            print("No unique solution exists.")
            return []

        # Trocando as linhas se o pivô não estiver na diagonal principal.
        if (pivot_row_index != i):                              # STEP 3.
            A[[i, pivot_row_index]] = A[[pivot_row_index, i]] 

        for k in range(i + 1, n):                               # STEP 4.
            m = -A[k][i] / A[i][i]                              # STEP 5.
            for j in range(i, n + 1):                           # STEP 6.
                if i == j:
                    A[k][j] = 0
                else:
                    A[k][j] += m * A[i][j]

    # Verificando se, após o processo de escalonamento que transforma A em uma matriz U (Upper triangular), há algum zero na diagonal principal. Se houver, implicará que o sistema linear não possui solução definida.
    for k in range(0, n):                                       # STEP 7. 
        if abs(A[k][k]) < precision:
            print("No unique solution exists.")
            return []

    # Backward substitution. Contains steps 8-10. Resolvendo o sistema para uma matriz triangular superior.
    return backward_substitution(A)


if __name__ == "__main__":
    for n in range(2,16, 2):
        # Gerando uma matriz de Hilbert de dimensões nxn
        hilbert_matrix = linalg.hilbert(n)

        # Gerando o vetor b, que contém em sua i-ésima linha a soma da i-ésima linha da matriz de Hilbert.
        vector_b = np.array([hilbert_matrix[i].sum() for i in range (0, len(hilbert_matrix))])
        vector_b = np.vstack(vector_b)

        augmented_hilbert_matrix = np.append(hilbert_matrix, vector_b, axis=1)

        vector_x = gauss(augmented_hilbert_matrix)
        print("The output vector is {}.".format(vector_x))