import sys
from math import sqrt
import numpy as np
from scipy import linalg
from sklearn import datasets
import time

np.set_printoptions(precision=6, linewidth=300, suppress=True)
precision = 0.0000000000000001

time_elapsed_cholesky = []
time_elapsed_naive = []

error_naive = []
error_cholesky = []
error_linalgsolve = []
 
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

    # STEP 9
    for i in range (1, n):
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

def find_first_non_null_pivot(A, start):
    """  Iterate over the first column searching for a bigger than zero pivot  """
    pivot_row_index = start
    n = len(A)

    for i in range(start, n):
        if abs(A[i][start]) > precision:
            pivot_row_index = i
            break

    return pivot_row_index

def backward_substitution_naive(A):
    n = len(A)
    x = np.ones(shape = n)                                  
                                             
    for i in range(n-1, -1, -1):                                # STEPS 8 and 9.
        x[i] = A[i][n] / A[i][i]                                     
        for j in range(i-1, -1, -1):
            A[j][n] -= A[j][i] * x[i]

    return x                                                    # STEP 10.

def gauss_naive(A):                                                   # A é a matriz aumentada de Ax = B
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
    return backward_substitution_naive(A)

if __name__ == "__main__":
    old_stdout = sys.stdout
    log_file = open("naive vs cholesky vs linalg.solve no hilbert.log", "w")
    sys.stdout = log_file

    print ("A precisão utilizada é {}".format(precision))

    for dimension in range(1, 202, 20):
    
        # ## Testing with Hilber t
        # reference_matrix = linalg.hilbert(dimension)
        # b = np.array([reference_matrix[i].sum() for i in range (0, len(reference_matrix))])
        # reference_vector_x = np.ones(dimension)

        # # Calculating Hilbert's error

        # print("--------------------------------------------------------------------------")
        # print("INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM {}".format(dimension))

        # augmented_matrix_to_naive = np.append(reference_matrix, np.vstack(b), axis=1)
        # naive_x = gauss_naive(augmented_matrix_to_naive)
        # error_naive = np.sqrt(np.dot(np.subtract(naive_x, reference_vector_x), np.subtract(naive_x, reference_vector_x)))
        # print("O erro da Eliminação Gaussiana sem pivotamento é {}".format(error_naive))

        # cholesky_x = cholesky(reference_matrix, b)
        # error_cholesky = np.sqrt(np.dot(np.subtract(cholesky_x, reference_vector_x), np.subtract(cholesky_x, reference_vector_x)))
        # print("O erro da resolução com matrizes fatoradas por Cholesky é {}".format(error_cholesky))

        # linalgsolve_x = np.linalg.solve(reference_matrix, b)
        # error_linalgsolve = np.sqrt(np.dot(np.subtract(linalgsolve_x, reference_vector_x), np.subtract(linalgsolve_x, reference_vector_x)))
        # print("O erro da biblioteca NumPy, através do np.linalg.solve, é {}".format(error_linalgsolve))



        # ### Testing with random positive definite matrices with random solutions
        reference_matrix = datasets.make_spd_matrix(dimension)
        b = np.random.rand(dimension)

        print("--------------------------------------------------------------------------")
        print("INICIANDO ITERAÇÃO PARA UMA MATRIZ DE ORDEM {}".format(dimension))
        

        # O vetor resultado retornado pelo linalg.solve será usado como solução referência 
        # à aplicação dos métodos de Cholesky e Naive Gauss.
        reference_vector_x = np.linalg.solve(reference_matrix, b)

        reference_determinant = np.linalg.det(reference_matrix)


        augmented_matrix_to_naive = np.append(reference_matrix, np.vstack(b), axis=1)
        start_naive = time.time()
        naive_x = gauss_naive(augmented_matrix_to_naive)
        end_naive = time.time()
        time_elapsed_naive.append(end_naive - start_naive)
        error_naive = np.sqrt(np.dot(np.subtract(naive_x, reference_vector_x), np.subtract(naive_x, reference_vector_x)))
        print("O erro da Eliminação Gaussiana sem pivotamento é {}".format(error_naive))
        
        start_cholesky = time.time()
        cholesky_x = cholesky(reference_matrix, b)
        end_cholesky = time.time()
        time_elapsed_cholesky.append(end_cholesky - start_cholesky)
        error_cholesky = np.sqrt(np.dot(np.subtract(cholesky_x, reference_vector_x), np.subtract(cholesky_x, reference_vector_x)))
        print("O erro da resolução com matrizes fatoradas por Cholesky é {}".format(error_cholesky))

        print("--- TEMPO GASTO EM CADA SOLUÇÃO ---")

        print("Tempo gasto pelo Naive Gauss nessa solução foi {:.3f}s".format(end_naive - start_naive))
        print("Tempo gasto pelo método de Cholesky nessa solução foi {:.3f}s".format(end_cholesky - start_cholesky))
        
        # # print("--- ERRO DOS DETERMINANTES EM MÓDULO---")
        # # print("Erro do de")



    sys.stdout = old_stdout
    log_file.close