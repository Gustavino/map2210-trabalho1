import sys
import numpy as np
from scipy import linalg

# Configuração para gerar o log
np.set_printoptions(precision=6, linewidth=200, suppress=True)

precision = 0.0000000000000001

number_of_row_exchanges = 0

def max_element_in_row(A, i):
    n = len(A)
    max = i
    for k in range(i, n):
        if abs((A[max][i] - A[k][i])) < precision: # Prevendo situação onde há overflow do float e a operação de subtração retorna um número negativo *estranho*, há a compa.
            continue

        elif (A[max][i] - A[k][i]) < 0:
            max = k
    return max

def backward_substitution(A, determinant_signal):
    n = len(A)
    x = np.ones(shape = n)                                  
                                             
    for i in range(n-1, -1, -1):
        x[i] = A[i][n] / A[i][i] 
        for j in range(i-1, -1, -1):
            A[j][n] -= A[j][i] * x[i]

    A = np.delete(A, n, axis=1)
    print("O determinante vale {:.10f}".format(np.linalg.det(A) * determinant_signal))
    return x                                                      

def gauss(A):                                    
    global number_of_row_exchanges                   
    n = len(A)
    determinant_signal = 1

    for i in range(0, n):                                       
        max_pivot_row_index = max_element_in_row(A, i)

        # Checando se o pivô é igual a zero. 
        if abs(A[max_pivot_row_index][i]) < precision: 
            print("no unique solution exists")
            return np.array(object=[])

        if (max_pivot_row_index != i):             
            determinant_signal *= -1                 
            A[[i, max_pivot_row_index]] = A[[max_pivot_row_index, i]] 

        for k in range(i + 1, n):
            m = -A[k][i] / A[i][i]
            for j in range(i, n + 1):
                if i == j:
                    A[k][j] = 0
                else:
                    A[k][j] += m * A[i][j]

    # STARTING BACKWARD SUBSTITUTION
    return backward_substitution(A)

if __name__ == "__main__":

    old_stdout = sys.stdout
    log_file = open("gaussian with pivoting.log", "w")
    sys.stdout = log_file

    print ("A precisão utilizada é {}".format(precision)) 

    for n in range(2, 13, 2):
        hilbert_matrix = linalg.hilbert(n) # Gerando uma matriz de Hilbert nxn
        
        vector_b = np.array([hilbert_matrix[i].sum() for i in range (0, len(hilbert_matrix))]) # Gerando o vetor b, a soma das linhas da matriz de Hilbert
        vector_b = np.vstack(vector_b)
        
        augmented_hilbert_matrix = np.append(hilbert_matrix, vector_b, axis=1) 

        print("--------------------------------------------------------------------------")
        print("INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM {}".format(n))
        vector_x = gauss(augmented_hilbert_matrix)

        print("O vetor resultante é {}".format(vector_x.view()))
        
        # Se o vetor retornado não é vazio, isto é, se o procedimento teve sucesso. 
        if (vector_x.shape != (0,)):
            print("O primeiro \"1.\" mostrado pelo Python, na verdade, é {:.18f}".format(vector_x.view()[0]))
            ones_vector = np.ones(n)
            print("A norma ideal é {}".format(np.dot(ones_vector, ones_vector)))
            print("A norma do vetor resultante é {}".format(np.dot(vector_x, vector_x)))
            print("Diferença entre as normas: {}".format(np.dot(vector_x, vector_x) - np.dot(ones_vector, ones_vector)))
        
    
    sys.stdout = old_stdout
    log_file.close