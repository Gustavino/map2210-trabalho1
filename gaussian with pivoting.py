
import sys
import numpy as np
from scipy import linalg

np.set_printoptions(precision=8, linewidth=150, suppress=True)

precision = 0.0000000000000001

def max_element_in_row(A, i):
    n = len(A)
    max = i
    for k in range(i, n):
        if A[max][i] < A[k][i]:
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
    n = len(A)
    determinant_signal = 1

    # print("A matriz aumentada inicial A é: ")
    # print(A.view())

    for i in range(0, n):                                       
        max_pivot_row_index = max_element_in_row(A, i)

        if abs(A[max_pivot_row_index][i]) < precision:          
            print("no unique solution exists")
            return np.array(object=[])

        if (max_pivot_row_index != i):
            # print("")
            # print("*** Matriz no momento de troca da linha {} ***".format(i+1))
            # print(A.view())
            determinant_signal *= -1                              
            A[[i, max_pivot_row_index]] = A[[max_pivot_row_index, i]]
            # print("")
            # print("Após a {}a verificação de troca, houve troca entre as linhas {} e {}: ".format(i+1, i+1, max_pivot_row_index+1))
            # print(A.view())
        # else:
        #     print("")
        #     print("Após a {}a verificação de troca, não houve troca de linhas: ".format(i+1))
        #     print(A.view())

        for k in range(i + 1, n):
            m = -A[k][i] / A[i][i]
            for j in range(i, n + 1):
                if i == j:
                    A[k][j] = 0
                else:
                    A[k][j] += m * A[i][j]

    # STARTING BACKWARD SUBSTITUTION
    return backward_substitution(A, determinant_signal)


if __name__ == "__main__":

    old_stdout = sys.stdout
    log_file = open("teste pivoting.log", "w")
    sys.stdout = log_file

    print ("A precisão utilizada é {}".format(precision))

    # for n in range(2, 13, 2):
    for n in range(1):
        n = 13
        hilbert_matrix = linalg.hilbert(n) # Generating a nxn Hilbert matrix
        #hilbert_matrix = np.random.rand(n, n)

        vector_b = np.array([hilbert_matrix[i].sum() for i in range (0, len(hilbert_matrix))]) # Generating vector b, the sum of Hilbert matrices rows
        vector_b = np.vstack(vector_b)
        
        augmented_hilbert_matrix = np.append(hilbert_matrix, vector_b, axis=1) 

        print("--------------------------------------------------------------------------")
        print("INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM {}".format(n))
        vector_x = gauss(augmented_hilbert_matrix)

        print("O vetor resultante é {}".format(vector_x.view()))
        

        if (vector_x.shape != (0,)):
            print("O primeiro \"1.\" mostrado pelo Python, na verdade, é {:.18f}".format(vector_x.view()[0]))

            
            ones_vector = np.ones(n)
            print("A norma ideal é {}".format(np.dot(ones_vector, ones_vector)))
            print("A norma do vetor resultante é {}".format(np.dot(vector_x, vector_x)))
            print("Diferença entre as normas: {}".format(np.dot(vector_x, vector_x) - np.dot(ones_vector, ones_vector)))
        
    
    sys.stdout = old_stdout
    log_file.close
