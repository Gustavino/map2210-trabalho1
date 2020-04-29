# Trabalho 1 - MAP2210 - Aplicações de Álgebra Linear

## Parte 1

### a. Código, comentado, em Python para resolução de sistemas lineares pelo método da Eliminação de Gauss sem pivotamento.  
  
  * Os comentários iniciados com "STEP" se referem ao algoritmo descrito no livro Numerical Analysis. Da mesma maneira, para manter conformidade com o idioma do livro-texto, diversos nomes de variáveis e funções estão em inglês.  
  * Em alguns pontos do código, é utilizado, para verificar se um número é zero, a função abs que entrega o módulo de um número (int, float).  
    * Exemplo: 
    ```python  
    precision = 0.00001
    abs(number) < precision # Nesse caso, são utilizadas cinco casas decimais para verificar se um número está próximo de zero.
    ```  



```python  
import numpy as np
from scipy import linalg
import math

precision = 0.0000000000000001

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
                                             
    for i in range(n-1, -1, -1):                                # STEPS 8 and 9.
        x[i] = A[i][n] / A[i][i]                                     
        for j in range(i-1, -1, -1):
            A[j][n] -= A[j][i] * x[i]


    #A = np.delete(A, n, axis=1)
    #print(A)
    #print(np.linalg.det(A))
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

    # Backward substitution. Contains steps 8-10. 
    return backward_substitution(A)


if __name__ == "__main__":
    for n in range(2,16, 2):
        # Gerando uma matriz de Hilbert de dimensões nxn
        hilbert_matrix = linalg.hilbert(n)

        # Gerando o vetor b, que contém em sua i-ésima linha a soma da i-ésima linha da matriz de Hilbert.
        vector_b = np.array([hilbert_matrix[i].sum() for i in range (0, len(hilbert_matrix))]) # Quantas casas decimais são utilizadas?
        vector_b = np.vstack(vector_b)

        augmented_hilbert_matrix = np.append(hilbert_matrix, vector_b, axis=1)

        vector_x = gauss(augmented_hilbert_matrix)
        print("O vetor resultante é {}, de ordem {}.".format(vector_x, n))
```  

Nessa demonstração de código são utilizadas **10** casas decimais de precisão, através de uma variável global, na definição de um possível zero na diagonal principal. Essa particularidade é decisiva, pois, devido aos elementos da matriz de Hilbert, o processo de escalonamento gera números muito pequenos (próximos de zero, com arredondamento de +5 casas decimais), inclusive na diagonal principal. Esse último detalhe é o principal motivo por trás da Eliminação de Gauss com pivotamento: pivôs muitos pequenos geram grandes erros no cálculo (será detalhado no item b).

#### Logs do processo para 10 casas decimais de precisão


```python

A precisão utilizada é 1e-10
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 2
O determinante vale 0.0833333333
O vetor resultante é [1. 1.].
O primeiro "1." mostrado pelo Python, na verdade, é 1.000000000000000444
O vetor diferença é [0. 0.]
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 4
O determinante vale 0.0000001653
O vetor resultante é [1. 1. 1. 1.].
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999999999977351
O vetor diferença é [0. 0. 0. 0.]
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 6
O determinante vale 0.0000000000
O vetor resultante é [1. 1. 1. 1. 1. 1.].
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999999999227840
O vetor diferença é [ 0.00000000e+00  0.00000000e+00  1.48208001e-10 -3.85369514e-10
  4.25416036e-10 -1.67680092e-10]
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 8
O determinante vale 0.0000000000
O vetor resultante é [1.         1.         0.99999998 1.00000008 0.99999978 1.00000031
 0.99999978 1.00000006].
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999999977596477
O vetor diferença é [ 0.00000000e+00 -1.17921117e-09  1.52438943e-08 -8.19868293e-08
  2.19829454e-07 -3.10170087e-07  2.20281768e-07 -6.20567091e-08]
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 10
no unique solution exists
O vetor resultante é [].
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 12
no unique solution exists
O vetor resultante é [].
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 14
no unique solution exists
O vetor resultante é [].


```