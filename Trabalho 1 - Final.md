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
  * A precisão definida no início do código é fundamental ao seu correto funcionamento: na demonstração abaixo, **a precisão limita a verificação do algoritmo até n=16**.
```python  
import numpy as np
from scipy import linalg

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
```  

Nessa demonstração de código são utilizadas **10** casas decimais de precisão, através de uma variável global, na definição de um possível zero na diagonal principal. Essa particularidade é decisiva, pois, devido aos elementos da matriz de Hilbert, o processo de escalonamento gera números muito pequenos (próximos de zero, com arredondamento de +5 casas decimais), inclusive na diagonal principal. Esse último detalhe é o principal motivo por trás da Eliminação de Gauss com pivotamento: pivôs muitos pequenos geram grandes erros no cálculo (será detalhado no item b). Assim, dada a forma como matrizes de Hilbert produzem elementos pequenos, espera-se que o algoritmo acumule mais erros ao aumentar a ordem da matriz de Hilbert utilizada. 

### Logs

Nos logs abaixo é o utilizado um vetor diferença definido pela diferença entre a solução obtida e a solução exata (todos as componentes unitárias) na seguinte ordem: v - x, onde v é o vetor de componentes unitárias e x é o vetor solução (também chamado de "vetor resultante" nos logs).
#### Log do processo com 10 casas decimais de precisão, para  2 <= n <= 7.


```python  
A precisão utilizada é 1e-10
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 2
O determinante vale 0.0833333333
O vetor resultante é [1. 1.]
O primeiro "1." mostrado pelo Python, na verdade, é 1.000000000000000444
O vetor diferença é [0. 0.]
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 3
O determinante vale 0.0004629630
O vetor resultante é [1. 1. 1.]
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999999999997891
O vetor diferença é [0. 0. 0.]
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 4
O determinante vale 0.0000001653
O vetor resultante é [1. 1. 1. 1.]
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999999999977351
O vetor diferença é [0. 0. 0. 0.]
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 5
O determinante vale 0.0000000000
O vetor resultante é [1. 1. 1. 1. 1.]
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999999999965472
O vetor diferença é [0. 0. 0. 0. 0.]
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 6
O determinante vale 0.0000000000
O vetor resultante é [1. 1. 1. 1. 1. 1.]
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999999999227840
O vetor diferença é [ 0.00000000e+00  0.00000000e+00  1.48208001e-10 -3.85369514e-10
  4.25416036e-10 -1.67680092e-10]
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 7
O determinante vale 0.0000000000
O vetor resultante é [1.         1.         1.         1.00000001 0.99999998 1.00000001
 1.        ]
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999999994452993
O vetor diferença é [ 0.00000000e+00 -2.21598517e-10  2.13563101e-09 -8.30333358e-09
  1.52219312e-08 -1.31521882e-08  4.31797642e-09]
```

  * Determinante: analisando-se brevemente a variação do determinante da matriz de Hilbert para entradas pequenas (n < 6), fica explícito que a matriz de Hilbert é má condicionada, o que implica numa "backward stability" ruim. Isto é, o processo de _backward substitution_ pode gerar muitas imprecisões no vetor x solução. 

#### Log do processo com 16 casas decimais de precisão, para  n = 2, 4, ... , 16.

```python
A precisão utilizada é 1e-16
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 2
O determinante vale 0.0833333333
O vetor resultante é [1. 1.]
O primeiro "1." mostrado pelo Python, na verdade, é 1.000000000000000444
A norma ideal é 2.0
A norma do vetor resultante é 1.9999999999999996
Diferença entre as normas: -4.440892098500626e-16
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 4
O determinante vale 0.0000001653
O vetor resultante é [1. 1. 1. 1.]
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999999999977351
A norma ideal é 4.0
A norma do vetor resultante é 4.000000000000034
Diferença entre as normas: 3.375077994860476e-14
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 6
O determinante vale 0.0000000000
O vetor resultante é [1. 1. 1. 1. 1. 1.]
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999999999227840
A norma ideal é 6.0
A norma do vetor resultante é 6.00000000000118
Diferença entre as normas: 1.1803891197814664e-12
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 8
O determinante vale 0.0000000000
O vetor resultante é [1.         1.         0.99999998 1.00000008 0.99999978 1.00000031
 0.99999978 1.00000006]
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999999977596477
A norma ideal é 8.0
A norma do vetor resultante é 8.000000000030838
Diferença entre as normas: 3.083755473198835e-11
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 10
O determinante vale 0.0000000000
O vetor resultante é [1.         1.00000013 0.99999725 1.00002475 0.99988305 1.00031898
 0.99948016 1.00049946 0.99973912 1.00005711]
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999998455165184
A norma ideal é 10.0
A norma do vetor resultante é 10.000000708903745
Diferença entre as normas: 7.089037445950908e-07
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 12
O determinante vale 0.0000000000
O vetor resultante é [0.99999995 1.00000656 0.99979705 1.00272894 0.98021524 1.08610951
 0.7620375  1.42766115 0.50179028 1.36282531 0.84990897 1.02691957]
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999947232208086
A norma ideal é 12.0
A norma do vetor resultante é 12.650441152699086
Diferença entre as normas: 0.6504411526990861
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 14
O determinante vale -0.0000000000
O vetor resultante é [ 0.99999995  1.00000676  0.99976399  1.00364936  0.96887594  1.16381344
  0.43302049  2.33729775 -1.18460166  3.4700271  -0.89519084  1.94052343
  0.72812698  1.0346873 ]
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999950892680345
A norma ideal é 14.0
A norma do vetor resultante é 31.562617034448234
Diferença entre as normas: 17.562617034448234
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 16
O determinante vale 0.0000000000
O vetor resultante é [ 1.00000003  0.99999517  1.00020507  0.99614982  1.03979013  0.7476902
  2.0373379  -1.81433493  5.89519165 -3.66145928  1.00545112  7.49509397
 -8.17118204  7.47272293 -1.42947208  1.38682036]
O primeiro "1." mostrado pelo Python, na verdade, é 1.000000028769932214
A norma ideal é 16.0
A norma do vetor resultante é 244.9988753089912
Diferença entre as normas: 228.9988753089912
```
  

**Vetor x solução**:  
  * Sobre suas componentes finais: observando-se os vetores resultantes até a terceira iteração, sobre a matriz de ordem 6, fica claro que os problemas decorrentes do má condicionamento da matriz de Hilbert ainda não alteraram drasticamente os resultados. Entretanto, já nota-se uma diferença: a presença do "." após o número um, como em "1.", no vetor resultante indica a presença de arrendondamento em casas decimais muito distantes da vírgula (certamente após a décima casa decimal, que é o padrão de exibição dos valores de np.darray ou listas do Python).  
  Dessa forma, da matriz de ordem 8 em diante, ficam claras as consequências dos arrendondamentos anteriores sobre o vetor solução.

  * Sobre a norma: como consequência das alterações sobre as componentes, a norma do vetor x nunca é exatamenge igual a ideal. De acordo com os logs, a diferença entre as normas produzida pelo algoritmo e ideal varia desde um número extremamente pequeno (a ponto de ocorrer um _floating point overflow_) até 15.25 vezes a norma ideal para n=16. Ou seja, quanto mais aumenta-se a ordem da matriz, mais aumenta-se a presença do erro no algoritmo.  
  
### b. Código estendendo a solução do item _a_ com o processo de **pivotamento**.  

* O código a seguir contém as mudanças necessárias ao pivotamento

* 

```python

import sys
import numpy as np
from scipy import linalg

np.set_printoptions(precision=6, linewidth=200, suppress=True)

precision = 0.0000000000000001

def find_first_non_null_pivot(A, start):
    """  Iterate over the first column searching for a bigger than zero pivot  """
    pivot_row_index = start

    for i in range(start, n):
        if abs(A[i][start]) > precision:
            pivot_row_index = i
            break

    return pivot_row_index

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

    print("A matriz aumentada inicial A é: ")
    print(A.view())

    for i in range(0, n):                                       
        max_pivot_row_index = max_element_in_row(A, i)

        # Checando se o pivô é igual a zero. 
        if abs(A[max_pivot_row_index][i]) < precision: 
            print("no unique solution exists")
            return np.array(object=[])

        if (max_pivot_row_index != i):             
            determinant_signal *= -1                 
            A[[i, max_pivot_row_index]] = A[[max_pivot_row_index, i]] 

        print("")
        print("Após a {}a verificação de possível troca: ".format(i+1))
        print(A.view())

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
```