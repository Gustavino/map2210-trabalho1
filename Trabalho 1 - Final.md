# Trabalho 1 - MAP2210 - Aplicações de Álgebra Linear

## Parte 1

> a. Código, comentado, em Python para resolução de sistemas lineares pelo método da Eliminação de Gauss sem pivotamento.  
  
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

Nessa demonstração de código são utilizadas **10** casas decimais de precisão, através de uma variável global, na definição de um possível zero na diagonal principal. Essa particularidade é decisiva, pois, devido aos elementos da matriz de Hilbert, o processo de escalonamento gera números muito pequenos (próximos de zero, com arredondamento de +5 casas decimais), inclusive na diagonal principal. Esse último detalhe é o principal motivo por trás da Eliminação de Gauss com pivotamento: pivôs muitos pequenos geram grandes erros no cálculo. Assim, dada a forma como matrizes de Hilbert produzem elementos pequenos, espera-se que o algoritmo acumule mais erros ao aumentar a ordem da matriz de Hilbert utilizada. 

### Logs

#### Log do processo com 10 casas decimais de precisão, para  n = 2, ... , 7.  

No log abaixo é o utilizado um vetor diferença definido pela diferença entre a solução obtida e a solução exata (todos as componentes unitárias) na seguinte ordem: v - x, onde v é o vetor de componentes unitárias e x é o vetor solução (também chamado de "vetor resultante" nos logs). 

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
O determinante vale 0.0000000000
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
  

**Sobre o vetor x solução**:  
* Acerca de suas componentes finais: observando-se os vetores resultantes até a terceira iteração, sobre a matriz de ordem 6, fica claro que os problemas decorrentes do má condicionamento da matriz de Hilbert ainda não alteraram drasticamente os resultados. Entretanto, já nota-se uma diferença: a presença do "." após o número um, como em "1.", no vetor resultante indica a presença de arrendondamento em casas decimais muito distantes da vírgula (certamente após a oitava casa decimal, que é o padrão de exibição dos valores de np.darray ou listas do Python).  
Dessa forma, da matriz de ordem 8 em diante, ficam claras as consequências incovenientes dos arrendondamentos anteriores sobre o vetor solução.

* Acerca de sua norma: como consequência das alterações sobre as componentes, a norma do vetor x nunca é exatamenge igual a ideal. De acordo com os logs, a diferença entre as normas produzida pelo algoritmo e ideal varia desde um número extremamente pequeno (a ponto de ocorrer um _floating point overflow_) até 15.25 vezes a norma ideal para n=16. Ou seja, quanto mais aumenta-se a ordem da matriz, mais aumenta-se a presença do erro no algoritmo.  
  
>b. Código estendendo a solução do item _a_ com o processo de **pivotamento**.  

* O código a seguir contém as mudanças necessárias ao pivotamento: na busca do pivô da i-ésima linha, é feita uma busca linear pelo maior elemento entre os elementos de A[i][j], com **0 <= i <= n-1** e **i <= j <= n-1**. Se o pivô encontrado não estiver na linha i, há uma troca das linhas i e j e o determinante terá o seu sinal alterado no momento do cálculo atráves da multiplicação pela variável _determinant\_signal_. Além disso, as instruções para a realização de logs, como aqueles realizados no item a, foram deixadas no corpo do código, por isso as chamadas para _print()_.

```python

import sys
import numpy as np
from scipy import linalg

# Configuração para gerar o log
np.set_printoptions(precision=6, linewidth=200, suppress=True)

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
```  

### Logs

#### Log do processo com 16 casas decimais de precisão, para  n = 4, demonstrando um exemplo onde há troca de linhas no escalonamento da matriz de Hilbert.

```python
A precisão utilizada é 1e-16
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 4
A matriz aumentada inicial A é: 
[[1.         0.5        0.33333333 0.25       2.08333333]
 [0.5        0.33333333 0.25       0.2        1.28333333]
 [0.33333333 0.25       0.2        0.16666667 0.95      ]
 [0.25       0.2        0.16666667 0.14285714 0.75952381]]

Após a 1a verificação de troca, não houve troca de linhas: 
[[1.         0.5        0.33333333 0.25       2.08333333]
 [0.5        0.33333333 0.25       0.2        1.28333333]
 [0.33333333 0.25       0.2        0.16666667 0.95      ]
 [0.25       0.2        0.16666667 0.14285714 0.75952381]]
*** Matriz no momento de troca da linha 2 ***
[[1.         0.5        0.33333333 0.25       2.08333333]
 [0.         0.08333333 0.08333333 0.075      0.24166667]
 [0.         0.08333333 0.08888889 0.08333333 0.25555556]
 [0.         0.075      0.08333333 0.08035714 0.23869048]]

Após a 2a verificação de troca, houve troca entre as linhas 2 e 3: 
[[1.         0.5        0.33333333 0.25       2.08333333]
 [0.         0.08333333 0.08888889 0.08333333 0.25555556]
 [0.         0.08333333 0.08333333 0.075      0.24166667]
 [0.         0.075      0.08333333 0.08035714 0.23869048]]
*** Matriz no momento de troca da linha 3 ***
[[ 1.          0.5         0.33333333  0.25        2.08333333]
 [ 0.          0.08333333  0.08888889  0.08333333  0.25555556]
 [ 0.          0.         -0.00555556 -0.00833333 -0.01388889]
 [ 0.          0.          0.00333333  0.00535714  0.00869048]]

Após a 3a verificação de troca, houve troca entre as linhas 3 e 4: 
[[ 1.          0.5         0.33333333  0.25        2.08333333]
 [ 0.          0.08333333  0.08888889  0.08333333  0.25555556]
 [ 0.          0.          0.00333333  0.00535714  0.00869048]
 [ 0.          0.         -0.00555556 -0.00833333 -0.01388889]]

Após a 4a verificação de troca, não houve troca de linhas: 
[[1.         0.5        0.33333333 0.25       2.08333333]
 [0.         0.08333333 0.08888889 0.08333333 0.25555556]
 [0.         0.         0.00333333 0.00535714 0.00869048]
 [0.         0.         0.         0.00059524 0.00059524]]
O determinante vale 0.0000001653
O vetor resultante é [1. 1. 1. 1.]
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999999999976796
A norma ideal é 4.0
A norma do vetor resultante é 4.000000000000034
Diferença entre as normas: 3.375077994860476e-14
``` 

#### Log do processo com 16 casas decimais de precisão, exibindo a aplicação do algoritmo à uma matriz 4x4 com entradas aleatórias (np.random.rand).

```python
A precisão utilizada é 1e-16
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE ORDEM 4 COM ENTRADAS ALEATÓRIAS
A matriz aumentada inicial A é: 
[[0.13446575 0.72418385 0.69022027 0.19263018 1.74150005]
 [0.23314232 0.47563839 0.26706262 0.73001839 1.70586172]
 [0.31962387 0.81546944 0.7618992  0.68640756 2.58340007]
 [0.36404771 0.10594377 0.70583653 0.77379825 1.94962627]]

*** Matriz no momento de troca da linha 1 ***
[[0.13446575 0.72418385 0.69022027 0.19263018 1.74150005]
 [0.23314232 0.47563839 0.26706262 0.73001839 1.70586172]
 [0.31962387 0.81546944 0.7618992  0.68640756 2.58340007]
 [0.36404771 0.10594377 0.70583653 0.77379825 1.94962627]]

Após a 1a verificação de troca, houve troca entre as linhas 1 e 4: 
[[0.36404771 0.10594377 0.70583653 0.77379825 1.94962627]
 [0.23314232 0.47563839 0.26706262 0.73001839 1.70586172]
 [0.31962387 0.81546944 0.7618992  0.68640756 2.58340007]
 [0.13446575 0.72418385 0.69022027 0.19263018 1.74150005]]

*** Matriz no momento de troca da linha 2 ***
[[ 0.36404771  0.10594377  0.70583653  0.77379825  1.94962627]
 [ 0.          0.4077902  -0.18496706  0.23446488  0.45728802]
 [ 0.          0.72245373  0.14219415  0.007034    0.87168187]
 [ 0.          0.68505214  0.42951038 -0.09318226  1.02138026]]

Após a 2a verificação de troca, houve troca entre as linhas 2 e 3: 
[[ 0.36404771  0.10594377  0.70583653  0.77379825  1.94962627]
 [ 0.          0.72245373  0.14219415  0.007034    0.87168187]
 [ 0.          0.4077902  -0.18496706  0.23446488  0.45728802]
 [ 0.          0.68505214  0.42951038 -0.09318226  1.02138026]]

*** Matriz no momento de troca da linha 3 ***
[[ 0.36404771  0.10594377  0.70583653  0.77379825  1.94962627]
 [ 0.          0.72245373  0.14219415  0.007034    0.87168187]
 [ 0.          0.         -0.26522879  0.23049453 -0.03473426]
 [ 0.          0.          0.29467765 -0.09985211  0.19482554]]

Após a 3a verificação de troca, houve troca entre as linhas 3 e 4: 
[[ 0.36404771  0.10594377  0.70583653  0.77379825  1.94962627]
 [ 0.          0.72245373  0.14219415  0.007034    0.87168187]
 [ 0.          0.          0.29467765 -0.09985211  0.19482554]
 [ 0.          0.         -0.26522879  0.23049453 -0.03473426]]

Após a 4a verificação de troca, não houve troca de linhas: 
[[ 0.36404771  0.10594377  0.70583653  0.77379825  1.94962627]
 [ 0.          0.72245373  0.14219415  0.007034    0.87168187]
 [ 0.          0.          0.29467765 -0.09985211  0.19482554]
 [ 0.          0.          0.          0.14062123  0.14062123]]
O determinante vale -0.0108984923
O vetor resultante é [1. 1. 1. 1.]
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999999999991340
A norma ideal é 4.0
A norma do vetor resultante é 3.9999999999999902
Diferença entre as normas: -9.769962616701378e-15
```

#### Log do processo com 16 casas decimais de precisão, para  n = 2, 4, ... , 12.
```python
A precisão utilizada é 1e-16
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 2
O determinante vale 0.0833333333
O vetor resultante é [1.0000000000000004 0.9999999999999993]
O primeiro "1." mostrado pelo Python, na verdade, é 1.000000000000000444
A norma ideal é 2.0
A norma do vetor resultante é 1.9999999999999996
Diferença entre as normas: -4.440892098500626e-16
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 4
O determinante vale 0.0000001653
O vetor resultante é [0.9999999999999768 1.000000000000256  0.9999999999993879 1.0000000000003963]
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999999999976796
A norma ideal é 4.0
A norma do vetor resultante é 4.000000000000034
Diferença entre as normas: 3.375077994860476e-14
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 6
O determinante vale 0.0000000000
O vetor resultante é [0.9999999999990724 1.0000000000266975 0.9999999998182728 1.0000000004748553 0.9999999994739818 1.0000000002078646]
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999999999072409
A norma ideal é 6.0
A norma do vetor resultante é 6.000000000001489
Diferença entre as normas: 1.4885870314174099e-12
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 8
O determinante vale 0.0000000000
O vetor resultante é [0.9999999999717035 1.0000000015083135 0.9999999803342513 1.000000106424826  0.999999713308501  1.0000004059917933 0.9999997108174259 1.0000000816638364]
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999999971703524
A norma ideal é 8.0
A norma do vetor resultante é 8.000000000041652
Diferença entre as normas: 4.165201517025707e-11
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 10
O determinante vale 0.0000000000
O vetor resultante é [0.9999999985728525 1.0000001203304385 0.9999974848388027 1.0000225232616968 0.9998938911782738 1.000288677455108  0.9995305432724556 1.000450221626113  0.9997652083919379 1.0000513318870956]
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999998572852489
A norma ideal é 10.0
A norma do vetor resultante é 10.000000577588136
Diferença entre as normas: 5.775881355418733e-07
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 12
O determinante vale 0.0000000000
O vetor resultante é [0.9999999429965561 1.000007084854173  0.9997806729974179 1.0029496989415063 0.9786115232101207 1.0931018343504375 0.7426838293059917 1.462491811781697  0.4611633218672396 1.392445105106977  0.8376438609336166 1.0291213456208856]
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999942996556146
A norma ideal é 12.0
A norma do vetor resultante é 12.760810213863056
Diferença entre as normas: 0.7608102138630564
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 13
O determinante vale 0.0000000000
O vetor resultante é [ 1.          0.99999856  1.00008291  0.99827934  1.01782144  0.8930774   1.40220178  0.01120354  2.61360819 -0.73302435  2.17717888  0.54160844
  1.07796391]
O primeiro "1." mostrado pelo Python, na verdade, é 1.000000002888341122
A norma ideal é 13.0
A norma do vetor resultante é 21.360293893532997
Diferença entre as normas: 8.360293893532997
```

* Dissertar sobre a continuação dos erros graves devido a arrendodamentos -> particularidade da Matriz de Hilbert

* É evidente que os erros acumulados de arredondamento continuam maiores que o esperado (para n = 13, diferença de quase 10 vezes na norma dos vetores solução adquirida e solução ideal). Entretanto, é necessário levar em conta que a matriz de Hilbert contém números extremamente pequenos para dimensões maiores, ou seja, é esperado que, com um maior número de operações, pequenos erros acumulem-se e gerem efeitos formidavelmente negativos no final dos cálculos. 
  * Como um carro indo reto numa rodovia muito extensa: o veículo passa sobre uma falha no asfalto que o força a virar 3° para a esquerda. Caso o veículo mantenha-se nessa nova direção por tempo suficiente, o carro baterá no acostamento em algum momento.  
  Dado esse efeito a longo prazo, o intuito de medidas numéricas, como a troca de linhas em busca do maior pivô, seria como tentar diminuir a variação do ângulo de direção que o carro sofre. Por exemplo, uma troca de linhas poderia diminuir a nova direção para 0.75° a esquerda da direção original e, dessa forma, talvez desse tempo da rodovia terminar sem o carro bater no acostamento.

* De acordo com os logs, a matriz de Hilbert se aproxima cada vez mais da **singularidade** conforme sua dimensão aumenta. Essa percepção torna-se verdadeira ao levar em conta que os elementos da diagonal principal só decrescem em tamanho, sempre menores que 1, conforme o crescimento da dimensão.

## Parte 2



