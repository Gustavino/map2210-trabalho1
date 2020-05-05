# Trabalho 1 - MAP2210 - Aplicações de Álgebra Linear

## Parte 1

> a. Código, comentado, em Python para resolução de sistemas lineares pelo método da Eliminação de Gauss sem pivotamento.  
  
  * Os comentários iniciados com "STEP" se referem ao algoritmo descrito no livro Numerical Analysis. Da mesma maneira, para manter conformidade com o idioma do livro-texto, diversos nomes de variáveis e funções estão em inglês (assim como algumas docstrings, descrições no início de funções).  
  * Em alguns pontos do código é utilizada, para verificar se um número é zero, a função _abs_, que entrega o módulo de um número (int, float).  
    * Exemplo: 
    ```python  
    precision = 0.00001
    abs(number) < precision # Nesse caso, são utilizadas cinco casas decimais para verificar se um número está próximo de zero.
    ```  
  * A precisão definida no início do código é fundamental ao seu correto funcionamento: na demonstração abaixo, **a precisão limita a verificação do algoritmo até n = 16**.
```python  
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
  

**Sobre o vetor _x_ (solução)**:  
* Acerca de suas componentes finais: observando-se os vetores resultantes até a terceira iteração, sobre a matriz de ordem 6, fica claro que os problemas decorrentes do má condicionamento da matriz de Hilbert ainda não alteraram drasticamente os resultados. Entretanto, já nota-se uma diferença: a presença do "." após o número um, como em "1.", no vetor resultante indica a presença de arrendondamento em casas decimais muito distantes da vírgula (certamente após a oitava casa decimal, que é o padrão de exibição dos valores de np.darray ou listas do Python).  
Dessa forma, da matriz de ordem 8 em diante, ficam claras as consequências incovenientes dos arrendondamentos anteriores sobre o vetor solução.

* Acerca de sua norma: como consequência das alterações sobre as componentes, a norma do vetor x nunca é exatamenge igual a ideal. De acordo com os logs, a diferença entre as normas produzida pelo algoritmo e ideal varia desde um número extremamente pequeno (a ponto de ocorrer um _floating point overflow_) até 15.25 vezes a norma ideal para n=16. Ou seja, quanto mais aumenta-se a ordem da matriz, mais aumenta-se a presença do erro no algoritmo.  

### Gráfico da norma 2. Quanto maior o valor no eixo y, maior o erro.  

!["Naive Gaussian elimination error"](development/graphs/error%20naive%20gaussian.png?raw=true)  

>b. Código estendendo a solução do item _a_ com o processo de **pivotamento**.  

* O código a seguir contém as mudanças necessárias ao pivotamento: na procura do pivô da i-ésima linha, é feita uma busca linear pelo maior elemento entre os elementos A[i][j], com **0 <= i <= n-1** e **i <= j <= n-1**. Se o pivô encontrado não estiver na linha i, há uma troca das linhas i e a linha com o maior pivô. Assim, o determinante terá o seu sinal alterado no momento de seu cálculo atráves da multiplicação pela variável _determinant\_signal_. 
* As instruções para a realização de logs, como aqueles realizados no item a, foram deixadas no corpo do código, por isso as chamadas para _print()_.

```python

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
```  

### Logs

#### Log do processo com 16 casas decimais de precisão, para  n = 4, demonstrando que há troca de linhas no escalonamento da matriz de Hilbert.

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

Após a 2a verificação de troca, não houve troca de linhas: 
[[1.         0.5        0.33333333 0.25       2.08333333]
 [0.         0.08333333 0.08333333 0.075      0.24166667]
 [0.         0.08333333 0.08888889 0.08333333 0.25555556]
 [0.         0.075      0.08333333 0.08035714 0.23869048]]

*** Matriz no momento de troca da linha 3 ***
[[1.         0.5        0.33333333 0.25       2.08333333]
 [0.         0.08333333 0.08333333 0.075      0.24166667]
 [0.         0.         0.00555556 0.00833333 0.01388889]
 [0.         0.         0.00833333 0.01285714 0.02119048]]

Após a 3a verificação de troca, houve troca entre as linhas 3 e 4: 
[[1.         0.5        0.33333333 0.25       2.08333333]
 [0.         0.08333333 0.08333333 0.075      0.24166667]
 [0.         0.         0.00833333 0.01285714 0.02119048]
 [0.         0.         0.00555556 0.00833333 0.01388889]]

Após a 4a verificação de troca, não houve troca de linhas: 
[[ 1.          0.5         0.33333333  0.25        2.08333333]
 [ 0.          0.08333333  0.08333333  0.075       0.24166667]
 [ 0.          0.          0.00833333  0.01285714  0.02119048]
 [ 0.          0.          0.         -0.0002381  -0.0002381 ]]

O determinante vale 0.0000001653
O vetor resultante é [1. 1. 1. 1.]
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999999999977907
A norma ideal é 4.0
A norma do vetor resultante é 4.000000000000034
A norma 2 é: 5.633769716366803e-25
O número de troca de linhas foi: 1
```

#### Log do processo com 16 casas decimais de precisão, para  n = 7, demonstrando que há troca de linhas no escalonamento da matriz de Hilbert.

```python
A precisão utilizada é 1e-16
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 7
A matriz aumentada inicial A é: 
[[1.         0.5        0.33333333 0.25       0.2        0.16666667 0.14285714 2.59285714]
 [0.5        0.33333333 0.25       0.2        0.16666667 0.14285714 0.125      1.71785714]
 [0.33333333 0.25       0.2        0.16666667 0.14285714 0.125      0.11111111 1.32896825]
 [0.25       0.2        0.16666667 0.14285714 0.125      0.11111111 0.1        1.09563492]
 [0.2        0.16666667 0.14285714 0.125      0.11111111 0.1        0.09090909 0.93654401]
 [0.16666667 0.14285714 0.125      0.11111111 0.1        0.09090909 0.08333333 0.81987734]
 [0.14285714 0.125      0.11111111 0.1        0.09090909 0.08333333 0.07692308 0.73013376]]

Após a 1a verificação de troca, não houve troca de linhas: 
[[1.         0.5        0.33333333 0.25       0.2        0.16666667 0.14285714 2.59285714]
 [0.5        0.33333333 0.25       0.2        0.16666667 0.14285714 0.125      1.71785714]
 [0.33333333 0.25       0.2        0.16666667 0.14285714 0.125      0.11111111 1.32896825]
 [0.25       0.2        0.16666667 0.14285714 0.125      0.11111111 0.1        1.09563492]
 [0.2        0.16666667 0.14285714 0.125      0.11111111 0.1        0.09090909 0.93654401]
 [0.16666667 0.14285714 0.125      0.11111111 0.1        0.09090909 0.08333333 0.81987734]
 [0.14285714 0.125      0.11111111 0.1        0.09090909 0.08333333 0.07692308 0.73013376]]

Após a 2a verificação de troca, não houve troca de linhas: 
[[1.         0.5        0.33333333 0.25       0.2        0.16666667 0.14285714 2.59285714]
 [0.         0.08333333 0.08333333 0.075      0.06666667 0.05952381 0.05357143 0.42142857]
 [0.         0.08333333 0.08888889 0.08333333 0.07619048 0.06944444 0.06349206 0.46468254]
 [0.         0.075      0.08333333 0.08035714 0.075      0.06944444 0.06428571 0.44742063]
 [0.         0.06666667 0.07619048 0.075      0.07111111 0.06666667 0.06233766 0.41797258]
 [0.         0.05952381 0.06944444 0.06944444 0.06666667 0.06313131 0.05952381 0.38773449]
 [0.         0.05357143 0.06349206 0.06428571 0.06233766 0.05952381 0.05651491 0.35972559]]

*** Matriz no momento de troca da linha 3 ***
[[1.         0.5        0.33333333 0.25       0.2        0.16666667 0.14285714 2.59285714]
 [0.         0.08333333 0.08333333 0.075      0.06666667 0.05952381 0.05357143 0.42142857]
 [0.         0.         0.00555556 0.00833333 0.00952381 0.00992063 0.00992063 0.04325397]
 [0.         0.         0.00833333 0.01285714 0.015      0.01587302 0.01607143 0.06813492]
 [0.         0.         0.00952381 0.015      0.01777778 0.01904762 0.01948052 0.08082973]
 [0.         0.         0.00992063 0.01587302 0.01904762 0.02061431 0.0212585  0.08671408]
 [0.         0.         0.00992063 0.01607143 0.01948052 0.0212585  0.02207614 0.08880722]]

Após a 3a verificação de troca, houve troca entre as linhas 3 e 6: 
[[1.         0.5        0.33333333 0.25       0.2        0.16666667 0.14285714 2.59285714]
 [0.         0.08333333 0.08333333 0.075      0.06666667 0.05952381 0.05357143 0.42142857]
 [0.         0.         0.00992063 0.01587302 0.01904762 0.02061431 0.0212585  0.08671408]
 [0.         0.         0.00833333 0.01285714 0.015      0.01587302 0.01607143 0.06813492]
 [0.         0.         0.00952381 0.015      0.01777778 0.01904762 0.01948052 0.08082973]
 [0.         0.         0.00555556 0.00833333 0.00952381 0.00992063 0.00992063 0.04325397]
 [0.         0.         0.00992063 0.01607143 0.01948052 0.0212585  0.02207614 0.08880722]]

*** Matriz no momento de troca da linha 4 ***
[[ 1.          0.5         0.33333333  0.25        0.2         0.16666667  0.14285714  2.59285714]
 [ 0.          0.08333333  0.08333333  0.075       0.06666667  0.05952381  0.05357143  0.42142857]
 [ 0.          0.          0.00992063  0.01587302  0.01904762  0.02061431  0.0212585   0.08671408]
 [ 0.          0.          0.         -0.00047619 -0.001      -0.001443   -0.00178571 -0.00470491]
 [ 0.          0.          0.         -0.0002381  -0.00050794 -0.00074212 -0.00092764 -0.00241579]
 [ 0.          0.          0.         -0.00055556 -0.00114286 -0.00162338 -0.00198413 -0.00530592]
 [ 0.          0.          0.          0.00019841  0.0004329   0.0006442   0.00081763  0.00209314]]

Após a 4a verificação de troca, houve troca entre as linhas 4 e 7: 
[[ 1.          0.5         0.33333333  0.25        0.2         0.16666667  0.14285714  2.59285714]
 [ 0.          0.08333333  0.08333333  0.075       0.06666667  0.05952381  0.05357143  0.42142857]
 [ 0.          0.          0.00992063  0.01587302  0.01904762  0.02061431  0.0212585   0.08671408]
 [ 0.          0.          0.          0.00019841  0.0004329   0.0006442   0.00081763  0.00209314]
 [ 0.          0.          0.         -0.0002381  -0.00050794 -0.00074212 -0.00092764 -0.00241579]
 [ 0.          0.          0.         -0.00055556 -0.00114286 -0.00162338 -0.00198413 -0.00530592]
 [ 0.          0.          0.         -0.00047619 -0.001      -0.001443   -0.00178571 -0.00470491]]

*** Matriz no momento de troca da linha 5 ***
[[1.         0.5        0.33333333 0.25       0.2        0.16666667 0.14285714 2.59285714]
 [0.         0.08333333 0.08333333 0.075      0.06666667 0.05952381 0.05357143 0.42142857]
 [0.         0.         0.00992063 0.01587302 0.01904762 0.02061431 0.0212585  0.08671408]
 [0.         0.         0.         0.00019841 0.0004329  0.0006442  0.00081763 0.00209314]
 [0.         0.         0.         0.         0.00001154 0.00003092 0.00005352 0.00009598]
 [0.         0.         0.         0.         0.00006926 0.00018038 0.00030525 0.00055489]
 [0.         0.         0.         0.         0.00003896 0.00010307 0.00017661 0.00031864]]

Após a 5a verificação de troca, houve troca entre as linhas 5 e 6: 
[[1.         0.5        0.33333333 0.25       0.2        0.16666667 0.14285714 2.59285714]
 [0.         0.08333333 0.08333333 0.075      0.06666667 0.05952381 0.05357143 0.42142857]
 [0.         0.         0.00992063 0.01587302 0.01904762 0.02061431 0.0212585  0.08671408]
 [0.         0.         0.         0.00019841 0.0004329  0.0006442  0.00081763 0.00209314]
 [0.         0.         0.         0.         0.00006926 0.00018038 0.00030525 0.00055489]
 [0.         0.         0.         0.         0.00001154 0.00003092 0.00005352 0.00009598]
 [0.         0.         0.         0.         0.00003896 0.00010307 0.00017661 0.00031864]]

*** Matriz no momento de troca da linha 6 ***
[[1.         0.5        0.33333333 0.25       0.2        0.16666667 0.14285714 2.59285714]
 [0.         0.08333333 0.08333333 0.075      0.06666667 0.05952381 0.05357143 0.42142857]
 [0.         0.         0.00992063 0.01587302 0.01904762 0.02061431 0.0212585  0.08671408]
 [0.         0.         0.         0.00019841 0.0004329  0.0006442  0.00081763 0.00209314]
 [0.         0.         0.         0.         0.00006926 0.00018038 0.00030525 0.00055489]
 [0.         0.         0.         0.         0.         0.00000086 0.00000264 0.0000035 ]
 [0.         0.         0.         0.         0.         0.00000161 0.00000491 0.00000652]]

Após a 6a verificação de troca, houve troca entre as linhas 6 e 7: 
[[1.         0.5        0.33333333 0.25       0.2        0.16666667 0.14285714 2.59285714]
 [0.         0.08333333 0.08333333 0.075      0.06666667 0.05952381 0.05357143 0.42142857]
 [0.         0.         0.00992063 0.01587302 0.01904762 0.02061431 0.0212585  0.08671408]
 [0.         0.         0.         0.00019841 0.0004329  0.0006442  0.00081763 0.00209314]
 [0.         0.         0.         0.         0.00006926 0.00018038 0.00030525 0.00055489]
 [0.         0.         0.         0.         0.         0.00000161 0.00000491 0.00000652]
 [0.         0.         0.         0.         0.         0.00000086 0.00000264 0.0000035 ]]

Após a 7a verificação de troca, não houve troca de linhas: 
[[1.         0.5        0.33333333 0.25       0.2        0.16666667 0.14285714 2.59285714]
 [0.         0.08333333 0.08333333 0.075      0.06666667 0.05952381 0.05357143 0.42142857]
 [0.         0.         0.00992063 0.01587302 0.01904762 0.02061431 0.0212585  0.08671408]
 [0.         0.         0.         0.00019841 0.0004329  0.0006442  0.00081763 0.00209314]
 [0.         0.         0.         0.         0.00006926 0.00018038 0.00030525 0.00055489]
 [0.         0.         0.         0.         0.         0.00000161 0.00000491 0.00000652]
 [0.         0.         0.         0.         0.         0.         0.00000003 0.00000003]]

O determinante vale 0.0000000000
O vetor resultante é [1.         1.         1.         1.00000001 0.99999998 1.00000001 1.        ]
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999999994484190
A norma ideal é 7.0
A norma do vetor resultante é 6.999999999992109
A norma 2 é: 4.919551313660094e-16
O número de troca de linhas foi: 4
``` 

#### Log do processo com 16 casas decimais de precisão, exibindo a aplicação do algoritmo à uma matriz 4x4 com entradas aleatórias (np.random.rand).

```python
A precisão utilizada é 1e-16
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 7
A matriz aumentada inicial A é: 
[[0.10950542 0.94012988 0.99858117 0.62075364 0.4948885  0.21548038 0.30060237 3.67994136]
 [0.6385048  0.99827153 0.27298437 0.48949654 0.99502634 0.8162339  0.02005824 4.23057571]
 [0.74888908 0.18865875 0.18552867 0.98068187 0.38835878 0.04007095 0.38502926 2.91721736]
 [0.54902272 0.6237597  0.46110593 0.97211545 0.16773326 0.5525978  0.56508593 3.89142079]
 [0.22088494 0.80929599 0.95924166 0.22150904 0.94837869 0.53194435 0.39153288 4.08278755]
 [0.92414251 0.97556572 0.60545753 0.3956732  0.2811925  0.87972155 0.79444446 4.85619747]
 [0.65070276 0.45005237 0.47416269 0.53905751 0.96207135 0.74839582 0.91542128 4.73986378]]

*** Matriz no momento de troca da linha 1 ***
[[0.10950542 0.94012988 0.99858117 0.62075364 0.4948885  0.21548038 0.30060237 3.67994136]
 [0.6385048  0.99827153 0.27298437 0.48949654 0.99502634 0.8162339  0.02005824 4.23057571]
 [0.74888908 0.18865875 0.18552867 0.98068187 0.38835878 0.04007095 0.38502926 2.91721736]
 [0.54902272 0.6237597  0.46110593 0.97211545 0.16773326 0.5525978  0.56508593 3.89142079]
 [0.22088494 0.80929599 0.95924166 0.22150904 0.94837869 0.53194435 0.39153288 4.08278755]
 [0.92414251 0.97556572 0.60545753 0.3956732  0.2811925  0.87972155 0.79444446 4.85619747]
 [0.65070276 0.45005237 0.47416269 0.53905751 0.96207135 0.74839582 0.91542128 4.73986378]]

Após a 1a verificação de troca, houve troca entre as linhas 1 e 6: 
[[0.92414251 0.97556572 0.60545753 0.3956732  0.2811925  0.87972155 0.79444446 4.85619747]
 [0.6385048  0.99827153 0.27298437 0.48949654 0.99502634 0.8162339  0.02005824 4.23057571]
 [0.74888908 0.18865875 0.18552867 0.98068187 0.38835878 0.04007095 0.38502926 2.91721736]
 [0.54902272 0.6237597  0.46110593 0.97211545 0.16773326 0.5525978  0.56508593 3.89142079]
 [0.22088494 0.80929599 0.95924166 0.22150904 0.94837869 0.53194435 0.39153288 4.08278755]
 [0.10950542 0.94012988 0.99858117 0.62075364 0.4948885  0.21548038 0.30060237 3.67994136]
 [0.65070276 0.45005237 0.47416269 0.53905751 0.96207135 0.74839582 0.91542128 4.73986378]]

*** Matriz no momento de troca da linha 2 ***
[[ 0.92414251  0.97556572  0.60545753  0.3956732   0.2811925   0.87972155  0.79444446  4.85619747]
 [ 0.          0.32423762 -0.14533589  0.21611961  0.80074596  0.20842026 -0.52883611  0.87535145]
 [ 0.         -0.6019017  -0.30511052  0.66004373  0.16049133 -0.67282111 -0.25875757 -1.01805583]
 [ 0.          0.04418703  0.10141038  0.73705043  0.00067994  0.02996507  0.09311536  1.00640822]
 [ 0.          0.57612008  0.81452756  0.12693678  0.88116915  0.32167674  0.20164787  2.92207818]
 [ 0.          0.82453112  0.92683804  0.5738687   0.46156885  0.11123858  0.2064654   3.10451069]
 [ 0.         -0.23685825  0.04785086  0.26045801  0.76407944  0.12897053  0.35604089  1.32054149]]

Após a 2a verificação de troca, houve troca entre as linhas 2 e 6: 
[[ 0.92414251  0.97556572  0.60545753  0.3956732   0.2811925   0.87972155  0.79444446  4.85619747]
 [ 0.          0.82453112  0.92683804  0.5738687   0.46156885  0.11123858  0.2064654   3.10451069]
 [ 0.         -0.6019017  -0.30511052  0.66004373  0.16049133 -0.67282111 -0.25875757 -1.01805583]
 [ 0.          0.04418703  0.10141038  0.73705043  0.00067994  0.02996507  0.09311536  1.00640822]
 [ 0.          0.57612008  0.81452756  0.12693678  0.88116915  0.32167674  0.20164787  2.92207818]
 [ 0.          0.32423762 -0.14533589  0.21611961  0.80074596  0.20842026 -0.52883611  0.87535145]
 [ 0.         -0.23685825  0.04785086  0.26045801  0.76407944  0.12897053  0.35604089  1.32054149]]

Após a 3a verificação de troca, não houve troca de linhas: 
[[ 0.92414251  0.97556572  0.60545753  0.3956732   0.2811925   0.87972155  0.79444446  4.85619747]
 [ 0.          0.82453112  0.92683804  0.5738687   0.46156885  0.11123858  0.2064654   3.10451069]
 [ 0.          0.          0.37147448  1.07896369  0.49743322 -0.59161776 -0.10803933  1.24821431]
 [ 0.          0.          0.05174068  0.70629653 -0.02405576  0.02400374  0.08205078  0.84003597]
 [ 0.          0.          0.16692313 -0.27403933  0.5586597   0.24395162  0.05738544  0.75288055]
 [ 0.          0.         -0.50980457 -0.00954782  0.61923918  0.16467693 -0.61002631 -0.34546259]
 [ 0.          0.          0.31409823  0.42530993  0.89667164  0.16092539  0.41535101  2.21235619]]

*** Matriz no momento de troca da linha 4 ***
[[ 0.92414251  0.97556572  0.60545753  0.3956732   0.2811925   0.87972155  0.79444446  4.85619747]
 [ 0.          0.82453112  0.92683804  0.5738687   0.46156885  0.11123858  0.2064654   3.10451069]
 [ 0.          0.          0.37147448  1.07896369  0.49743322 -0.59161776 -0.10803933  1.24821431]
 [ 0.          0.          0.          0.55601349 -0.09334055  0.10640699  0.097099    0.66617893]
 [ 0.          0.          0.         -0.75887478  0.33513665  0.50979676  0.10593322  0.19199185]
 [ 0.          0.          0.          1.47120158  1.30190714 -0.64724811 -0.75829745  1.36756316]
 [ 0.          0.          0.         -0.48700196  0.47606968  0.66116458  0.50670307  1.15693537]]

Após a 4a verificação de troca, houve troca entre as linhas 4 e 6: 
[[ 0.92414251  0.97556572  0.60545753  0.3956732   0.2811925   0.87972155  0.79444446  4.85619747]
 [ 0.          0.82453112  0.92683804  0.5738687   0.46156885  0.11123858  0.2064654   3.10451069]
 [ 0.          0.          0.37147448  1.07896369  0.49743322 -0.59161776 -0.10803933  1.24821431]
 [ 0.          0.          0.          1.47120158  1.30190714 -0.64724811 -0.75829745  1.36756316]
 [ 0.          0.          0.         -0.75887478  0.33513665  0.50979676  0.10593322  0.19199185]
 [ 0.          0.          0.          0.55601349 -0.09334055  0.10640699  0.097099    0.66617893]
 [ 0.          0.          0.         -0.48700196  0.47606968  0.66116458  0.50670307  1.15693537]]

Após a 5a verificação de troca, não houve troca de linhas: 
[[ 0.92414251  0.97556572  0.60545753  0.3956732   0.2811925   0.87972155  0.79444446  4.85619747]
 [ 0.          0.82453112  0.92683804  0.5738687   0.46156885  0.11123858  0.2064654   3.10451069]
 [ 0.          0.          0.37147448  1.07896369  0.49743322 -0.59161776 -0.10803933  1.24821431]
 [ 0.          0.          0.          1.47120158  1.30190714 -0.64724811 -0.75829745  1.36756316]
 [ 0.          0.          0.          0.          1.00668602  0.17593343 -0.28521155  0.89740789]
 [ 0.          0.          0.          0.         -0.58537233  0.35102247  0.38368353  0.14933367]
 [ 0.          0.          0.          0.          0.90703124  0.44691039  0.25568897  1.6096306 ]]

Após a 6a verificação de troca, não houve troca de linhas: 
[[ 0.92414251  0.97556572  0.60545753  0.3956732   0.2811925   0.87972155  0.79444446  4.85619747]
 [ 0.          0.82453112  0.92683804  0.5738687   0.46156885  0.11123858  0.2064654   3.10451069]
 [ 0.          0.          0.37147448  1.07896369  0.49743322 -0.59161776 -0.10803933  1.24821431]
 [ 0.          0.          0.          1.47120158  1.30190714 -0.64724811 -0.75829745  1.36756316]
 [ 0.          0.          0.          0.          1.00668602  0.17593343 -0.28521155  0.89740789]
 [ 0.          0.          0.          0.          0.          0.45332503  0.21783742  0.67116246]
 [ 0.          0.          0.          0.          0.          0.28839312  0.51266661  0.80105973]]

Após a 7a verificação de troca, não houve troca de linhas: 
[[ 0.92414251  0.97556572  0.60545753  0.3956732   0.2811925   0.87972155  0.79444446  4.85619747]
 [ 0.          0.82453112  0.92683804  0.5738687   0.46156885  0.11123858  0.2064654   3.10451069]
 [ 0.          0.          0.37147448  1.07896369  0.49743322 -0.59161776 -0.10803933  1.24821431]
 [ 0.          0.          0.          1.47120158  1.30190714 -0.64724811 -0.75829745  1.36756316]
 [ 0.          0.          0.          0.          1.00668602  0.17593343 -0.28521155  0.89740789]
 [ 0.          0.          0.          0.          0.          0.45332503  0.21783742  0.67116246]
 [ 0.          0.          0.          0.          0.          0.          0.37408433  0.37408433]]

O determinante vale -0.0710919495
O vetor resultante é [1. 1. 1. 1. 1. 1. 1.]
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999999999999556
A norma ideal é 7.0
A norma do vetor resultante é 7.000000000000001
A norma 2 é: 2.1755304651798216e-29
O número de troca de linhas foi: 3
```

#### Log do processo com 16 casas decimais de precisão, para  n = 2, 4, ... , 12.
```python
A precisão utilizada é 1e-16
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 2

O determinante vale 0.0833333333
O vetor resultante é [1. 1.]
O primeiro "1." mostrado pelo Python, na verdade, é 1.000000000000000444
A norma ideal é 2.0
A norma do vetor resultante é 1.9999999999999996
A norma 2 é: 6.409494854920721e-31
O número de troca de linhas foi: 0
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 4

O determinante vale 0.0000001653
O vetor resultante é [1. 1. 1. 1.]
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999999999977907
A norma ideal é 4.0
A norma do vetor resultante é 4.000000000000034
A norma 2 é: 5.633769716366803e-25
O número de troca de linhas foi: 1
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 6

O determinante vale 0.0000000000
O vetor resultante é [1. 1. 1. 1. 1. 1.]
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999999999301115
A norma ideal é 6.0
A norma do vetor resultante é 6.000000000001085
A norma 2 é: 3.1458024884821903e-19
O número de troca de linhas foi: 4
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 8

O determinante vale 0.0000000000
O vetor resultante é [1.         1.         0.99999999 1.00000007 0.99999981 1.00000028 0.9999998  1.00000006]
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999999980144438
A norma ideal é 8.0
A norma do vetor resultante é 8.000000000027464
A norma 2 é: 1.6026804802031263e-13
O número de troca de linhas foi: 9
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 10

O determinante vale 0.0000000000
O vetor resultante é [1.         1.00000011 0.99999759 1.00002161 0.99989797 1.00027809 0.99954705 1.00043497 0.9997729  1.0000497 ]
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999998646136534
A norma ideal é 10.0
A norma do vetor resultante é 10.00000053822474
A norma 2 é: 5.366236287173831e-07
O número de troca de linhas foi: 16
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 12

O determinante vale 0.0000000000
O vetor resultante é [0.99999994 1.00000762 0.99976287 1.00320232 0.97669954 1.10172273 0.71814492 1.50771499 0.40732815 1.4323956  0.82084215 1.0321792 ]
O primeiro "1." mostrado pelo Python, na verdade, é 0.999999939052567344
A norma ideal é 12.0
A norma do vetor resultante é 12.919476516234003
A norma 2 é: 0.9194764437612212
O número de troca de linhas foi: 25
--------------------------------------------------------------------------
A precisão utilizada é 1e-16
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 13

O determinante vale 0.0000000000
O vetor resultante é [ 1.00000002  0.99999677  1.0001479   0.9972467   1.02673091  0.84639642  1.5601826  -0.34537935  3.15569814 -1.28133924  2.53090874  0.4099167
  1.09949372]
O primeiro "1." mostrado pelo Python, na verdade, é 1.000000015054611779
A norma ideal é 13.0
A norma do vetor resultante é 27.701488474640623
A norma 2 é: 14.701488393459883
O número de troca de linhas foi: 10

```

* É evidente que os erros acumulados de arredondamento continuam maiores que o esperado (para n = 13, diferença de quase 10 vezes na norma dos vetores solução adquirida e solução ideal). Entretanto, é necessário levar em conta que a matriz de Hilbert contém números extremamente pequenos para dimensões maiores, ou seja, é esperado que, com um maior número de operações, pequenos erros acumulem-se e gerem efeitos formidavelmente negativos no final dos cálculos. 
  * Como um carro indo reto numa rodovia muito extensa: o veículo passa sobre uma falha no asfalto que o força a virar 3° para a esquerda. Caso o veículo mantenha-se nessa nova direção por tempo suficiente, o carro baterá no acostamento em algum momento.  
  Dado esse efeito a longo prazo, o intuito de medidas numéricas, como a troca de linhas em busca do maior pivô, seria como tentar diminuir a variação do ângulo de direção que o carro sofre. Por exemplo, uma troca de linhas poderia diminuir a nova direção para 0.75° a esquerda da direção original e, dessa forma, talvez desse tempo da rodovia terminar sem o carro bater no acostamento.

* De acordo com os logs, a matriz de Hilbert se aproxima cada vez mais da **singularidade** conforme sua dimensão aumenta. Essa percepção torna-se verdadeira ao levar em conta que os elementos da diagonal principal só decrescem em tamanho, sempre menores que 1, conforme o crescimento da dimensão.

### Gráfico da norma 2. Quanto maior o valor no eixo y, maior o erro.  

!["Gaussian elimination with pivoting error"](development/graphs/error%20gaussian%20with%20pivoting.png?raw=true)  

## Parte 2

> Implementação do algoritmo de decomposição de Cholesky, como escrito no livro Numerical Analysis, em Python:

```python
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
```

* O algoritmo de Cholesky implementado transforma uma matriz A em uma matriz L, que, após operações de substituição (com o uso de sua transposta), devolve um vetor, solução, x. As principais vantagens desse procedimentos são a velocidade de execução e a precisão que é mantida.
    * Velocidade de execução: o termo dominante da fatorização de Cholesky é _1/6\*n³_. Para efeito de comparação, a velocidade do Naive Gauss é proporcinal a _1/3\*n³_. Ou seja, em uma análise não-assintótica, Cholesky é duas vezes mais rápido que o Naive Gauss.
    * Precisão: dado que as únicas divisões efetuadas no algoritmo são feitas por números da diagonal principal, que, pelas propriedades das matrizes SPD (symmetric, positive and definite), são sempre os maiores da matriz, há uma tendência de não gerar números que explodem em crescimento após a operação.

### Logs

#### Log comparando Naive Gauss, Cholesky e numpy.linalg.solve na resolução de matrizes de Hilbert
```python
A precisão utilizada é 1e-16
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 1
O erro da Eliminação Gaussiana sem pivotamento é 0.0
O erro da resolução com matrizes fatoradas por Cholesky é 0.0
O erro da biblioteca NumPy, através do np.linalg.solve, é 0.0
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 2
O erro da Eliminação Gaussiana sem pivotamento é 8.005932084973442e-16
O erro da resolução com matrizes fatoradas por Cholesky é 8.95090418262362e-16
O erro da biblioteca NumPy, através do np.linalg.solve, é 8.005932084973442e-16
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 3
O erro da Eliminação Gaussiana sem pivotamento é 1.7634731639015542e-14
O erro da resolução com matrizes fatoradas por Cholesky é 8.3820000221454525e-16
O erro da biblioteca NumPy, através do np.linalg.solve, é 1.3516024908118682e-14
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 4
O erro da Eliminação Gaussiana sem pivotamento é 7.555898631878062e-13
O erro da resolução com matrizes fatoradas por Cholesky é 2.9489011596635683e-13
O erro da biblioteca NumPy, através do np.linalg.solve, é 9.543859542259525e-13
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 5
O erro da Eliminação Gaussiana sem pivotamento é 3.4726644211552867e-12
O erro da resolução com matrizes fatoradas por Cholesky é 6.350949804677493e-12
O erro da biblioteca NumPy, através do np.linalg.solve, é 1.7400885594819817e-12
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 6
O erro da Eliminação Gaussiana sem pivotamento é 6.164839936768172e-10
O erro da resolução com matrizes fatoradas por Cholesky é 1.221549333506457e-09
O erro da biblioteca NumPy, através do np.linalg.solve, é 1.0652555092866373e-09
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 7
O erro da Eliminação Gaussiana sem pivotamento é 2.2290975050937107e-08
O erro da resolução com matrizes fatoradas por Cholesky é 2.1731152578348703e-08
O erro da biblioteca NumPy, através do np.linalg.solve, é 3.751318450290021e-08
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 8
O erro da Eliminação Gaussiana sem pivotamento é 4.515098799979707e-07
O erro da resolução com matrizes fatoradas por Cholesky é 2.7898483285077586e-07
O erro da biblioteca NumPy, através do np.linalg.solve, é 4.944033795675855e-07
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 9
O erro da Eliminação Gaussiana sem pivotamento é 1.638830755145328e-05
O erro da resolução com matrizes fatoradas por Cholesky é 5.9934486699236576e-05
O erro da biblioteca NumPy, através do np.linalg.solve, é 3.0371306931003123e-05
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 10
O erro da Eliminação Gaussiana sem pivotamento é 0.0008408643847455104
O erro da resolução com matrizes fatoradas por Cholesky é 0.0001297626569700117
O erro da biblioteca NumPy, através do np.linalg.solve, é 0.0006260495736752984
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 11
O erro da Eliminação Gaussiana sem pivotamento é 0.007885381408015743
O erro da resolução com matrizes fatoradas por Cholesky é 0.01179585904866715
O erro da biblioteca NumPy, através do np.linalg.solve, é 0.02419196620654922
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE HILBERT DE ORDEM 12
O erro da Eliminação Gaussiana sem pivotamento é 0.8064992831151371
O erro da resolução com matrizes fatoradas por Cholesky é 0.7403411720899308
O erro da biblioteca NumPy, através do np.linalg.solve, é 1.199575129508879
```

### Gráfico comparando os erros do processo descrito no log acima.
!["Error comparision - Hilbert, n < 13"](development/graphs/naive%20gauss%20vs%20cholesky%20vs%20linalg.solve%20-%20error%20.png?raw=true)  


* Para matrizes de Hilbert de dimensão n >= 13, o seu mal condicionamento faz com que a matriz deixe de ser positiva definida. Dessa forma, o fatoramento pelo método de Cholesky torna-se impossível.

```python
>> np.linalg.cholesky(linalg.hilbert(13))

Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "<__array_function__ internals>", line 6, in cholesky
  File "/home/gus-araujo/.local/lib/python3.7/site-packages/numpy/linalg/linalg.py", line 755, in cholesky
    r = gufunc(a, signature=signature, extobj=extobj)
  File "/home/gus-araujo/.local/lib/python3.7/site-packages/numpy/linalg/linalg.py", line 100, in _raise_linalgerror_nonposdef
    raise LinAlgError("Matrix is not positive definite")
numpy.linalg.LinAlgError: Matrix is not positive definite
```

 > \>\> np.linalg.cholesky(linalg.hilbert(13))  
 > **numpy.linalg.LinAlgError: Matrix is not positive definite**


#### Log exibindo o tempo gasto em cada iteração por Naive Gauss e Cholesky. Além disso, seus erros também são exibidos, utilizando o linalg.solve como referência para comparação.

```python
A precisão utilizada é 1e-16
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE ORDEM 1
O erro da Eliminação Gaussiana sem pivotamento é 0.0
O erro da resolução com matrizes fatoradas por Cholesky é 0.0
--- TEMPO GASTO EM CADA SOLUÇÃO ---
Tempo gasto pelo Naive Gauss nessa solução foi 0.000s
Tempo gasto pelo método de Cholesky nessa solução foi 0.000s
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE ORDEM 21
O erro da Eliminação Gaussiana sem pivotamento é 8.130367573911198e-14
O erro da resolução com matrizes fatoradas por Cholesky é 3.21964150263472e-12
--- TEMPO GASTO EM CADA SOLUÇÃO ---
Tempo gasto pelo Naive Gauss nessa solução foi 0.007s
Tempo gasto pelo método de Cholesky nessa solução foi 0.004s
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE ORDEM 41
O erro da Eliminação Gaussiana sem pivotamento é 5.319231748110063e-12
O erro da resolução com matrizes fatoradas por Cholesky é 8.561180755472533e-09
--- TEMPO GASTO EM CADA SOLUÇÃO ---
Tempo gasto pelo Naive Gauss nessa solução foi 0.049s
Tempo gasto pelo método de Cholesky nessa solução foi 0.027s
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE ORDEM 61
O erro da Eliminação Gaussiana sem pivotamento é 2.1534852138452376e-12
O erro da resolução com matrizes fatoradas por Cholesky é 5.412146048481012e-09
--- TEMPO GASTO EM CADA SOLUÇÃO ---
Tempo gasto pelo Naive Gauss nessa solução foi 0.122s
Tempo gasto pelo método de Cholesky nessa solução foi 0.051s
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE ORDEM 81
O erro da Eliminação Gaussiana sem pivotamento é 8.042962711608777e-13
O erro da resolução com matrizes fatoradas por Cholesky é 1.2722791867059034e-10
--- TEMPO GASTO EM CADA SOLUÇÃO ---
Tempo gasto pelo Naive Gauss nessa solução foi 0.247s
Tempo gasto pelo método de Cholesky nessa solução foi 0.118s
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE ORDEM 101
O erro da Eliminação Gaussiana sem pivotamento é 1.0850083192216472e-11
O erro da resolução com matrizes fatoradas por Cholesky é 1.4944549890304308e-08
--- TEMPO GASTO EM CADA SOLUÇÃO ---
Tempo gasto pelo Naive Gauss nessa solução foi 0.450s
Tempo gasto pelo método de Cholesky nessa solução foi 0.212s
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE ORDEM 121
O erro da Eliminação Gaussiana sem pivotamento é 1.4929345912857306e-12
O erro da resolução com matrizes fatoradas por Cholesky é 1.1190002615964225e-09
--- TEMPO GASTO EM CADA SOLUÇÃO ---
Tempo gasto pelo Naive Gauss nessa solução foi 0.809s
Tempo gasto pelo método de Cholesky nessa solução foi 0.369s
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE ORDEM 141
O erro da Eliminação Gaussiana sem pivotamento é 1.6100897513658367e-12
O erro da resolução com matrizes fatoradas por Cholesky é 6.6843383783108016e-09
--- TEMPO GASTO EM CADA SOLUÇÃO ---
Tempo gasto pelo Naive Gauss nessa solução foi 1.193s
Tempo gasto pelo método de Cholesky nessa solução foi 0.618s
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE ORDEM 161
O erro da Eliminação Gaussiana sem pivotamento é 7.862319672234349e-12
O erro da resolução com matrizes fatoradas por Cholesky é 4.538003683711476e-08
--- TEMPO GASTO EM CADA SOLUÇÃO ---
Tempo gasto pelo Naive Gauss nessa solução foi 1.723s
Tempo gasto pelo método de Cholesky nessa solução foi 0.821s
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE ORDEM 181
O erro da Eliminação Gaussiana sem pivotamento é 5.3231677760402715e-12
O erro da resolução com matrizes fatoradas por Cholesky é 9.706022397945798e-09
--- TEMPO GASTO EM CADA SOLUÇÃO ---
Tempo gasto pelo Naive Gauss nessa solução foi 2.485s
Tempo gasto pelo método de Cholesky nessa solução foi 1.196s
--------------------------------------------------------------------------
INICIANDO ITERAÇÃO PARA UMA MATRIZ DE ORDEM 201
O erro da Eliminação Gaussiana sem pivotamento é 1.8526202989707152e-10
O erro da resolução com matrizes fatoradas por Cholesky é 1.3089628204472875e-05
--- TEMPO GASTO EM CADA SOLUÇÃO ---
Tempo gasto pelo Naive Gauss nessa solução foi 3.301s
Tempo gasto pelo método de Cholesky nessa solução foi 1.580s
```

### Gráfico comparando o tempo gasto pelo Naive Gauss e por Cholesky para transformarem matrizes SPD (symmetric, positive and definite) de dimensão até 200.

!["Time elapsed - Naive Gauss vs Choleksy"](development/graphs/cholesky-vs-naive-gaus-time-elapsed.png?raw=true) 

* O gráfico e os logs explicitam a velocidade de execução discutida no início do capítulo: Cholesky tem uma vantagem de fator 2 sobre o Naive Gauss, o que torna-o vantajoso a médio prazo. Entretanto, numa análise assintótica, ambos comportam-se da mesma maneira.

* Geração de matrizes SPD com entradas aleatórias: para esse processo foi utilizada a biblioteca sklearn e o seu pacote datasets. O comando final foi: matriz_spd_aleatoria = sklearn.datasets.make_spd_matrix(n), onde n é a dimensão da matriz. Essa decisão foi tomada devido a problemas com raízes quadradas de números negativos em Cholesky. Entretanto, ficou claro como produzir matrizes SPD: basta haver uma matriz não singular e multiplicá-la pela sua transposta.
