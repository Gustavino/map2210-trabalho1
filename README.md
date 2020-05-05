# Universidade de São Paulo - Instituto de Matemática e Estatística (IME-USP)

## MAP 2210 - Aplicações de Álgebra Linear

### Trabalho Computacional I


#### Parte 1

Implemente dois algoritmos para a solução de sistemas lineares em python. 
* a) sem pivotamento.  
* b) com pivotamento parcial. 

Sua implementação deve calcular o determinante da matriz como diagnóstico de proximidade de singularidade da matriz, usando o produto da diagonal da matriz U.

> Para os dois algoritmos, utilize a matriz de Hilbert. Gere matrizes de ordens crescentes (por exemplo, n = 2, 4, 8, 16, ...). Crie o vetor do lado direito, b, através da soma das componentes das linhas de H, isso fará que a solução do sistema Hx = b são vetores x com todas as componentes unitárias.  
* i) Resolva cada sistema usando os algoritmos desenvolvidos, e calcule a norma 2 da diferença entre a solução obtida e a solução exata (todas as componentes unitárias).
* ii) Compare essa norma do erro com o determinante da respectiva matriz de Hilbert.
* iii) O que se pode concluir desses resultados?

#### Parte 2 
De modo similar à Parte 1, crie aletoriamente matrizes de dimensões crescentes (por exemplo n = 2, 4, 8, 16, ...). Multiplique cada matriz por sua transposta produzindo matrizes simétricas positivas definidas (se a matriz original não é singular). Da mesma forma anterior use vetores com todas as componentes unitárias para gerar artificialmente os vetores do lado direito b's.
* i) Faça um código que implemente a decomposição de Cholesky e as respectivas substituições diretas e reversas para resolver esses sistemas calcule a norma do erro na norma 2 (o comando linalg.cholesky, ou semelhante, é vedado).  
* ii) Para cada sistema resolva utilizando seu código de Cholesky, o seu código de Eliminação de Gauss sem pivotamento parcial, e para comparação, outro código usando o comando linalg.solve.
* iii) Compare a precisão da solução em cada caso, o tempo computacional e o
determinante da matriz.
* iv) O que se pode concluir desses resultados?