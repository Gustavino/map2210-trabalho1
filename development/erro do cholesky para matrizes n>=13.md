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