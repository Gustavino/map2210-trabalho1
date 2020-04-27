import numpy as np

def find_smallest_integer_p():
    print("achei")

def gauss(n):
    p = 0
    for i in range (0, n-2):
        p = find_smallest_integer_p()
        if (p == "non-integer number or zero"): # is it possible to p be zero?
            print("no unique solution exists")
            return "something appropriate to this case"
        if (p != i):
            print("numpy swap rows: A[i]-swap-A[p]")
            