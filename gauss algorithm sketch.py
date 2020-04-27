import numpy as np
from scipy import linalg

def find_smallest_integer_p():                  # Why is it a requisite to find the smallest integers? 
    print("achei")
    return 23

def gauss(A):                                   # A is the augmented matrix
    p = 0
    
    for i in range(0, n-2):                     # Changed because of zero index
        p = find_smallest_integer_p()
        if (p == "non-integer number or zero"): # is it possible to p be zero?
            print("no unique solution exists")
            return "something appropriate to this case"
        if (p != i):
            print("numpy swap rows: A[i]-swap-A[p]")
        for j in range(i, n-1):                 # Changed because of zero index
            m[j][i] = a[j][i]/a[i][i]
            row[j] = row[j] - m*row[i]
    
    for k in range (0, n-1):
        if A[k][k] == 0:                        # A zero in the main diagonal     
            print("no unique solution exists")
            return "something appropriate to this case"
    
    x[n] = np.array
    for k in range (0, n-1):                    # Backward substitution
        x[k] = a[n-1][n]/a[n-1][n-1]
    
    for i in range(n-2, 0):
        sum_axj = 0
        for j in range(i, n-1):
            sum_axj = (sum_axj + a[i][j]*x[j]) 
        x[i] = (a[i][n] - sum_axj) / a[i][i]
    print("results are ready")
    return "x array with the results"
        

# Develop some main function

approximation = gauss(linalg.hilbert(3))