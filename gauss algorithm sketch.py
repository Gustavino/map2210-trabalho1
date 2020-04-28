import numpy as np
from scipy import linalg


def find_first_non_null_pivot(n):

    
    return 23


def gauss(A, b):                                               # A is the augmented matrix
    n = len(A)
    pivot_row_index = 0

    for i in range(0, n-2):                                    # Changed because of zero index
        pivot_row_index = find_first_non_null_pivot(n)
        if (pivot_row_index == "non-integer number or zero"):  # is it possible to p be zero?
            print("no unique solution exists")
            return np.empty
        if (pivot_row_index != i):
            print("numpy swap rows: A[i]-swap-A[p]")
            A[i], A[pivot_row_index] = A[pivot_row_index], A[i]
        for j in range(i+1, n-1):                
            m = A[j][i]/A[i][i]
            # np.subtract(A row j ...)
            A[j] = A[j] - m*A[i]                                # Eliminating the first j-th elements; 
                                                                # A[j] means all columns of the j-th row of A

    for k in range(0, n-1):                                     # Verifying if there is any zero in the main diagonal
        if A[k][k] == 0:                        
            print("no unique solution exists")
            return np.empty(shape=(n, n))


    # STARTING BACKWARD SUBSTITUTION

    x = np.ones(shape=n)
    for k in range(0, n-1):                     # Backward substitution
        x[k] = A[n-1][n]/A[n-1][n-1]

    for i in range(n-2, 0):
        sum_axj = 0
        for j in range(i, n-1):
            sum_axj = (sum_axj + A[i][j]*x[j])
        x[i] = (A[i][n] - sum_axj) / A[i][i]
    print("results are ready")
    return x


# Develop some main function


hilbert_solution = "that sum over the rows"
approximation = gauss(linalg.hilbert(3), hilbert_solution)
