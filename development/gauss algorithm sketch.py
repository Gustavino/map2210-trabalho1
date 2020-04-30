import numpy as np
from scipy import linalg


def find_first_non_null_pivot(A, start):
    """  Iterate over the first column searching for a bigger than zero pivot  """
    pivot_row_index = start

    for i in range(start, n):
        if A[i][start] != 0:
            pivot_row_index = i
            break

    return pivot_row_index


def gauss(A):                                                   # A is the augmented matrix
    n = len(A)
    for i in range(0, n):                                       # STEP 1. i = i-th.
        pivot_row_index = find_first_non_null_pivot(A, i)

        if abs(A[pivot_row_index][0]) < 0.00001:               # STEP 2. A[0][0] is equal to 0.
            return np.empty(shape = (n, n))

        if (pivot_row_index != i):                              # STEP 3. Swaping rows.
            A[[i, pivot_row_index]] = A[[pivot_row_index, i]]

        for j in range(i+1, n):                                 # STEP 4.      
            m = A[j][i]/A[i][i]                                 # STEP 5.
            A[j] = A[j] - m*A[i]                                # STEP 6. Eliminating the first j-th elements.
                                                                # A[j] means all columns of the j-th row of A.

    print(A)
    for k in range(0, n):                                       # STEP 7. Verifying if there is any zero in the main diagonal.
        if abs(A[k][k]) < 0.00001:                              # Checking if A[k][k] is zero.
            print("no unique solution exists")
            return 0

    # STARTING BACKWARD SUBSTITUTION
    x = np.zeros(shape = n)
    x[n-1] = A[n-1][n]/A[n-1][n-1]                              # STEP 8
                                             
    for i in range(n-2, 0):                                     # STEP 9. Backward substitution.
        sum_axj = 0
        for j in range(i, n-1):
            sum_axj = (sum_axj + A[i][j]*x[j])
        x[i] = (A[i][n] - sum_axj) / A[i][i]

    return x





# Develop some main function


n = input()
hilbert_matrix = linalg.hilbert(int(n)) # Generating a nxn Hilbert matrix
vector_b = np.array([hilbert_matrix[i].sum() for i in range (0, len(hilbert_matrix))]) # Generating vector b, the sum of Hilbert matrices rows
vector_b = np.vstack(vector_b)

augmented_hilbert_matrix = np.append(hilbert_matrix, vector_b, axis=1)

vector_x = gauss(augmented_hilbert_matrix)
