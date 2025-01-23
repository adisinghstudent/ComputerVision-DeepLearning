import numpy as np

def manipulate_matrix(A, B, c):
    """
    Perform matrix manipulations using NumPy.

    Args:
        A (np.ndarray): A 2D matrix of size (m, n).
        B (np.ndarray): A 2D matrix of size (n, p).
        c (float): A scalar constant.

    Returns:
        tuple: A tuple containing:
            - Shape of A (tuple).
            - Shape of B (tuple).
            - Reshaped A (np.ndarray).
            - Dot product result (np.ndarray).
            - Exponentiation result (np.ndarray).
            - Argmax result (np.ndarray).
            - Sum over axis 0 (np.ndarray).
            - Matrix multiplication result (np.ndarray).
            - Element-wise multiplication result (np.ndarray).
            - Matrix addition with constant result (np.ndarray).
    """
    assert A.ndim == 2, "A must be a 2D matrix"
    assert B.ndim == 2, "B must be a 2D matrix"
    assert A.shape[1] == B.shape[0], "Number of columns in A must match number of rows in B"
    
    # Implement the required operations
    shape_A = A.shape
    shape_B = B.shape
    A_reshaped = A.reshape(-1)  # Reshape A into a 1D vector
    dot_product_result = np.dot(A, B)  # Dot product of A and B
    A_exp = np.exp(A)  # Element-wise exponentiation of A
    A_argmax = np.argmax(A, axis=1)  # Argmax of each row in A
    A_sum_axis_0 = np.sum(A, axis=0, keepdims=True)  # Sum over axis 0, keepdims
    
    # Matrix multiplication of A and B
    matrix_mult_result = np.dot(A, B)  # Matrix multiplication of A and B
    
    # Element-wise multiplication of A and B (must have the same size)
    elementwise_mult_result = np.multiply(A, B.T)  # Element-wise multiplication of A and B
    
    # Matrix addition with constant
    matrix_plus_constant_result = A + c  # Add constant to each element of A

    return shape_A, shape_B, A_reshaped, dot_product_result, A_exp, A_argmax, A_sum_axis_0, matrix_mult_result, elementwise_mult_result, matrix_plus_constant_result

# Example usage:
A = np.array([[1, 2, 3], [4, 5, 6]])
B = np.array([[1, 4], [2, 5], [3, 6]])
c = 10
result = manipulate_matrix(A, B, c)
# print("Results:", result)


print(result.shape)