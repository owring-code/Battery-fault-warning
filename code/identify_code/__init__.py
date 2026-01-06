import numpy as np


def find_min_rows_in_columns(matrix):
    """
    Given a 2D array (matrix), this function returns a list of row indices representing
    the row index of the minimum value in each column.

    :param matrix: 2D list or numpy array
    :return: List of row indices
    """
    if not matrix or not matrix[0]:
        return []

    # Using numpy's argmin function to find the index of the minimum value in each column
    return np.argmin(matrix, axis=0).tolist()


# Example matrix
matrix_example = [
    [3, 2, 1],
    [1, 3, 2],
    [2, 1, 3]
]

# Finding the row indices of the minimum values in each column
print(find_min_rows_in_columns(matrix_example))
