Creating an Array:
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(arr)

Output:
[1 2 3 4 5]

----------------------------------
Accessing Elements:
print(arr[0])  # First element
print(arr[-1])  # Last element

Output:
1
5
----------------------------------
Unique Elements in an Array:
arr = np.array([1, 2, 2, 3, 4, 4, 5])
unique = np.unique(arr)
print(unique)

Output:
[1 2 3 4 5]
----------------------------------
Flatten a Multi-dimensional Array:
arr = np.array([[1, 2], [3, 4]])
flattened = arr.flatten()
print(flattened)

Output:
[1 2 3 4]
----------------------------------
Transpose of a Matrix:
matrix = np.array([[1, 2], [3, 4], [5, 6]])
transposed = matrix.T
print(transposed)

Output:
[[1 3 5]
 [2 4 6]]
----------------------------------
Array Broadcasting:
arr = np.array([1, 2, 3])
result = arr + 5
print(result)

Output:
[6 7 8]


