import numpy as np

transformation_matrix = np.array([[0, -1],
                                  [1, 0]])

# The input vector
input_vector = np.array([3, 1])

# Rotate the input vector
rotated_vector = transformation_matrix * input_vector

# Print the rotated vector
print(rotated_vector)
