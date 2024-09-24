# Computational Robotics
# Component 1
# Dylan Turner & Noor Hasan

import numpy as np
import numpy.linalg 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math

############################################################
#                  Validating Rotations                    #
############################################################


#TODO: Choose how to implement epsilon and must report it.

#Input: matrix ∈ ×. Return: True if m∈SO(n) (within epsilon numerical preci-sion), false otherwise. (Consider n=2 or n=3 only)
def check_SOn(matrix, epsilon = 0.01):
    #Must be a square matrix.
    if(matrix.shape[0] != matrix.shape[1]):
        return False
    
    #Must be orthagonal: R^T * R = I
    identity = np.eye(matrix.shape[0])  
    orthogonal_check = np.allclose(np.dot(matrix.T, matrix), identity, atol=epsilon)

    #Determinant must equal 1
    determinant_check = np.isclose(np.linalg.det(matrix), 1, atol=epsilon)

    return determinant_check and orthogonal_check

#Examples
# R = np.array([[1, 0], [0, 1]])  
# print(check_SOn(R, epsilon=1e-6))
# R2 = np.array([[1, 0], [0, 0.5]])
# print(check_SOn(R2, epsilon=1e-6))

# Input: vector ∈ ×. Return: True if vector ∈ S3 (within epsilon numericalprecision), false otherwise
def check_quaternion(vector, epsilon = 0.01):
    #Quaternion vector must be length 4
    if len(vector) != 4:
        return False
    
    magnitude = np.sqrt(np.sum(np.square(vector)))
    return np.isclose(magnitude, 1, atol=epsilon)

# q = [1, 0, 0, 0]  
# print(check_quaternion(q))
# q2 = [1, 1, 0, 1]  
# print(check_quaternion(q2))


# Input: matrix ∈×. Return: True if ∈SE(n) (within epsilon numerical precision),false otherwise. (Consider n=2 or n=3 only)
def check_SEn(matrix, epsilon = 0.01):
    #Make sure the rotation part of the matrix is valid
    n = matrix.shape[0] - 1 #dim of rot matrix

    if matrix.shape != (n + 1, n + 1): #correct shape
        return False
    
    rotation_matrix = matrix[:n, :n] #extract

    if not check_SOn(rotation_matrix, epsilon):
        return False
    
    #Make sure the bottom row is [0, 0, ..., 0, 1]
    bottom_row = matrix[n, :]
    expected_bottom_row = np.zeros(n + 1)
    expected_bottom_row[-1] = 1 
    
    if not np.allclose(bottom_row, expected_bottom_row, atol=epsilon):
        return False

    #All good at this point    
    return True

# T = np.array([[1, 0, 0, 3], [0, 1, 0, 4], [0, 0, 1, 5], [0, 0, 0, 1]])  # SE(3) matrix
# print(check_SEn(T))  # Output: True


#EXTRA CREDIT

#Each function corrects the given input if the element is not part of the group within an epsilon
#distance. The corrected matrix has to be close to the input matrix (always returning the identity is
#an invalid implementation). The correction implies that you can compute a distance between the
#given matrix (which may not be part of the group) and a member of the group
def correct_SOn(matrix, epsilon=0.01):
    return matrix

def correct_quaternion(vector, epsilon=0.01):
    return vector

def correct_SEn(matrix, epsilon=0.01):
    return matrix

#TODO: Test your function with multiple random rotations and report your implementation and the results in your report.