# Computational Robotics
# Component 1
# Dylan Turner & Noor Hasan

import numpy as np

############################################################
#                  Validating Rotations                    #
############################################################

'''
Input: matrix ∈ ×. Return: True if m∈SO(n) (within epsilon numerical preci-sion), false otherwise. (Consider n=2 or n=3 only)
'''
def check_SOn(matrix, epsilon = 0.01):
    #must be a square matrix.
    if(matrix.shape[0] != matrix.shape[1]):
        return False
    
    #must be orthagonal: R^T * R = I
    identity = np.eye(matrix.shape[0])  
    orthogonal_check = np.allclose(np.dot(matrix.T, matrix), identity, atol=epsilon)

    #determinant must equal 1
    determinant_check = np.isclose(np.linalg.det(matrix), 1, atol=epsilon)

    return determinant_check and orthogonal_check

'''
Input: vector ∈ ×. Return: True if vector ∈ S3 (within epsilon numericalprecision), false otherwise
'''
def check_quaternion(vector, epsilon = 0.01):
    #quaternion vector must be length 4
    if len(vector) != 4:
        return False
    
    magnitude = np.sqrt(np.sum(np.square(vector)))
    return np.isclose(magnitude, 1, atol=epsilon)

'''
Input: matrix ∈×. Return: True if ∈SE(n) (within epsilon numerical precision),false otherwise. (Consider n=2 or n=3 only)
'''
def check_SEn(matrix, epsilon = 0.01):
    #make sure the rotation part of the matrix is valid
    n = matrix.shape[0] - 1 #dim of rot matrix

    if matrix.shape != (n + 1, n + 1): #correct shape
        return False
    
    rotation_matrix = matrix[:n, :n] #extract

    if not check_SOn(rotation_matrix, epsilon):
        return False
    
    #make sure the bottom row is [0, 0, ..., 0, 1]
    bottom_row = matrix[n, :]
    expected_bottom_row = np.zeros(n + 1)
    expected_bottom_row[-1] = 1 
    
    if not np.allclose(bottom_row, expected_bottom_row, atol=epsilon):
        return False

    #all good at this point    
    return True

'''
Returns the closest matrix to the input that is a part of SO(n)
'''
def correct_SOn(matrix, epsilon=0.01):
    #must be a square matrix to be corrected.
    if matrix.shape[0] != matrix.shape[1]:
        print("Cannot correct a nonsquare matrix!")
        return

    #first make sure it doesnt already belong to SO(n)
    if(check_SOn(matrix, epsilon)):
        #print('already good')
        return matrix
    
    #perform single value decomposition to get closest orthagonal matrix: A' = UV^T
    U, S, V = np.linalg.svd(matrix)
    corrected_matrix = np.matmul(U, np.transpose(V))

    #make sure the determinant is 1. Flip sign of a column if not
    
    if(not np.isclose(np.linalg.det(matrix), 1, atol=epsilon)):
        #print('flipping det')
        U[0] = -U[0]
        corrected_matrix = np.matmul(U, np.transpose(V))

    return corrected_matrix

'''
Returns the closest valid quaternion to the input.
'''
def correct_quaternion(vector, epsilon=0.01):
    #make sure it has four componenets
    if(len(vector) != 4):
        print("Cannot correct quarernion that is not of length 4!")
        return
    
    #simply divide each componenet by its magnitude to make it a unit vector.
    magnitude = np.linalg.norm(vector)
    if(magnitude == 0):
        print("Cannot correct a vector of magnitude 0!")
        return
    
    correct_quaternion = vector / magnitude
    return correct_quaternion

'''
Returns the closest matrix to the input that is a part of SE(n).
'''
def correct_SEn(matrix, epsilon=0.01):
    #first make sure the rotation part is in SO(n)
    n = matrix.shape[0]
    correct_rotation = correct_SOn(matrix[:n-1,:n-1])
    # print(f"\nrotation{correct_rotation}\n")
    # print(f"\ntranslation{matrix[:n-1,n-1]}\n")

    #create a new matrix with the corrected rotation and previous translation
    correct_matrix = np.column_stack([correct_rotation, matrix[:n-1,n-1]]) #column stack rotation and translation
    
    #add (0001) row
    bottom_row = np.zeros(n)
    bottom_row[-1] = 1
    correct_matrix = np.row_stack ([correct_matrix, bottom_row])
    return correct_matrix