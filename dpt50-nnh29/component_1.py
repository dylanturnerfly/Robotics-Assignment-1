# Computational Robotics
# Component 1
# Dylan Turner & Noor Hasan

import numpy as np
import unittest

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

############################################################
#                       Extra Credit                       #
############################################################

'''
Returns the closest matrix to the input that is a part of SO(n)
'''
def correct_SOn(matrix, epsilon=0.01):
    #must be a square matrix to be corrected.
    if matrix.shape[0] != matrix.shape[1]:
        #Cannot correct a nonsquare matrix
        return

    #first make sure it doesnt already belong to SO(n)
    if(check_SOn(matrix, epsilon)):
        return matrix
    
    #perform single value decomposition to get closest orthagonal matrix: A' = UV^T
    U, S, V = np.linalg.svd(matrix)
    corrected_matrix = np.matmul(U, np.transpose(V))

    #make sure the determinant is 1. Flip sign of a column if not
    
    if(not np.isclose(np.linalg.det(corrected_matrix), 1, atol=epsilon)):
        U[0] = -U[0]
        corrected_matrix = np.matmul(U, np.transpose(V))

    return corrected_matrix

'''
Returns the closest valid quaternion to the input.
'''
def correct_quaternion(vector, epsilon=0.01):
    #make sure it has four componenets
    if(len(vector) != 4):
        #Cannot correct quarernion that is not of length 4
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

############################################################
#                      Unit Tests                          #
############################################################

class TestRotationFunctions(unittest.TestCase):

#check_SOn()
    def test_check_SOn_valid_2D_1(self):
        matrix = np.array([[0, -1], [1, 0]])  #90 degree rotation
        self.assertTrue(check_SOn(matrix))

    def test_check_SOn_valid_2D_2(self):
        matrix = np.array([[1, 0], [0, 1]])  #identity matrix
        self.assertTrue(check_SOn(matrix))

    def test_check_SOn_valid_2D_3(self):
        matrix = np.array([[0, 1], [-1, 0]])  #270 degree rotation
        self.assertTrue(check_SOn(matrix))

    def test_check_SOn_valid_2D_4(self):
        matrix = np.array([[np.sqrt(2)/2, -np.sqrt(2)/2], [np.sqrt(2)/2, np.sqrt(2)/2]])  #45 degree rot
        self.assertTrue(check_SOn(matrix))

    def test_check_SOn_valid_3D_1(self):
        matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  #identity matrix
        self.assertTrue(check_SOn(matrix))

    def test_check_SOn_valid_3D_2(self):
        matrix = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  #90 deg about x
        self.assertTrue(check_SOn(matrix))

    def test_check_SOn_valid_3D_3(self):
        matrix = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])  #90 deg about y
        self.assertTrue(check_SOn(matrix))

    def test_check_SOn_valid_3D_4(self):
        matrix = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])  #90 deg about z
        self.assertTrue(check_SOn(matrix))

    def test_check_SOn_invalid_1(self):
        matrix = np.array([[1, 2], [3, 4]])  #not orthogonal
        self.assertFalse(check_SOn(matrix))

    def test_check_SOn_invalid_2(self):
        matrix = np.array([[1], [3], [5]])  #not square matrix
        self.assertFalse(check_SOn(matrix))

    def test_check_SOn_invalid_3(self):
        matrix = np.array([[1,6], [3,2], [5,2]])  #not square matrix
        self.assertFalse(check_SOn(matrix))

#check_quaternion()

    def test_check_quaternion_valid_1(self):
        vector = np.array([1, 0, 0, 0])  #valid unit quaternion
        self.assertTrue(check_quaternion(vector))

    def test_check_quaternion_valid_1(self):
        vector = np.array([0, 0, 0, 1])  #valid unit quaternion
        self.assertTrue(check_quaternion(vector))

    def test_check_quaternion_invalid_1(self):
        vector = np.array([1, 1, 1, 1])  #not a unit quaternion
        self.assertFalse(check_quaternion(vector))

    def test_check_quaternion_invalid_1(self):
        vector = np.array([1, 1, 1, 1, 6])  #not four componenets
        self.assertFalse(check_quaternion(vector))

#check_SEn()
    def test_check_SEn_valid(self):
        matrix = np.array([[0, -1, 0, 1],
                           [1, 0, 0, 2],
                           [0, 0, 1, 3],
                           [0, 0, 0, 1]])  #valid SE(3)
        self.assertTrue(check_SEn(matrix))

    def test_check_SEn_invalid_1(self):
        matrix = np.array([[1, 0, 0, 1],
                           [0, 1, 0, 2],
                           [0, 0, 0, 3],
                           [0, 0, 0, 1]])  #invalid rotation
        self.assertFalse(check_SEn(matrix))

    def test_check_SEn_invalid_2(self):
        matrix = np.array([[1, 0, 0, 1],
                           [0, 1, 0, 2],
                           [0, 0, 0, 3],
                           [0, 0, 1, 1]])  #invalid bottom row
        self.assertFalse(check_SEn(matrix))

#correct_SOn
    def test_correct_SOn_1(self):
        matrix = np.array([[2, 0], [0, 3]])  #not in SO(2)
        corrected = correct_SOn(matrix)
        self.assertTrue(check_SOn(corrected))

    def test_correct_SOn_2(self):
        matrix = np.array([[2.5, -1.2], [1.4, 3.54]])  #not in SO(2)
        corrected = correct_SOn(matrix)
        self.assertTrue(check_SOn(corrected))

    def test_correct_SOn_3(self):
        matrix = np.array([[2, 0, 5], [1, 0, 3], [1, 2, 3]])  #not in SO(3)
        corrected = correct_SOn(matrix)
        self.assertTrue(check_SOn(corrected))

    def test_correct_SOn_4(self):
        matrix = np.array([[2.5, -1.2, 9.5], [1.4, -1.34, 3.54], [1.2, -2.33, 1.44]])  #not in SO(3)
        corrected = correct_SOn(matrix)
        self.assertTrue(check_SOn(corrected))

    def test_correct_SOn_5(self):
        matrix = np.array([[2, 0], [0, 3], [1,2]])  #not square
        corrected = correct_SOn(matrix)
        self.assertTrue(corrected == None)

    def test_correct_SOn_6(self):
        matrix = np.array([[2, 0, 1], [1, 0, 3], [2, 0, 1], [1, 0, 3]])  #not square
        corrected = correct_SOn(matrix)
        self.assertTrue(corrected == None)

#correct_quaternion
    def test_correct_quaternion_1(self):
        vector = np.array([1, 1, 1, 1])  #not a unit quaternion
        corrected = correct_quaternion(vector)
        self.assertTrue(check_quaternion(corrected))

    def test_correct_quaternion_2(self):
        vector = np.array([1.2, 1.41, 2.1, 5.4])  #not a unit quaternion
        corrected = correct_quaternion(vector)
        self.assertTrue(check_quaternion(corrected))
    
    def test_correct_quaternion_3(self):
        vector = np.array([100, 200, 300, 400])  #not a unit quaternion
        corrected = correct_quaternion(vector)
        self.assertTrue(check_quaternion(corrected))

    def test_correct_quaternion_4(self):
        vector = np.array([1, 0, 0, -1])  #not a unit quaternion
        corrected = correct_quaternion(vector)
        self.assertTrue(check_quaternion(corrected))

    def test_correct_quaternion_4(self):
        vector = np.array([1, 0, 0])  #not length 4
        corrected = correct_quaternion(vector)
        self.assertTrue(corrected == None)

#correct_SEn
    def test_correct_SEn(self):
        matrix = np.array([[2, 0, 0, 1],
                           [0, 2, 0, 2],
                           [0, 0, 0, 3],
                           [0, 0, 0, 1]])  #not in SE(3)
        corrected = correct_SEn(matrix)
        self.assertTrue(check_SEn(corrected))

    def test_correct_SEn(self):
        matrix = np.array([[2, 0, 0, 1],
                           [0, 2, 0, 2],
                           [0, 0, 0, 3],
                           [1, 1, 0, 1]])  #bottom row is wrong
        corrected = correct_SEn(matrix)
        self.assertTrue(check_SEn(corrected))

    def test_correct_SEn(self):
        matrix = np.array([[2, 0, 0],
                           [0, 0, 0],
                           [0, 0, 1]])  #not in SE(2)
        corrected = correct_SEn(matrix)
        self.assertTrue(check_SEn(corrected))

    def test_correct_SEn(self):
        matrix = np.array([[2, 0, 0],
                           [0, 0, 0],
                           [1, 0, 1]])  #bottom row wrong
        corrected = correct_SEn(matrix)
        self.assertTrue(check_SEn(corrected))

if __name__ == '__main__':
    unittest.main()