# Computational Robotics
# Component 2
# Dylan Turner & Noor Hasan

import numpy as np
import matplotlib.pyplot as plt
import unittest
from component_1 import check_quaternion
from component_1 import check_SOn

############################################################
#                Uniform Random Rotations                  #
############################################################

'''
Input: A boolean that defines how the rotation is generated. If
naive is true, implement a naive solution (for example, random euler angles and convert to
rotation matrix). If naive is false, implement the function as defined in Figure 1 in [1]. Return:
A randomly generated element R ∈ SO(3).
'''
def random_rotation_matrix(naive):
    if(naive):
        #Naive solution
        x_rot = np.random.uniform(0, 2 * np.pi)
        y_rot = np.random.uniform(0, 2 * np.pi)  
        z_rot= np.random.uniform(0, 2 * np.pi) 

        #X-axis rotation
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(x_rot), -np.sin(x_rot)],
            [0, np.sin(x_rot), np.cos(x_rot)]
        ])
        #Y-axis rotation
        R_y = np.array([
            [np.cos(y_rot), 0, np.sin(y_rot)],
            [0, 1, 0],
            [-np.sin(y_rot), 0, np.cos(y_rot)]
        ])
        #Z-axis rotation
        R_z = np.array([
                [np.cos(z_rot), -np.sin(z_rot), 0],
                [np.sin(z_rot), np.cos(z_rot), 0],
                [0, 0, 1]
        ])

        return np.dot(R_z, np.dot(R_y, R_x)) #combine them
    else:
        #Algorithm 1 Solution
        pole_rotation = np.random.uniform(0, 2 * np.pi)
        pole_deflection_direction = np.random.uniform(0, 2 * np.pi)  
        pole_deflection_amount = np.random.uniform(0, 1) #not sure what the range of this variable should be
    
        reflection_vector = [np.cos(pole_deflection_direction) * np.sqrt(pole_deflection_amount),
                             np.sin(pole_deflection_direction) * np.sqrt(pole_deflection_amount),
                             np.sqrt(1-pole_deflection_amount)]
        
        VV_T = np.outer(reflection_vector, reflection_vector)
        VV_T = 2 * VV_T - np.eye(3)

        rand_orientation = np.array([[np.cos(pole_rotation), np.sin(pole_rotation), 0],
                                    [-np.sin(pole_rotation), np.cos(pole_rotation), 0],
                                    [0, 0, 1]])
        M = np.dot(VV_T, rand_orientation)

        return M
    
'''
Visualizes uniformly random rotations on unit sphere.
'''
def visualize_rotation_sphere(amount, naive):
    v0 = np.array([0, 0, 1]) #north vector
    epsilon = 0.1 
    v1 = np.array([0, epsilon, 0]) + v0  #slightly offset north vector

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #plot unit sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='b', alpha=0.3)

    for _ in range(amount): #plot vectors
        R = random_rotation_matrix(naive)  #new random rotation
        
        #apply rotation
        v0_rotated = R @ v0
        v1_rotated = R @ v1 - v0_rotated

        #plot
        ax.quiver(*v0_rotated, *v1_rotated, length=0.1, color='r', arrow_length_ratio=0.1)
        ax.scatter(*v0_rotated, color='g', s=100)  # Rotated North Pole
        ax.scatter(*v1_rotated + v0_rotated, color='b', s=100)  # Rotated nearby point

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Visualization of 50 Random Rotations on Unit Sphere')
    plt.show()

############################################################
#                       Extra Credit                       #
############################################################

'''
#Input: A boolean that defines how the rotation is generated. If naive
# is true, implement a naive solution (for example, random euler angles and convert to rotation
# matrix). If naive is false, implement the function as defined in Algorithm 2 in [2]. Return: A
# randomly generated element q∈S3
'''
def random_quaternion(naive):
    if(naive):
        #my naive solution is to simply generate 4 random numbers and divide by their magnitude
        random_numbers = np.random.rand(4)
        magnitude = np.linalg.norm(random_numbers)
        if magnitude == 0:
            quaternion = np.array([1, 0, 0, 0])  
        else:
            quaternion = random_numbers / magnitude
        return quaternion
    
    else:
        #Algoriithm 2
        s = np.random.rand()
        sigma1 = np.sqrt(1-s)
        sigma2 = np.sqrt(s)
        theta1 = 2 * np.pi * np.random.rand()
        theta2 = 2 * np.pi * np.random.rand()

        w = np.cos(theta2) * sigma2
        x = np.sin(theta1) * sigma1
        y = np.cos(theta1) * sigma1
        z = np.sin(theta2) * sigma2

    return np.array([w,x,y,z])

############################################################
#                      Unit Tests                          #
############################################################

class TestRandomFunctions(unittest.TestCase):

#random_rotation_matrix(True)
    def test_random_rotation_1(self):
        for _ in range(50):
            rotation = random_rotation_matrix(True)
            self.assertTrue(check_SOn(rotation))

#random_rotation_matrix(False)
    def test_random_rotation_2(self):
        for _ in range(50):
            rotation = random_rotation_matrix(False)
            self.assertTrue(check_SOn(rotation))

#random_quaternion(True)
    def test_random_quaternion_1(self):
        for _ in range(50):
            quaternion = random_quaternion(True)
            self.assertTrue(check_quaternion(quaternion))

#random_quaternion(False)
    def test_random_quaternion_2(self):
        for _ in range(50):
            quaternion = random_quaternion(False)
            self.assertTrue(check_quaternion(quaternion))

#Visualize sphere
visualize_rotation_sphere(100, False)

if __name__ == '__main__':
    unittest.main()