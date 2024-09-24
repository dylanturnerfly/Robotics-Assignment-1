# Computational Robotics
# Component 2
# Dylan Turner & Noor Hasan

import numpy as np
import numpy.linalg 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import math


############################################################
#                Uniform Random Rotations                  #
############################################################

#Input: A boolean that defines how the rotation is generated. If
#naive is true, implement a naive solution (for example, random euler angles and convert to
#rotation matrix). If naive is false, implement the function as defined in Figure 1 in [1]. Return:
#A randomly generated element R∈SO(3).

#Your function should always return valid rotations (calling your function check_SOn should always return True). You can check this 
#through visualizaton: Check out Algorithm 1 on the report

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
    

def visualize_rotation_sphere(R):
    print('hi')
    v0 = np.array([0, 0, 1])
    epsilon = 0.1
    v1 = np.array([0, epsilon, 0]) + v0

    v0_rotated = R @ v0
    v1_rotated = R @ v1 - v0_rotated

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #plot unit sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='b', alpha=0.3)
    
    ax.quiver(*v0_rotated, *v1_rotated, length=0.1, color='r', label='Rotation Effect', arrow_length_ratio=0.1)
    ax.scatter(*v0_rotated, color='g', s=100, label='Rotated North Pole (v0\')')
    ax.scatter(*v1_rotated + v0_rotated, color='b', s=100, label='Rotated Nearby Point (v1\')')

    # Labels and Legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('3D Rotation Visualization on Unit Sphere')
    ax.legend()
    plt.show()

# visualize_rotation_sphere(random_rotation_matrix(True))
# print(random_rotation_matrix(True))
# print()
# print(random_rotation_matrix(False))

#TODO: In your report include and analyze any design choice of your implementation. Include visual-
#izations of the random samples (the spheres)

#EXTRA CREDIT:

#Input: A boolean that defines how the rotation is generated. If naive
# is true, implement a naive solution (for example, random euler angles and convert to rotation
# matrix). If naive is false, implement the function as defined in Algorithm 2 in [2]. Return: A
# randomly generated element q∈S3
def random_quaternion(naive):
    vector = 1
    return vector
