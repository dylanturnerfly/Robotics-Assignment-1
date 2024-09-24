# Computational Robotics
# Assignment 1
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


############################################################
#                  Rigid Body in Motion                    #
############################################################

#Define a planar environment of dimensions 20x20, with bounds [-10,10] for x and y. A rectangular
# robot of dimensions 0.5x0.3. This robot is controlled via a velocity vector V = (,y,θ). There
# are no obstacles in this environment.

#Input: start pose x0 ∈SE(2) and goal position xG in nSE(2). Output
# a path (sequence of poses) that start at x0 and ends in xG.
def interpolate_rigid_body(start_pose, goal_pose):
    return 1

#Input: start pose x0 ∈SE(2) and a Plan ( a sequence of N tuples (velocity, duration)) that is 
# applied to the start pose and results in a path of N+ 1 states.
def forward_propagate_rigid_body(start_pose, plan):
    return 1

#Input: A path to visualize. Your visualization must include the path and an animation of the robot’s movement.
def visualize_path(path):
    return 1

#TODO: In your report include and analyze any design choice of your implementation. Include visual-
# izations of the paths generated by both methods. Submit an example animation (gif is prefered
# but can also be a small mp4).


############################################################
#                   Movement of an Arm                     #
############################################################


#Using the same environment as before, implement the 2-joint, 2-link arm in figure 1 arm using
#boxes as the geometries of individual links. Here are more details regarding your robotic arm:

#   • The first link has length 2 and the second has lenght 1.5.
#   • All frames associated with links {Li} are at the center of the boxes. The frames associated ith joints {Ji}are located at the box’s bottom.

#To implement your arm, define the coordinate frame of each link and the relative poses with
#each other. Start with only the first link and its transformations before moving to the second link

# Input: start configuration q0 = (θ0,θ1) and goal configuration qG. Output
# a path (sequence of poses) that start at q0 and ends in qG.
def interpolate_arm(start, goal):
    return 1

#Input: start pose q0 ∈SE(2) and a Plan ( a sequence of N tuples
# (velocity, duration)) that is applied to the start pose and results in a path of N+ 1 states.
def forward_propagate_arm(start_pose, plan):
    return 1

#Input: A path to visualize. Your visualization must include the path
# and an animation of the robot’s movement.
def visualize_arm_path(path):
    return 1

#TODO: In your report include and analyze any design choice of your implementation. Include visual-
# izations of the paths generated by both methods. Submit an example animation (gif is prefered
# but can also be a small mp4).