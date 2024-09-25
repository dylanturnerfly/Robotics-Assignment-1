import component_1, component_2, component_3, component_4
import numpy as np

TestComponent1 = True
TestComponent2 = True
TestComponent3 = True
TestComponent4 = True


############################################################
#                  Validating Rotations                    #
############################################################
if(TestComponent1):
    #check_SOn()
    R = np.array([[1, 0], [0, 1]])  
    print(f"\ncheck_SOn() test - Valid SO(2) matrix\nR:\n {R}\n Returns: {component_1.check_SOn(R, epsilon=1e-6)}")

    R = np.array([[1, 0], [0, 0.5]])
    print(f"\ncheck_SOn() test - Invalid SO(2) matrix\nR:\n {R}\n Returns: {component_1.check_SOn(R, epsilon=1e-6)}")

    R = np.array([[1], [0], [1]])
    print(f"\ncheck_SOn() test - Invalid SO(2) matrix: Not square matrix\nR:\n {R}\n Returns: {component_1.check_SOn(R, epsilon=1e-6)}")

    R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])  
    print(f"\ncheck_SOn() test - Valid SO(3) matrix\nR:\n {R}\n Returns: {component_1.check_SOn(R, epsilon=1e-6)}")

    R = np.array([[1, 0, 0], [0, 0.5, 0], [0, 0, 1]])  
    print(f"\ncheck_SOn() test - Invalid SO(3) matrix\nR:\n {R}\n Returns: {component_1.check_SOn(R, epsilon=1e-6)}")

    R = np.array([[1,0], [0, 1], [1,0]])
    print(f"\ncheck_SOn() test - Invalid SO(2) matrix: Not square matrix\nR:\n {R}\n Returns: {component_1.check_SOn(R, epsilon=1e-6)}")

    #check_quaternion()
    q = [1, 0, 0, 0]  #Valid quaternion
    print(f"\ncheck_quaternion() test - Valid quaternion\nq:\n {q}\n Returns: {component_1.check_quaternion(q)}")

    q = [1, 1, 0, 1]  #Invalid quaternion
    print(f"\ncheck_quaternion() test - Invalid quaternion: not unit length\nq:\n {q}\n Returns: {component_1.check_quaternion(q)}")

    q = [1, 1, 0, 1, 1, 1, 1]  #Invalid quaternion
    print(f"\ncheck_quaternion() test - Invalid quaternion: length != 4\nq:\n {q}\n Returns: {component_1.check_quaternion(q)}")

    #check_SEn()
    T = np.array([[1, 0, 0, 3], [0, 1, 0, 4], [0, 0, 1, 5], [0, 0, 0, 1]])   #Valid SE(3) matrix
    print(f"\ncheck_SEn() test - Valid SE(3) matrix\nT:\n {T}\n Returns: {component_1.check_SEn(T)}")

    T = np.array([[1, 0, 0, 3], [0, 1, 0, 4], [0, 0, 1, 5], [0, 0, 0, 0]])   #Invalid SE(3) matrix: bottom row is incorrect
    print(f"\ncheck_SEn() test - Invalid SE(3) matrix: bottom row is incorrect\nT:\n {T}\n Returns: {component_1.check_SEn(T)}")

    T = np.array([[1, 0, 0, 3], [0, 1, 0, 4], [0, 0, 0.5, 5], [0, 0, 0, 1]]) #Invalid SE(3) matrix: rotation matrix not part of SO(n)
    print(f"\ncheck_SEn() test - Invalid SE(3) matrix: rotation matrix not part of SO(n)\nT:\n {T}\n Returns: {component_1.check_SEn(T)}")

############################################################
#                Uniform Random Rotations                  #
############################################################

if(TestComponent2):
    component_2.visualize_rotation_sphere(100, False)

############################################################
#                  Rigid Body in Motion                    #
############################################################

if(TestComponent3):
    #Propogate
    start = [[1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]]
    path = [[2, 2, np.pi/4, 1],
            [2, 2, np.pi/4, 1],
            [2, 2, np.pi/4, 1],
            [2, 2, np.pi/4, 1],
            [2, 2, np.pi/4, 1],
            [2, 2, np.pi/4, 1],
            [2, 2, np.pi/4, 1],
            [2, 2, np.pi/4, 1]]
    component_3.visualize_path(component_3.forward_propagate_rigid_body(start, path), 'component_3_propogate.gif')

    #Interpolate
    start_pose = [0,-8,0]
    goal_pose = [0,8,np.pi]
    component_3.visualize_path(component_3.interpolate_rigid_body(start_pose, goal_pose, 10), 'component_3_interpolate.gif')


############################################################
#                   Movement of an Arm                     #
############################################################

if(TestComponent4):
    #Interpolate
    start_angles = (np.deg2rad(0), np.deg2rad(0))
    goal_angles = (np.deg2rad(360), np.deg2rad(-180))
    path = component_4.interpolate_arm(start_angles, goal_angles, 25)
    component_4.visualize_arm_path(path, 'component_4_interpolate.gif', 50)

    #Propogate
    start_angles = (np.deg2rad(0), np.deg2rad(90))

    plan = [[np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],
            [np.pi/4, -np.pi/2, 0.1],]

    path = component_4.forward_propagate_arm(start_angles, plan)
    component_4.visualize_arm_path(path, 'component_4_propogate.gif', 20)







#
# R = np.array([[0.123, 0.521, 12], [1, 3.4, 1.9], [4, 2.32, 1.12]])  
# print(check_SOn(R))
# print(correct_SOn(R))
# print(check_SOn(correct_SOn(R)))

# R = np.array([[1,0,1],[1,1,1],[1,0.5,0.25]])
# print(check_SOn(correct_SOn(R)))
# print(check_SEn(R))
# print(correct_SEn(R))
# R2 = correct_SEn(R)
# print(check_SEn(R2))

# q = [1, 1, 0, 1]
# print(check_quaternion(q))
# print(correct_quaternion(q))
# q2 = correct_quaternion(q)
# print(check_quaternion(q2))