# Computational Robotics
# Component 4
# Dylan Turner & Noor Hasan

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

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
def interpolate_arm(start, goal, steps):
    path = []
    l1_length = 2
    l2_length = 1.5
    
    for step in range(steps + 1):
        # Linear interpolation of angles
        theta1 = start[0] + (goal[0] - start[0]) * step / steps
        theta2 = start[1] + (goal[1] - start[1]) * step / steps
        
        #transform for link 1 relative to base
        transform1 = np.eye(3)
        transform1[:2, :2] = np.array([[np.cos(theta1), -np.sin(theta1)],
                     [np.sin(theta1),  np.cos(theta1)]])
        transform1[:2, 2] = [l1_length * np.cos(theta1), l1_length * np.sin(theta1)]
        
        #transform for link 2 relative to the end of link 1
        transform2 = np.eye(3)
        transform2[:2, :2] = np.array([[np.cos(theta2), -np.sin(theta2)],
                     [np.sin(theta2),  np.cos(theta2)]])
        transform2[:2, 2] = [l2_length * np.cos(theta2), l2_length * np.sin(theta2)]  

        #cmbined transformations: end effector relative to base.
        combined_transform = np.dot(transform1, transform2)
        
        path.append([transform1, combined_transform])
    
    return path

#Input: start pose q0 ∈SE(2) and a Plan ( a sequence of N tuples
# (velocity, duration)) that is applied to the start pose and results in a path of N+ 1 states.
def forward_propagate_arm(start_pose, plan):
    path = []
    l1_length = 2
    l2_length = 1.5
    prev_angle1 = start_pose[0]
    prev_angle2 = start_pose[1]

    for i in range(len(plan)):
        
        #calculate new angle
        dtheta1 = plan[i][0] * plan[i][2]
        dtheta2 = plan[i][1] * plan[i][2]
        
        theta1 = prev_angle1 + (dtheta1 - start_pose[0])
        theta2 = prev_angle2 + (dtheta2 - start_pose[1])
       
        #transform for link 1 relative to base
        transform1 = np.eye(3)
        transform1[:2, :2] = np.array([[np.cos(theta1), -np.sin(theta1)],
                     [np.sin(theta1),  np.cos(theta1)]])
        transform1[:2, 2] = [l1_length * np.cos(theta1), l1_length * np.sin(theta1)]
        
        #transform for link 2 relative to the end of link 1
        transform2 = np.eye(3)
        transform2[:2, :2] = np.array([[np.cos(theta2), -np.sin(theta2)],
                     [np.sin(theta2),  np.cos(theta2)]])
        transform2[:2, 2] = [l2_length * np.cos(theta2), l2_length * np.sin(theta2)]  

        #cmbined transformations: end effector relative to base.
        combined_transform = np.dot(transform1, transform2)
        
        path.append([transform1, combined_transform])
        prev_angle1 = prev_angle1 + dtheta1
        prev_angle2 = prev_angle2 + dtheta2

    return path

#Input: A path to visualize. Your visualization must include the path
# and an animation of the robot’s movement.
def visualize_arm_path(path, name = 'Animation2.gif', speed = 250):
    ax = plt.subplots()[1]

    frames = []
    prev_end_point = [path[0][1][0,2], path[0][1][1,2]]
    path_x = [prev_end_point[0]]
    path_y = [prev_end_point[1]]
    for transform1, transform2 in path:
        ax.cla()
        ax.set_aspect('equal')
        ax.set_xlim([-5,5])
        ax.set_ylim([-5,5])
        plt.xlabel("X position")
        plt.ylabel("Y position")
        plt.grid(True)
        plt.title("Movement of an Arm Visualization")

        #link 1 end
        x1, y1 = transform1[0, 2], transform1[1, 2]
        #link 2 end
        x2, y2 = transform2[0, 2], transform2[1, 2]
        
        #plot the path from previous end point to new end point
        path_x.append(x2)
        path_y.append(y2)
        ax.plot(path_x, path_y, 'go-', label='Path', markersize=0)

        #plot from base to link 1
        ax.plot([0, x1], [0, y1], 'ro-', label='Link 1')
        #plot from link 1 to link 2
        ax.plot([x1, x2], [y1, y2], 'bo-', label='Link 2')

        #save frame
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        frames.append(img)
        prev_end_point = [x2, y2]

    #save gif
    frames[0].save(name, save_all=True, append_images=frames[1:], duration=speed, loop=0)

#TODO: In your report include and analyze any design choice of your implementation. Include visual-
# izations of the paths generated by both methods. Submit an example animation (gif is prefered
# but can also be a small mp4).


start_angles = (np.deg2rad(0), np.deg2rad(0))
goal_angles = (np.deg2rad(360), np.deg2rad(-180))
path = interpolate_arm(start_angles, goal_angles, 25)
visualize_arm_path(path, 'Interplolate Arm.gif', 50)


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

path = forward_propagate_arm(start_angles, plan)
visualize_arm_path(path, 'Foward Propogate Arm.gif', 20)