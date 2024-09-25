# Computational Robotics
# Component 3
# Dylan Turner & Noor Hasan

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io

############################################################
#                  Rigid Body in Motion                    #
############################################################

'''
Input: start pose x0 ∈SE(2) and goal position xG in nSE(2). Output
a path (sequence of poses) that start at x0 and ends in xG.
'''
def interpolate_rigid_body(start_pose, goal_pose, steps = 10):
    #x,y interpolation
    x_vals = np.linspace(start_pose[0], goal_pose[0], steps)
    y_vals = np.linspace(start_pose[1], goal_pose[1], steps)
    
    #theta interpolation: make sure to go shortest path
    diff = goal_pose[2] - start_pose[2]
    if diff > np.pi:
        diff -= 2 * np.pi
    elif diff < -np.pi:
        diff += 2 * np.pi
        
    theta_vals = np.linspace(start_pose[2], start_pose[2] + diff, steps)
    
    # Create the path as a list of (x, y, theta) tuples
    path = []
    for i in range(len(x_vals)):
        x = x_vals[i]
        y = y_vals[i]
        theta = theta_vals[i]
        rotation = [[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]]
        state = np.eye(3)
        state[:2, :2] = rotation
        state[:2, 2] = [x,y]
        path.append(state)
    
    return path

'''
Input: start pose x0 ∈SE(2) and a Plan ( a sequence of N tuples (velocity, duration)) that is 
applied to the start pose and results in a path of N+ 1 states.
'''
def forward_propagate_rigid_body(start_pose, plan):    
    path = []
    start_pose = np.array(start_pose)
    path.append(start_pose)

    for i in range(len(plan)):
        #calculate change in each variable
        current_state = path[-1].copy()
        Vx, Vy, Vtheta, duration = plan[i]
        Dx, Dy, Dtheta = Vx * duration, Vy * duration, Vtheta * duration
        rotation = [[np.cos(Dtheta), -np.sin(Dtheta)],
                   [np.sin(Dtheta), np.cos(Dtheta)]]
        
        #calculate rotation
        current_rotation = current_state[:2, :2]
        new_rotation = current_rotation @ rotation
        
        #calculate translate in the current orientation's frame
        translation = np.array([Dx, Dy])
        current_position = current_state[:2, 2]
        new_position = current_position + current_rotation @ translation
        
        #create and append new state
        new_state = np.eye(3)
        new_state[:2, :2] = new_rotation
        new_state[:2, 2] = new_position
        path.append(new_state)

    return path

'''
Input: A path to visualize. Your visualization must include the path and an animation of the robot’s movement.
'''
def visualize_path(path, name = 'Animation.gif'):
    ax = plt.subplots()[1]
    ax.set_aspect('equal')
    ax.set_xlim([-10,10])
    ax.set_ylim([-10,10])
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title("Rigid Body Path Visualization")
    plt.grid(True)

    frames = []
    counter = 0
    for state in path:
        #extract info from state
        x, y = state[0, 2], state[1, 2]
        theta = np.arctan2(state[1, 0], state[0, 0])  

        #previous state info
        if(counter > 0):
            x2, y2 = path[counter - 1][0,2], path[counter - 1][1,2]

            #plot path
            ax.plot([x,x2], [y, y2], 'g')

        #plot state
        ax.plot(x, y, 'bo') 
        
        #plot orientation arrow
        ax.arrow(x, y, np.cos(theta) , np.sin(theta) , head_width=0.5, head_length=0.5, fc='r', ec='r')
        
        #save frame
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        frames.append(img)
        counter += 1

    #save gif
    frames[0].save(name, save_all=True, append_images=frames[1:], duration=500, loop=0)


############################################################
#                         Tests                            #
############################################################


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
visualize_path(forward_propagate_rigid_body(start, path), 'component_3_propogate.gif')

#Interpolate
start_pose = [0,-8,0]
goal_pose = [0,8,np.pi]
visualize_path(interpolate_rigid_body(start_pose, goal_pose, 10), 'component_3_interpolate.gif')