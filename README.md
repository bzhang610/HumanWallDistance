# Human wall distance detection
This program finds the distance between a human and the furtherest wall in a corridor using a depth image. The program will also output which side the furtherest wall is on.  

## Environment
The python script is written with Python 3.6.5 and uses the following libraries.
* Numpy 1.15.4  
* Opencv 4.0.0    
* Scipy 1.2.1    

## Files
* **[find_clearance.py](utils.py) :** Main executable python script
* **[utils.py](utils.py) :** Functions used in the main script
* **[result_showcase.ipynb](result_showcase.ipynb) :** Jupyter notebook that showcases the result at each step 

## Approach
The program tackles the clearance detection task using the following steps: 
1. Reconstruct world frame coordinates from depth image pixels  
Using the range of vertical and horizontal field of view data from the camera, assuming each pixel is at the same size and no lens distortion, we can calculate the corresponding coordinates of each pixel represents in the world frame.   
We set the camera to be the origin in the world frame, x-axis perpendicular to the walls pointing to the right, y-axis parallel to the wall and pointing away from the camera, z-axis pointing upwards perpendicular to the ground.   
We remove points that are too close or too far, and remove the points representing the ground.  


2. Get contours of obstacles in the images  
Using the x y coordinates, we can reconstruct a sudo bird's eye view from the depth image. Assuming the obstable(human) is a cylindar, the obstacle's y value range in the reconstructed bird's eye view image should be approximately half of the real object. We use this to determine the range of y values used in the xy image to find the contours of the obstacle and walls. 


3. Calculate the gap size and make decision  
Once we have the contours of the obstacle and the walls, we calculate the minimum gap on each side of the obstacle. Finally, the program returns the side with the larger gap and the gap size in meters.

## Generalization
Corner Cases:
* The human is too close to the wall and the program cannot distinguish the contour of the human from the wall  
If this is the case, then in the step of finding contours, it is possible for the human bounding box to be the left or right most. The program has already accounted for these cases and should perform correctly.   


* The human is partly connected to the wall (e.g. one arm touching the wall but enough space underneath)
In this program, when selecting the points for finding contours, there is only a low limit for Z values to remove the ground. If we have the information about the robot's height, we can also set a high limit for Z values that we only need to consider the gap size under such height.  


In order to generalize the program, we will look at the assumptions we made in this program: 
* The robot is at the center of the corridor and parallel to the walls  
If the robot is positioned arbitrarily, both location and orientation, we can still calculate the XYZ coordinates from the depth image, but they would be in the camera frame. We then can obtain the world frame coordinates through the pose of the camera wTc in SE(3):  
\begin{equation}
wTc = 
\begin{bmatrix} 
R & p \\
0 & 1 
\end{bmatrix}
\end{equation} 
where R is the orientation and p = [x,y,z] is the location of the robot in the world frame. Then we can obtain the world frame coordinates for each point in the camera frame through:

\begin{equation}
\begin{bmatrix} 
x_W\\
y_W \\
z_W\\
1\\
\end{bmatrix}
=
T
\begin{bmatrix} 
x_B \\
y_B \\
z_B\\
1\\
\end{bmatrix}
\end{equation}

* Ground is flat  
Since the task is detecting human in corridors, if the ground is not flat, we can still assume it is smooth but sloped, we then need to approximate the slope and remove the ground accordingly.  
We can also use pretrained network to identify background and floor for more complicated ground removal cases.  


* There is only one obstacle  
In this program, we assume there is only one human in the depth image and therefore 2 gaps to choose from. This allow us to assume the obstacle has contour closest to the center of the image. When we have multiple obstacles, we need to locate all gaps and the corresponding sizes.  
We can also use a different approach to detect human obstacles. Instead of simply setting the bounding box closest to the center as human and others as walls, we can use pretrained CNN networks to identify humans from walls, then calculate the appropriate gaps between humans and walls.  
