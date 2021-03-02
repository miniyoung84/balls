import pybullet as p
import time
import pybullet_data
import numpy as np
import math
import random
import matplotlib.pyplot as plt

max_dist = 9 #max distance the balls can be from each other
min_dist = 3 #min disrance the balls can be from each other
k_vel = 3 #constant that varies the velocity
rolling_mass = 5 # mass of the rolling ball
static_mass = 10 # mass of the initially static ball
restitution = 0.97 #bouncyness of contact. Keep it a bit less than 1, preferably closer to 0.
gravity = -9.8


#rollingFriction : torsional friction orthogonal to contact normal (keep this value very close to zero,
#otherwise the simulation can become very unrealistic.)
rollingFriction = 0.0

physicsClient = p.connect(p.DIRECT)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,gravity)
planeId = p.loadURDF("plane.urdf")

#Get random distance from the center ball
while(True):
    r_dist = random.randint(-max_dist, max_dist)
    if abs(r_dist) >= min_dist: #have a minimum distance
        break


#get random angle
angle = math.radians(random.randint(0, 360))

x_dist = r_dist*math.cos(angle)
y_dist = r_dist*math.sin(angle)

#find the velocity
x_vel = -x_dist*k_vel
y_vel = -y_dist*k_vel



print("x = ", x_dist, 'Y = ',y_dist  )
print("x vel = ", x_vel, 'Y vel = ', y_vel  )

startOrientation = p.getQuaternionFromEuler([0,0,0])
sphere1 = p.loadURDF("sphere2.urdf",[x_dist, y_dist, 0.5], startOrientation)
sphere2 = p.loadURDF("sphere2.urdf",[0,0,0.5], startOrientation)
p.resetBaseVelocity(sphere1, [x_vel, y_vel , 0])
p.changeDynamics(sphere2, -1, mass = static_mass) # look on page 37-38 of https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#
p.changeDynamics(sphere1, -1, mass = rolling_mass)
p.changeDynamics(sphere1, -1, restitution = restitution)
p.changeDynamics(sphere2, -1, restitution = restitution)
p.changeDynamics(sphere1, -1, rollingFriction = rollingFriction)
p.changeDynamics(sphere2, -1, rollingFriction = rollingFriction)

# randomize camera parameters
max_camera_dist = 50
min_camera_dist = 3
camera_radius = random.randint(min_camera_dist, max_camera_dist)
camera_phi = math.radians(random.randint(0, 90))
camera_theta = math.radians(random.randint(0, 360))

camera_x = camera_radius * math.sin(camera_theta) * math.cos(camera_phi)
camera_y = camera_radius * math.sin(camera_theta) * math.sin(camera_phi)
camera_z = camera_radius * math.cos(camera_theta)

view_matrix = p.computeViewMatrix(cameraEyePosition=[camera_x, camera_y, camera_z], cameraTargetPosition=[0, 0, 0], cameraUpVector=[0, 0, 1])
proj_matrix = p.computeProjectionMatrixFOV(fov=40.0, aspect=1.0, nearVal=0.5*min_camera_dist, farVal=2*max_camera_dist)

#print("sphere 1 data: ", p.getJointInfo(sphere1))

#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
Iter = 200
vel = np.zeros(Iter)
vel2 = np.zeros(Iter)
frames = np.zeros([Iter, 250 * 250], dtype=np.uint8)
for i in range (Iter):
    p.stepSimulation()
    
    # get an image
    [wid, hei, rgbPixels, dpth, segmask] = p.getCameraImage(width=250, height=250, viewMatrix=view_matrix, projectionMatrix=proj_matrix)

    img = np.asarray(rgbPixels)
    greyscale = np.zeros([250, 250], dtype=np.uint8)
    for j in range(250):
        for k in range(250):
            greyscale[j, k] = img[j, k, 0] / 3.0 + img[j, k, 1] / 3.0 + img[j, k, 2] / 3.0
    
    # plt.imshow(np.reshape(greyscale, [250, 250]), cmap="gray")
    # plt.show()
    
    frames[i,:] = greyscale.flatten()
    print(i)    

    #time.sleep(1./240.)
    # time.sleep(1./480.)
    v = p.getBaseVelocity(sphere1)
    v2 = p.getBaseVelocity(sphere2)
#    print("v = ", v)#
    #print("v[0][0]" ,  v[0][0])
    #print("v[0][1]" ,  v[0][1])
    v_mag = np.sqrt((v[0][0]*v[0][0]) + (v[0][1]*v[0][1]))
    vel[i] = v_mag

    v_mag2 = np.sqrt((v2[0][0]*v2[0][0]) + (v2[0][1]*v2[0][1]))
    vel2[i] = v_mag2

    #print("sphere 1 velocity ", v_mag)

np.save(f'runtimebaby.npy', frames)

plt.plot(vel)
plt.plot(vel2)
plt.show()
p.disconnect()
