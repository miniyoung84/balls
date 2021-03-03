import pybullet as p
import time
import pybullet_data
import numpy as np
import math
import random
import matplotlib.pyplot as plt

max_dist = 15 #max height
min_dist = 5 #min height
k_vel = 3 #constant that varies the velocity
mass = 1 # mass of the ball
restitution = 0.9 #bouncyness of contact. Keep it a bit less than 1, preferably closer to 0.
gravity = -9.8


#rollingFriction : torsional friction orthogonal to contact normal (keep this value very close to zero,
#otherwise the simulation can become very unrealistic.)
rollingFriction = 0.0

random.seed()

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,gravity)
planeId = p.loadURDF("plane.urdf")
perlin_id = p.loadTexture("perlin.png")
white_id = p.loadTexture("white.png")

#Get random distance from the center ball

h = random.randint(min_dist, max_dist)


startOrientation = p.getQuaternionFromEuler([0,0,0])
sphere1 = p.loadURDF("sphere2.urdf",[0, 0,4], startOrientation)
p.resetBaseVelocity(sphere1, [0, 0 , -1])
p.changeDynamics(sphere1, -1, mass = mass)
p.changeDynamics(sphere1, -1, restitution = restitution)
p.changeDynamics(sphere1, -1, rollingFriction = rollingFriction)


# ****************p.changeVisualShape(planeId, -1, rgbaColor=[random.random(), random.random(), random.random(), 1], textureUniqueId=perlin_id, physicsClientId=physicsClient)
# ****************p.changeVisualShape(sphere1, -1, rgbaColor=[random.random(), random.random(), random.random(), 1], textureUniqueId=white_id, physicsClientId=physicsClient)
# ****************p.changeVisualShape(sphere2, -1, rgbaColor=[random.random(), random.random(), random.random(), 1], textureUniqueId=white_id, physicsClientId=physicsClient)

# randomize camera parameters
# ****************max_camera_dist = 50
# ****************min_camera_dist = 3
# ****************camera_radius = random.randint(min_camera_dist, max_camera_dist)
# ****************camera_phi = math.radians(random.randint(0, 360))
# ****************camera_theta = math.radians(random.randint(0, 85))

# ****************camera_x = camera_radius * math.sin(camera_theta) * math.cos(camera_phi)
# ****************camera_y = camera_radius * math.sin(camera_theta) * math.sin(camera_phi)
# ****************camera_z = camera_radius * math.cos(camera_theta)

# ****************view_matrix = p.computeViewMatrix(cameraEyePosition=[camera_x, camera_y, camera_z], cameraTargetPosition=[0, 0, 0], cameraUpVector=[0, 0, 1])
# ****************proj_matrix = p.computeProjectionMatrixFOV(fov=40.0, aspect=1.0, nearVal=0.5*min_camera_dist, farVal=2*max_camera_dist)

#print("sphere 1 data: ", p.getJointInfo(sphere1))

#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
Iter = 500
vel = np.zeros(Iter)

# ****************frames = np.zeros([Iter, 250 * 250], dtype=np.uint8)
for i in range (Iter):
    p.stepSimulation()
    
    # get an image
    # ****************[wid, hei, rgbPixels, dpth, segmask] = p.getCameraImage(width=250, height=250, viewMatrix=view_matrix, projectionMatrix=proj_matrix, shadow=1, physicsClientId=physicsClient)

    # ****************img = np.asarray(rgbPixels)
    # ****************greyscale = np.zeros([250, 250], dtype=np.uint8)
    # ****************for j in range(250):
    # ****************    for k in range(250):
    # ****************        greyscale[j, k] = img[j, k, 0] / 3.0 + img[j, k, 1] / 3.0 + img[j, k, 2] / 3.0
    
    ## **************** plt.imshow(np.reshape(greyscale, [250, 250]), cmap="gray")
    # # ****************plt.show()
    
    # ****************frames[i,:] = greyscale.flatten()
    # ****************print(i)    

    time.sleep(1./240.)

    v = p.getBaseVelocity(sphere1)


    v_mag = np.sqrt((v[0][0]*v[0][0]) + (v[0][1]*v[0][1]) + (v[0][2]*v[0][2]))
    vel[i] = v_mag

    print("sphere 1 velocity ", v_mag)

# ****************np.save(f'runtimebaby.npy', frames)

plt.plot(vel)
plt.show()
p.disconnect()
