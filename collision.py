import pybullet as p
import glob, os
import time
import pybullet_data
import numpy as np
import math
import random
import matplotlib.pyplot as plt
#WIKI: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#

max_dist = 9 #max distance the balls can be from each other
min_dist = 2 #min disrance the balls can be from each other
k_vel = 3 #constant that varies the velocity
rolling_mass = 5 # mass of the rolling ball
static_mass = 10 # mass of the initially static ball
gravity = -9.81
num_frames = 100 #number of frames per simulation
num_samples = 500 # number of simulations we want to run

#rollingFriction : torsional friction orthogonal to contact normal (keep this value very close to zero,
#otherwise the simulation can become very unrealistic.)
rollingFriction = 0.0

random.seed()

physicsClient = p.connect(p.DIRECT)#or p.DIRECT for non-graphical version
p.setTimeStep(1 / 30, physicsClientId=physicsClient)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,gravity)
planeId = p.loadURDF("plane.urdf")
perlin_id = p.loadTexture("perlin.png")
white_id = p.loadTexture("white.png")

texture_paths = glob.glob(os.path.join('dtd', '**', '*.jpg'), recursive=True)
random_texture_path = texture_paths[random.randint(0, len(texture_paths) - 1)]

start_time = time.strftime("%d_%b_%H-%M-%S", time.localtime())
os.mkdir(start_time)
for sample_index in range(num_samples):
    plane_texture = p.loadTexture(random_texture_path)

    restitution = random.random() * 0.6 #bouncyness of contact. Keep it a bit less than 1, preferably closer to 0.
    drop_height = random.random() * (max_dist - min_dist) + min_dist

    startOrientation = p.getQuaternionFromEuler([0,0,0])
    sphere = p.loadURDF("sphere2.urdf",[0, 0, drop_height + 0.5], startOrientation)
    p.resetBaseVelocity(sphere, [0, 0, 0])
    # look on page 37-38 of https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#
    p.changeDynamics(sphere, -1, mass = rolling_mass)
    p.changeDynamics(sphere, -1, restitution = restitution)
    p.changeDynamics(sphere, -1, rollingFriction = rollingFriction)

    # p.changeVisualShape(planeId, -1, rgbaColor=[random.random(), random.random(), random.random(), 1], textureUniqueId=perlin_id, physicsClientId=physicsClient)
    p.changeVisualShape(planeId, -1, textureUniqueId=plane_texture, physicsClientId=physicsClient)
    p.changeVisualShape(sphere, -1, rgbaColor=[random.random(), random.random(), random.random(), 1], textureUniqueId=white_id, physicsClientId=physicsClient)

    # randomize camera parameters
    max_camera_dist = 20
    min_camera_dist = 2
    camera_radius = random.randint(min_camera_dist, max_camera_dist)
    camera_phi = math.radians(random.randint(0, 360))
    camera_theta = math.radians(random.randint(0, 85))

    camera_x = camera_radius * math.sin(camera_theta) * math.cos(camera_phi)
    camera_y = camera_radius * math.sin(camera_theta) * math.sin(camera_phi)
    camera_z = camera_radius * math.cos(camera_theta)

    # randomize light parameters
    light_radius = random.randint(2 * min_camera_dist, 2 * max_camera_dist)
    light_phi = math.radians(random.randint(0, 360))
    light_theta = math.radians(random.randint(0, 85))

    light_x = math.sin(camera_theta) * math.cos(light_phi)
    light_y = math.sin(camera_theta) * math.sin(light_phi)
    light_z = math.cos(camera_theta)

    view_matrix = p.computeViewMatrix(cameraEyePosition=[camera_x, camera_y, camera_z], cameraTargetPosition=[0, 0, 1], cameraUpVector=[0, 0, drop_height / 2])
    proj_matrix = p.computeProjectionMatrixFOV(fov=40.0, aspect=1.0, nearVal=0.5*min_camera_dist, farVal=2*max_camera_dist)

    #print("sphere 1 data: ", p.getJointInfo(sphere))

    #set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
    vel = np.zeros(num_frames)
    vel2 = np.zeros(num_frames)
    frames = np.zeros([num_frames, 250 * 250], dtype=np.uint8)
    for i in range(num_frames):
        p.stepSimulation(physicsClientId=physicsClient)
        
        # get an image
        [wid, hei, rgbPixels, dpth, segmask] = p.getCameraImage(width=250, height=250, viewMatrix=view_matrix, projectionMatrix=proj_matrix, lightDirection=[-light_x, -light_y, -light_z], lightDistance=light_radius, lightColor=[100/255, 100/255, 93/255], shadow=1, physicsClientId=physicsClient)

        img = np.asarray(rgbPixels)

        greyscale = np.array(list(map(lambda px: px[0] / 3.0 + px[1] / 3.0 + px[2] / 3.0, np.reshape(img, [250 * 250, 4]))), dtype=np.uint8)

        # plt.imshow(np.reshape(greyscale, [250, 250]), cmap="gray")
        # plt.show()
        
        frames[i,:] = greyscale
        print(f'Sample {sample_index} frame {i}/{num_frames}')    

        v = p.getBaseVelocity(sphere)
        v_mag = np.sqrt((v[0][0]*v[0][0]) + (v[0][1]*v[0][1]) + (v[0][2]*v[0][2]))
        vel[i] = v_mag

        # print("sphere velocity ", v_mag)

    np.save(f'{start_time}{os.sep}sample_{sample_index}.npy', frames)
    np.save(f'{start_time}{os.sep}sample_{sample_index}_restitution.npy', restitution)

p.disconnect()
