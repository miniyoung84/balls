import pybullet as p
import glob, os
import scipy.io
import time
import pybullet_data
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import time
#WIKI: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#

gravity = -9.81
num_frames = 30 #number of frames per simulation
num_samples = 100000 # number of simulations we want to run
frame_dimensions = 170

random.seed()

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
p.setTimeStep(1 / 30 / 50, physicsClientId=physicsClient)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #optionally
p.setGravity(0,0,gravity)
planeId = p.loadURDF("plane.urdf")
white_id = p.loadTexture("white.png")

texture_paths = glob.glob(os.path.join('dtd', '**', '*.jpg'), recursive=True)

start_time = time.strftime("%d_%b_%H-%M-%S", time.localtime())
os.mkdir(start_time)
print(f'                                                                ')
for sample_index in range(num_samples):
    print(f'\rSample {sample_index} of {num_samples} ({sample_index/num_samples: f}%)   ')    
    random_texture_path = texture_paths[random.randint(0, len(texture_paths) - 1)]
    plane_texture = p.loadTexture(random_texture_path)

    # Set up the physics body stuff
    drop_height = random.random() * 4 + 1
    mass = random.random() * 4 + 1
    restitution = random.random() * 0.7 + 0.2

    baseOrientation = p.getQuaternionFromEuler(np.random.rand(3) * 2 * math.pi)
    basePosition = [0, 0, drop_height]
    baseVelocity = np.random.rand(3) * 2.9 + 0.1
    baseAngular = np.random.rand(3) * 10
    ellipse= p.createCollisionShape(p.GEOM_MESH, fileName='sphere.obj', meshScale=[random.random() * 0.2 + 0.3, random.random() * 0.3 + 0.2, 0.5])
    physics_body = p.createMultiBody(mass, ellipse, -1, basePosition, baseOrientation)
    p.resetBaseVelocity(physics_body, baseVelocity, baseVelocity)
    p.changeDynamics(physics_body, -1, mass = mass)
    p.changeDynamics(physics_body, -1, restitution = restitution)
    p.changeDynamics(planeId, -1, restitution = 0.99)

    # p.changeVisualShape(planeId, -1, rgbaColor=[random.random(), random.random(), random.random(), 1], textureUniqueId=perlin_id, physicsClientId=physicsClient)
    p.changeVisualShape(planeId, -1, textureUniqueId=plane_texture, physicsClientId=physicsClient)
    p.changeVisualShape(physics_body, -1, rgbaColor=[random.random(), random.random(), random.random(), 1], textureUniqueId=white_id, physicsClientId=physicsClient)

    # randomize camera parameters
    max_camera_dist = 20
    min_camera_dist = 5
    camera_fov = random.random() * 20 * (1 if random.random() > 0.5 else -1) + 40
    camera_radius = random.randint(min_camera_dist, max_camera_dist)
    camera_phi = math.radians(random.randint(0, 360))
    camera_theta = math.radians(random.randint(0, 80))

    # randomize light parameters
    light_radius = random.randint(2 * min_camera_dist, 2 * max_camera_dist)
    light_phi = math.radians(random.randint(0, 360))
    light_theta = math.radians(random.randint(0, 85))
    light_color = np.random.rand(3) * 0.5 + 0.5

    light_x = math.sin(camera_theta) * math.cos(light_phi)
    light_y = math.sin(camera_theta) * math.sin(light_phi)
    light_z = math.cos(camera_theta)

    diffuse = random.random() * 0.7 + 0.3
    ambient = random.random() * 0.2 + 0.2
    specular = random.random() * 0.7 + 0.3

    # Find where the object will hit the ground
    roots = np.roots([0.5 * gravity, baseVelocity[2], drop_height])
    time_to_hit = np.max(np.real(roots))

    x_pos = baseVelocity[0] * time_to_hit
    y_pos = baseVelocity[1] * time_to_hit
    
    # Define view/proj matrices
    camera_x = camera_radius * math.sin(camera_theta) * math.cos(camera_phi) + x_pos
    camera_y = camera_radius * math.sin(camera_theta) * math.sin(camera_phi) + y_pos
    camera_z = camera_radius * math.cos(camera_theta)
    final_pos_relative = np.asarray([x_pos, y_pos, 0]) - np.asarray([camera_x, camera_y, camera_z])
    lookat = final_pos_relative + np.random.rand(3) / np.sqrt(np.sum(np.square(final_pos_relative))) * camera_fov / 60
    view_matrix = p.computeViewMatrix(cameraEyePosition=[camera_x, camera_y, camera_z], cameraTargetPosition=lookat, cameraUpVector=[0, 0, drop_height / 2])
    proj_matrix = p.computeProjectionMatrixFOV(fov=camera_fov, aspect=1.0, nearVal=0.01, farVal=2*max_camera_dist)

    #print("sphere 1 data: ", p.getJointInfo(sphere))

    #set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
    vel = np.zeros(num_frames)
    vel2 = np.zeros(num_frames)
    frames = np.zeros([num_frames, frame_dimensions * frame_dimensions, 3], dtype=np.uint8)

    bounce_frame = int(time_to_hit * 30)
    for i in range(int(bounce_frame - num_frames / 2)):
      for j in range(50):
        p.stepSimulation(physicsClientId=physicsClient)

    for i in range(num_frames):
        for j in range(50):
          p.stepSimulation(physicsClientId=physicsClient)
        
        # get an image
        [wid, hei, rgbPixels, dpth, segmask] = p.getCameraImage(width=frame_dimensions, height=frame_dimensions, viewMatrix=view_matrix, projectionMatrix=proj_matrix, lightDirection=[-light_x, -light_y, -light_z], lightDistance=light_radius, lightColor=light_color, shadow=1, renderer= p.ER_TINY_RENDERER, lightAmbientCoeff=ambient, lightDiffuseCoeff=diffuse, lightSpecularCoeff=specular, physicsClientId=physicsClient)

        img = np.asarray(rgbPixels, dtype=np.uint8)

        # plt.imshow(img)
        # plt.show()
        
        frames[i,:,0] = img[:,:,0].flatten()
        frames[i,:,1] = img[:,:,1].flatten()
        frames[i,:,2] = img[:,:,2].flatten()

        v = p.getBaseVelocity(physics_body)
        v_mag = np.sqrt((v[0][0]*v[0][0]) + (v[0][1]*v[0][1]) + (v[0][2]*v[0][2]))
        vel[i] = v_mag

        # print("sphere velocity ", v_mag)

    # Continue simulation and record where the thing lands
    z_vel = p.getBaseVelocity(physics_body)[0][2]
    while(z_vel > 0):
      p.stepSimulation(physicsClientId=physicsClient)
      z_vel = p.getBaseVelocity(physics_body)[0][2]

    count = 0
    while(z_vel < 0 and count < 1000 * 50):
      p.stepSimulation(physicsClientId=physicsClient)
      z_vel = p.getBaseVelocity(physics_body)[0][2]
      count += 1
    [final_position, final_orientation] = p.getBasePositionAndOrientation(physics_body)
    view_mat = np.asarray(view_matrix).reshape([4,4], order='F')
    final_position = np.matmul(view_mat, np.array([final_position[0], final_position[1], final_position[2], 1]))
    final_velocity = p.getBaseVelocity(physics_body)[0]
    final_velocity = np.matmul(view_mat, np.array([final_velocity[0], final_velocity[1], final_velocity[2], 0]))

    # Make a dict of our labels and data
    out_dict = {
      'frames': frames,
      'position': final_position[:-1],
      'velocity': final_velocity[:-1]
    }

    scipy.io.savemat(f'{start_time}{os.sep}sample_{sample_index}.npy', out_dict, do_compression=True)
    p.removeBody(physics_body)

p.disconnect()
