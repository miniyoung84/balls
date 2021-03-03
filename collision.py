import pybullet as p
import time
import pybullet_data
import numpy as np
import math
import random
import matplotlib.pyplot as plt
#WIKI: https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#
max_dist = 9 #max distance the balls can be from each other
min_dist = 3 #min disrance the balls can be from each other
k_vel = 3 #constant that varies the velocity
rolling_mass = 5 # mass of the rolling ball
static_mass = 10 # mass of the initially static ball
restitution = 0.5 #bouncyness of contact. Keep it a bit less than 1, preferably closer to 0.
gravity = -9.81
Iter = 500 #number of iterations

#rollingFriction : torsional friction orthogonal to contact normal (keep this value very close to zero,
#otherwise the simulation can become very unrealistic.)
rollingFriction = 0.0

physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
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



#print("sphere 1 data: ", p.getJointInfo(sphere1))

#set the center of mass frame (loadURDF sets base link frame) startPos/Ornp.resetBasePositionAndOrientation(boxId, startPos, startOrientation)
vel = np.zeros(Iter)
vel2 = np.zeros(Iter)
for i in range (Iter):
    p.stepSimulation()
    #time.sleep(1./240.)
    time.sleep(1./480.)
    v = p.getBaseVelocity(sphere1)
    v2 = p.getBaseVelocity(sphere2)
    v_mag = np.sqrt((v[0][0]*v[0][0]) + (v[0][1]*v[0][1]))
    vel[i] = v_mag

    v_mag2 = np.sqrt((v2[0][0]*v2[0][0]) + (v2[0][1]*v2[0][1]))
    vel2[i] = v_mag2

plt.plot(vel)
plt.plot(vel2)
plt.show()
p.disconnect()
