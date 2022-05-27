import matplotlib.pyplot as plt
import numpy as np


# (np.array((gt,dt)),np.squeeze(xc),np.squeeze(xd),np.squeeze(xdot),np.squeeze(qdot),np.squeeze(vision.targetPose),np.squeeze(vision.targetPoseRaw)))

# z data = logs/log_20220422-103056.dat # Motion along z axis
# x-y data = logs/log_20220422-103741.dat # Motion along x-y axis
# x-y-z data = logs/log_20220422-112251.dat # Motion along x-y-z axis

# new data 25 april
# z data = logs/log_20220425-100041.dat # Motion along z axis
# x-y-z data = logs/log_20220425-101505.dat# Motion along x-y-z axis
# sin along z = logs/log_20220425-103131.dat # shiftted
# sin sin sin along x-y-z = logs/log_20220425-104709.dat # 
# sin cos sin along x-y-z = logs/log_20220425-104948.dat # 
# sin cos sin large along x-y-z = logs/log_20220425-105246.dat # 
# sin cos sin large along x-y-z = logs/log_20220425-105631.dat # 





data = np.loadtxt ("logs/log_20220518-095824-exp-minus5xminus5y.dat")
# print (data.shape)


fig = plt.figure()

plt.plot(data[:,0],data[:,2],'r')
plt.plot(data[:,0],data[:,5],'r--')

plt.plot(data[:,0],data[:,3],'g')
plt.plot(data[:,0],data[:,6],'g--')

plt.plot(data[:,0],data[:,4],'b')
plt.plot(data[:,0],data[:,7],'b--')
plt.grid()
plt.show()


