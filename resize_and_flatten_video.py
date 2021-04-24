import cv2
import numpy as np
import os, glob
import scipy.io

sample_paths = os.listdir('real_data_raw')

for i in range(len(sample_paths)):
  directory = os.path.join('real_data_raw', sample_paths[i])
  labels = np.loadtxt('real_data_labels.csv', delimiter=',', skiprows=1)

  frames = np.zeros([10, 170 * 170, 3], dtype=np.uint8)
  pos_label = labels[i,1:] / 100 # from cm to meters
  for j in range(10):
    print(os.path.join(directory, f'image000{j}.png'))
    img = cv2.imread(os.path.join(directory, f'image000{j}.png'), cv2.IMREAD_COLOR)
    for channel in range(3):
      color_channel = img[:,:,channel]
      color_channel = color_channel[:,420:-420] # Crop to square
      frames[j, :, int(2 - channel)] = cv2.resize(color_channel, (170, 170)).flatten()
    
  out_dict = {
      'frames': frames,
      'position': pos_label,
      'velocity': np.asarray([1, 0, 0])
  }
  scipy.io.savemat(f'real_data{os.sep}sample{i+1}.mat', out_dict, do_compression=True)