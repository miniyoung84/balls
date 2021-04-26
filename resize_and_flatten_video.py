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
  z = pos_label[2] # We got the axes wrong!
  pos_label[2] = pos_label[1]
  pos_label[1] = -z
  pos_label = -pos_label # wink wink don't mention this to H.M.S. Park
  for j in range(10):
    print(os.path.join(directory, f'image000{j}.png'))
    img = cv2.imread(os.path.join(directory, f'image000{j}.png'), cv2.IMREAD_COLOR)
    is_landscape = (img.shape[0] < img.shape[1])
    margin = int((img.shape[1] - img.shape[0] if is_landscape else img.shape[0] - img.shape[1]) / 2)
    for channel in range(3):
      color_channel = img[:,:,channel]
      # Crop to square
      if is_landscape:
        color_channel = color_channel[:, margin:-margin]
      else:
        color_channel = color_channel[margin:-margin, :]
      frames[j, :, int(2 - channel)] = cv2.resize(color_channel, (170, 170)).flatten()
    
  out_dict = {
      'frames': frames,
      'position': pos_label,
      'velocity': np.asarray([1, 0, 0])
  }
  scipy.io.savemat(f'real_data{os.sep}sample{i+1}.mat', out_dict, do_compression=True)
