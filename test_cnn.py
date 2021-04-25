import torch, os, glob
from torch import nn
import scipy.io
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import matplotlib.pyplot as plt
from train_cnn import *
import sys

class RealDataset(Dataset):

    def __init__(self, sample_paths, transform=None):
        self.sample_paths = sample_paths
        self.transform = transform

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, idx):
        num_frames_to_fuse = 10

        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = scipy.io.loadmat(self.sample_paths[idx])

        # Do some preprocessing on the data
        bounce_frame = 14 # clips already generated such that bounce occurs on frame 14 or 15
        start_frame = int(bounce_frame - 0.5 * num_frames_to_fuse)
        end_frame = int(bounce_frame + 0.5 * num_frames_to_fuse)

        frames = np.array(sample['frames'], dtype=float).reshape([10, 170, 170, 3])
        for c in range(3):
            for j in range(num_frames_to_fuse):
                channel_mean = np.mean(frames[j,:,:,c])
                frames[j,:,:,c] -= channel_mean
        pos_label = sample['position'][0]
        vel_label = sample['velocity'][0]
        label = np.zeros(6)
        label[:3] = pos_label[:]
        label[3:6] = vel_label[:]

        # Rearrange axes of frames to match dataset
        data_object = torch.from_numpy(np.transpose(frames, axes=[3, 0, 1, 2]))

        full_sample = {'tensor': data_object, 'label': torch.from_numpy(label)}

        if self.transform:
            full_sample = self.transform(full_sample)

        return full_sample

model_save_path = 'most_recent_model.pt'

if __name__ == '__main__':
  batch_size = 64
  # Load model
  device = torch.device('cpu')
  loaded_model = torch.load(model_save_path, map_location=device).to(device)
  loaded_model.eval()
  model = EFBNet(10).to(device)
  model.load_state_dict(loaded_model.state_dict(), strict=False)
  model.eval()

  # Load test data
  sample_paths = []
  dataset, data_loader = None, None
  color = None
  if len(sys.argv) < 2 or (sys.argv[1] != 'sim' and sys.argv[1] != 'real'):
    print(f'usage: {sys.argv[0]} <\'real\' or \'sim\'>')
    exit()
  elif sys.argv[1] == 'real':
    sample_paths = glob.glob(os.path.join('real_data', '**', '*.mat'), recursive=True)
    dataset = RealDataset(sample_paths)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    color = 'r'
  else:
    sample_paths = glob.glob(os.path.join('test_data', '**', '*.npy'), recursive=True)
    dataset = BounceDataset(sample_paths)
    data_loader = DataLoader(dataset, batch_size=batch_size)
    color = 'b'


  # Metrics
  euclidean_dist = lambda x, y: np.sqrt(np.sum(np.square(x - y)))
  cosine_dist = lambda x, y: 1 - (np.dot(x, y) / (np.sqrt(np.sum(np.square(x))) * np.sqrt(np.sum(np.square(y)))))
  magnitude_percent_error = lambda x, y: np.abs(np.sqrt(np.sum(np.square(x))) - np.sqrt(np.sum(np.square(y)))) / np.sqrt(np.sum(np.square(y))) * 100

  # Find error for each data point
  err_pos, err_vel = np.zeros([len(dataset), 3]), np.zeros([len(dataset), 2])
  for i in range(len(data_loader)):
    print(f'Batch {i+1}/{len(data_loader)}')
    batch = next(iter(data_loader))
    sample, label = batch['tensor'].float(), batch['label'].numpy()

    # Predict
    prediction = model(sample).detach().numpy()

    for j in range(prediction.shape[0]):
      pred_pos = prediction[j, :3]
      pred_vel = prediction[j, 3:]
      true_pos = label[j,:3]
      true_vel = label[j,3:]

      # Compute error
      err_pos[i+j,0] = euclidean_dist(pred_pos, true_pos)
      err_pos[i+j,1] = cosine_dist(pred_pos, true_pos)
      err_pos[i+j,2] = magnitude_percent_error(pred_pos, true_pos)
      err_vel[i+j,0] = euclidean_dist(pred_vel, true_vel)
      err_vel[i+j,1] = cosine_dist(pred_vel, true_vel)

  # Remove NaNs because i'm a bitch
  err_pos[np.isnan(err_pos)] = -1
  err_vel[np.isnan(err_vel)] = -1

  # Visualize position error
  fig=plt.figure()
  fig.add_subplot(1, 3, 1)
  ax = plt.gca()
  plt.hist(err_pos[:,0], bins=10, color=color)
  plt.title(f'Mean error in position: {np.mean(err_pos[:,0]) : 0.3f}')
  ax.set_xlabel('Euclidean distance from bounce location')
  ax.set_ylabel('Frequency')

  fig.add_subplot(1, 3, 2)
  ax = plt.gca()
  plt.hist(err_pos[:,1], bins=10, color=color)
  plt.title(f'Mean error in position: {np.mean(err_pos[:,1]) : 0.3f}')
  ax.set_xlabel('Cosine distance')
  ax.set_ylabel('Frequency')

  fig.add_subplot(1, 3, 3)
  ax = plt.gca()
  plt.hist(err_pos[:,2], bins=10, color=color)
  plt.title(f'Mean percent error in position magnitude: {np.mean(err_pos[:,2]) : 0.3f}')
  ax.set_xlabel('Percent error')
  ax.set_ylabel('Frequency')

  if sys.argv[1] == 'sim':
    # Visualize velocity error
    fig=plt.figure()
    fig.add_subplot(1, 2, 1)
    ax = plt.gca()
    plt.hist(err_vel[:,0], bins=10, color=color)
    plt.title(f'Mean error in velocity: {np.mean(err_vel[:,0]) : 0.3f}')
    ax.set_xlabel('Euclidean distance from bounce velocity')
    ax.set_ylabel('Frequency')

    fig.add_subplot(1, 2, 2)
    ax = plt.gca()
    plt.hist(err_vel[:,1], bins=10, color=color)
    plt.title(f'Mean error in velocity: {np.mean(err_vel[:,1]) : 0.3f}')
    ax.set_xlabel('Cosine distance')
    ax.set_ylabel('Frequency')

  plt.show()





