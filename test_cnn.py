import torch, os, glob
from torch import nn
import scipy.io
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import matplotlib.pyplot as plt
from train_cnn import *

model_save_path = 'most_recent_model.pt'

if __name__ == '__main__':
  batch_size = 64
  # Load model
  model = torch.load(model_save_path)
  model.to('cpu')

  # Load test data
  sample_paths = glob.glob(os.path.join('test_data', '**', '*.npy'), recursive=True)
  dataset = BounceDataset(sample_paths)
  data_loader = DataLoader(dataset, batch_size=batch_size)

  # Find error for each data point
  err_pos, err_vel = np.zeros(len(dataset)), np.zeros(len(dataset))
  for i in range(len(data_loader)):
    print(f'Batch {i+1}/{len(data_loader)}')
    batch = next(iter(data_loader))
    sample, label = batch['tensor'].float(), batch['label'].numpy()

    # Predict
    prediction = model(sample).detach().numpy()

    for j in range(prediction.shape[0]):
      pred_pos = prediction[j, :3]
      pred_vel = prediction[j, 3:]

      # Compute error
      err_pos[i+j] = np.sqrt(np.dot(pred_pos - label[j,:3], pred_pos - label[j,3:])) # L2 norm
      err_vel[i+j] = np.dot(pred_vel, label[j,3:]) / np.sqrt(np.dot(pred_vel, pred_vel)) / np.sqrt(np.dot(label[j,3:], label[j, 3:])) # cosine similarity

  # Remove NaNs because i'm a bitch
  err_pos[np.isnan(err_pos)] = -1
  err_vel[np.isnan(err_vel)] = -1

  print(f'Mean error in position: {np.mean(err_pos)}, median: {np.median(err_pos)}')
  print(f'Mean error in velocity: {np.mean(err_vel)}, median: {np.median(err_vel)}')

  # Visualize position error
  plt.figure()
  ax1 = plt.gca()
  plt.hist(err_pos, bins=20)
  plt.title('Error in position prediction')
  ax1.set_xlabel('Euclidean distance from bounce location')
  ax1.set_ylabel('Frequency')

  # Visualize velocity error
  plt.figure()
  ax2 = plt.gca()
  plt.hist(err_vel, bins=20)
  plt.title('Error in velocity prediction')
  ax2.set_xlabel('Cosine similarity')
  ax2.set_ylabel('Frequency')

  plt.show()





