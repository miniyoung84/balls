import torch, os, glob
from torch import nn
import scipy.io
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision
import json

# A print module for debugging
class Print(nn.Module):
    def forward(self, x):
        print(x.size())
        return x

class Squeeze(nn.Module):
    def forward(self, x):
        return torch.squeeze(x)

# (E)arly (F)usion (B)ounce (Net)work!
class EFBNet(nn.Module):
    def __init__(self, num_frames_fused):
        super(EFBNet, self).__init__()
        
        # Architecture copied from https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42455.pdf
        #   (which, in turn, adapts architecture from ImageNet)
        # Has some baked-in assumtions on frame dimensions :/
        self.sequence = nn.Sequential(
            nn.Conv3d(3, 96, (num_frames_fused, 11, 11), stride=(1, 3, 3), padding=(0, 5, 5)), # Convolves along u and v
            Squeeze(),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)), # Max pools spatial dims only
            nn.Conv2d(96, 256, (5, 5), stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),    
            nn.Conv2d(256, 384, (3, 3), stride=1, padding=1),
            nn.Conv2d(384, 384, (3, 3), stride=1, padding=1),
            nn.Conv2d(384, 256, (3, 3), stride=1, padding=1),
            nn.MaxPool2d((2, 2)),    
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 4096),
            nn.Linear(4096, 6) # ig we have it output the x,y,z coordinates of where the ball will land and the x,y,z velocity
        )

    def forward(self, x):
        """ x should be a tensor of shape (N, C, T, H, W)
                N = batch size
                C = color channel count = 3
                T = num frames to merge
                H = frame height = 170
                W = frame width = 170
        """

        pred = self.sequence(x)
        return pred


class BounceDataset(Dataset):

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

        trimmed_frame = np.array(sample['frames'][start_frame:end_frame, :,:], dtype=float)
        frames = np.reshape(trimmed_frame, [num_frames_to_fuse, 170, 170, 3])
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

def load_data(num_frames_to_fuse):
    bounce_frame = 14 # clips already generated such that bounce occurs on frame 14 or 15
    start_frame = int(bounce_frame - 0.5 * num_frames_to_fuse)
    end_frame = int(bounce_frame + 0.5 * num_frames_to_fuse)

    # Find paths to each mat
    sample_paths = glob.glob(os.path.join('simulated_data', '**', '*.npy'), recursive=True)
    sample_paths = sample_paths[:1000]

    dataset = np.zeros([len(sample_paths), 3, num_frames_to_fuse, 170, 170], dtype=float)
    labels = np.zeros([len(sample_paths), 6])
    for i in range(len(sample_paths)):
        sample = scipy.io.loadmat(sample_paths[i])
        trimmed_frame = np.array(sample['frames'][start_frame:end_frame, :,:], dtype=float)
        frames = np.reshape(trimmed_frame, [num_frames_to_fuse, 170, 170, 3])
        for c in range(3):
            for j in range(num_frames_to_fuse):
                channel_mean = np.mean(frames[j,:,:,c])
                frames[j,:,:,c] -= channel_mean
        pos_label = sample['position'][0]
        vel_label = sample['velocity'][0]
        labels[i,:3] = pos_label[:]
        labels[i,3:6] = vel_label[:]

        # Rearrange axes of frames to match dataset
        dataset[i,:,:,:,:] = np.transpose(frames, axes=[3, 0, 1, 2])
    
    return torch.from_numpy(dataset), torch.from_numpy(labels)
    
if __name__ == '__main__':
    # Use GPU if available
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    print('Using {} device'.format(device))

    batch_size = 64
    num_frames = 10

    # Gotta read that fuckin data in bro
    # also maybe add a bit of noise to the images??
    sample_paths = glob.glob(os.path.join('simulated_data', '**', '*.npy'), recursive=True)
    dataset = DataLoader(BounceDataset(sample_paths), batch_size=batch_size, shuffle=True, num_workers=0)

    # Instantiate model
    model = EFBNet(num_frames).to(device) # Create the network and send it to the GPU
    L2_dist = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    for epoch in range(1000000):
        net_loss = 0
        print(f'Epoch {epoch}')
        for i in range(len(dataset)):
            batch = next(iter(dataset))
            batch_x = batch['tensor'].float()
            batch_y = batch['label'].float()

            # Predict
            y_pred = model(batch_x)

            # Find loss
            loss = L2_dist(y_pred, batch_y)
            net_loss += loss.item()
            print(f'    Loss of batch {i} = {loss.item()}')

            # Backpropagate thru some syntactic magic
            optimizer.zero_grad() # Zero optimizer's gradients
            loss.backward() # Backpropagate loss
            optimizer.step()
        
        json = json.dumps(model.state_dict)
        f = open(f'last_state_dict.json',"w")
        f.write(json)
        f.close()

        print(f'Epoch {epoch}. Loss = {net_loss}')