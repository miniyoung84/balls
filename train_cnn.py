import torch, os, glob
from torch import nn as nn
import torch.nn.functional as nnf
import scipy.io
import numpy as np

# A print module for debugging
class Print(nn.Module):
    def forward(self, x):
        print(x.size())
        return x

# (E)arly (F)usion (B)ounce (Net)work!
class EFBNet(nn.Module):
    def __init__(self, num_frames_fused):
        super(EFBNet, self).__init__()

        # Architecture copied from https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/42455.pdf
        #   (which, in turn, adapts architecture from ImageNet)
        # Has some baked-in assumtions on frame dimensions :/
        self.sequence = nn.Sequential(
            nn.Conv2d(3, 96, (11, 11, num_frames_fused), stride=3, padding=5), # Convolves along u and v
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Max pools spatial dims only
            nn.Conv2d(96, 256, (5, 5, num_frames_fused), stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(256, 384, (3, 3, num_frames_fused), stride=1, padding=1),
            nn.Conv2d(384, 384, (3, 3, num_frames_fused), stride=1, padding=1),
            nn.Conv2d(384, 256, (3, 3, num_frames_fused), stride=1, padding=1),
            nn.MaxPool2d(2, 2),
            Print(),
            nn.Flatten(),
            nn.Linear(256 * num_frames_fused, 4096), # ngl I have no idea if these dimensions will work with our videos, print statement above should let us know what to put here
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


def load_data(num_frames_to_fuse):
    bounce_frame = 14 # clips already generated such that bounce occurs on frame 14 or 15
    start_frame = int(bounce_frame - 0.5 * num_frames_to_fuse)
    end_frame = int(bounce_frame + 0.5 * num_frames_to_fuse)

    # Find paths to each mat
    sample_paths = glob.glob(os.path.join('simulated_data', '**', '*.npy'), recursive=True)
    sample_paths = sample_paths[:1000]

    dataset = np.zeros([len(sample_paths), 3, num_frames_to_fuse, 170, 170], dtype=np.uint8)
    labels = np.zeros([len(sample_paths), 6])
    for i in range(len(sample_paths)):
        sample = scipy.io.loadmat(sample_paths[i])
        trimmed_frame = sample['frames'][start_frame:end_frame, :,:]
        frames = np.reshape(trimmed_frame, [num_frames_to_fuse, 170, 170, 3])
        pos_label = sample['position'][0]
        vel_label = sample['velocity'][0]
        labels[i,:3] = pos_label[:]
        labels[i,3:6] = vel_label[:]

        # Rearrange axes of frames to match dataset
        dataset[i,:,:,:,:] = np.transpose(frames, axes=[3, 0, 1, 2])

    return torch.from_numpy(dataset), torch.from_numpy(labels)


if __name__ == '__main__':
    # Use GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))

    batch_size = 64
    num_frames = 10

    # Gotta read that fuckin data in bro
    # also maybe add a bit of noise to the images??
    full_data, labels = load_data(num_frames)

    # Instantiate model
    model = EFBNet(num_frames).to(device) # Create the network and send it to the GPU
    L2_dist = nn.MSELoss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

    for epoch in range(500):
        # Minibatch
        permutation = torch.randperm(full_data.size()[0])

        for i in range(0, full_data.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = full_data[indices], labels[indices]

            # Predict
            y_pred = model(batch_x)

            # Find loss
            loss = L2_dist(y_pred, batch_y)
            print(f'Epoch {epoch}. Loss = {loss.item()}')

            # Backpropagate thru some syntactic magic
            optimizer.zero_grad() # Zero optimizer's gradients
            loss.backward() # Backpropagate loss
            optimizer.step()
