import torch, os, glob
from torch import nn
import torch.nn.utils
import scipy.io
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision

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
        self.conv_with_fusion =    nn.Conv3d(3, 96, (num_frames_fused, 11, 11), stride=(1, 3, 3), padding=(0, 5, 5)) # Convolves along u and v
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d((2, 2)) # Max pools spatial dims only
        self.conv2 = nn.Conv2d(96, 256, (5, 5), stride=1, padding=2)
        self.conv3 = nn.Conv2d(256, 384, (3, 3), stride=1, padding=1)
        self.conv4 = nn.Conv2d(384, 384, (3, 3), stride=1, padding=1)
        self.conv5 = nn.Conv2d(384, 256, (3, 3), stride=1, padding=1)
        self.fc1 = nn.Linear(256 * 7 * 7 + 1, 4096)
        self.fc2 = nn.Linear(4096, 6) # ig we have it output the x,y,z coordinates of where the ball will land and the x,y,z velocity

    def forward(self, x, scales):
        """ x should be a tensor of shape (N, C, T, H, W)
                N = batch size
                C = color channel count = 3
                T = num frames to merge
                H = frame height = 170
                W = frame width = 170
            
            scales should be an array of length N
        """
        x = self.conv_with_fusion(x)
        x = torch.squeeze(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.max_pool(x)
        x = torch.flatten(x, start_dim=1)

        # Add in scale factor before FC layers
        x = torch.cat([x, scales], 1)

        x = self.fc1(x)
        pred = self.fc2(x)

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
                frames[j,:,:,c] += np.random.normal(scale=2/255) * 255 / 2 # Adds a bit of noise to the image
                np.clip(frames[j,:,:,c], 0, 255, out=frames[j,:,:,c])
                channel_mean = np.mean(frames[j,:,:,c])
                frames[j,:,:,c] -= channel_mean
        pos_label = sample['position'][0]
        vel_label = sample['velocity'][0]
        scale = sample['scale'][0]
        label = np.zeros(6)
        label[:3] = pos_label[:]
        label[3:6] = vel_label[:]

        # Rearrange axes of frames to match dataset
        data_object = torch.from_numpy(np.transpose(frames, axes=[3, 0, 1, 2]))

        full_sample = {'tensor': data_object, 'scale': scale, 'label': torch.from_numpy(label)}

        if self.transform:
            full_sample = self.transform(full_sample)

        return full_sample


if __name__ == '__main__':
    model_save_path = 'most_recent_model.pt'
    loss_save_path = 'epoch_loss.csv'
    batch_loss_path = 'batch_loss.csv'

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
    model = None
    if os.path.exists(model_save_path):
        print("Resuming training from most recent model...")
        model = torch.load(model_save_path)
    else:
        model = EFBNet(num_frames).to(device) # Create the network and send it to the GPU
    
    # Define loss, optimization technique
    L2_dist = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

    for epoch in range(1000000):
        net_loss = 0
        print(f'Epoch {epoch}')
        total = len(dataset)
        for i in range(len(dataset)):
            batch = next(iter(dataset))
            batch_x = batch['tensor'].float().to(device)
            batch_scales = batch['scale'].float().to(device)
            batch_y = batch['label'].float().to(device)

            # Predict
            y_pred = model(batch_x, batch_scales)

            # Find loss
            loss = L2_dist(y_pred, batch_y)
            net_loss += loss.item()
            print(f'    Loss of batch {i}/{total} = {loss.item()}')

            # Backpropagate thru some syntactic magic
            optimizer.zero_grad() # Zero optimizer's gradients
            nn.utils.clip_grad_value_(model.parameters(), 1.0)
            loss.backward() # Backpropagate loss
            optimizer.step()

            # Save batch loss
            batch_loss_file = open(batch_loss_path, 'a')
            batch_loss_file.write(f'{loss}\n')
            batch_loss_file.close()
            
            if (i + 1) % 100 == 0:
                torch.save(model, model_save_path)
        
        # Save things
        torch.save(model, model_save_path)
        loss_file = open(loss_save_path, 'a')
        loss_file.write(f'{net_loss}\n')
        loss_file.close()

        print(f'Epoch {epoch}. Loss = {net_loss}')