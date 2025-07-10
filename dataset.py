"""dataset
"""

import os
import torch
from torch.utils.data import Dataset
import cv2
import random

from torchvision import transforms

import numpy as np

class EveLidSteeringAngleDataset(Dataset):
    
    def __init__(self, root_dir, transform=None, clip_length=2, train=True, flip_prob=0.3, mean=177.243, std=17.290):
    
        self.root_dir = root_dir	# List of sequence directories
        self.transform = transform
        
        self.clip_length = clip_length	# Number of consecutive frames to use as input
        self.train = train
        
        self.flip_prob = flip_prob
        
        
        self.mean = mean / 255.0	## mean=[177.20475391290034, 177.33883136399982]
        self.std = std / 255.0	## std=[17.36767217482142, 17.095717266294667]
        
        # Load data: paths to images and corresponding angles, Prepare lists to hold all data from multiple folders
        self.event_frame_paths = []	# List to store paths
        
        self.lidar_depth_paths = []
        self.steering_angles = []
        
        # Loop through directories and load data, Iterate over each subdirectory in the root directory
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                event_dir = os.path.join(subdir_path, "event_frame")
                
                lidar_dir = os.path.join(subdir_path, "lidar_depth")
                angles_file = os.path.join(subdir_path, "steering_angles.txt")
                
                # Load the steering angles for this sequence
                with open(angles_file, 'r') as f:
                    angles = [float(line.strip().split(" ")[-1]) for line in f]
                    
                
                num_frames = len(angles)	# verify the total number of frames
                
                event_files = sorted(os.listdir(event_dir))
                lidar_files = sorted(os.listdir(lidar_dir))
                
                # Store the paths to the images and the steering angles
                for i in range(num_frames):
                    event_path = os.path.join(event_dir, event_files[i])
                    lidar_path = os.path.join(lidar_dir, lidar_files[i])
                    
                    self.event_frame_paths.append(event_path)
                    
                    self.lidar_depth_paths.append(lidar_path)
                    self.steering_angles.append(angles[i])
        
        # Generate list of starting indices for each clip
        self.clip_indices = list(range(len(self.steering_angles) - (self.clip_length - 1)))
        
        if self.train:
            # Shuffle the clip indices if in training mode
            random.shuffle(self.clip_indices)
    
    def __len__(self):
        '''
        # Length is reduced by (clip_length - 1) to avoid out-of-bound errors for training mode
        if self.train:
            return len(self.steering_angles) - (self.clip_length - 1)
        else
            return len(self.steering_angles)
        '''
        return len(self.clip_indices)
    
    def __getitem__(self, idx):
        if self.train:
            start_idx = self.clip_indices[idx]
            
            # Prepare lists to hold the frames
            
            lidar_depth_frames = []
            
            # Load consecutive frames using cv2
            
            event_frame = cv2.imread(self.event_frame_paths[start_idx + self.clip_length - 1])
            
            lidar_depth_frames = [cv2.imread(self.lidar_depth_paths[start_idx + i], cv2.IMREAD_GRAYSCALE) for i in range(self.clip_length)]
            
            
            # get frame's steering angle
            
            steering_angle = self.steering_angles[start_idx + self.clip_length - 1]
            
            # Random horizontal flip, consistent across the entire clip
            if random.random() < self.flip_prob:
                
                event_frame = cv2.flip(event_frame, 1)	# Flip horizontally
                
                lidar_depth_frames = [cv2.flip(frame, 1) for frame in lidar_depth_frames]	# Flip horizontally
                steering_angle = - steering_angle	# Reverse the steering angle when images are flipped
            
            # Convert images to PyTorch tensors
            
            event_frame = torch.from_numpy(event_frame).permute(2, 0, 1).float() / 255.0
            
            lidar_depth_frames = [torch.from_numpy(img).unsqueeze(0).float() / 255.0 for img in lidar_depth_frames]
            
            if self.transform:
                
                event_frame = self.transform(event_frame)
                
                lidar_depth_frames = [normalize_depth_map(frame, self.mean, self.std) for frame in lidar_depth_frames]
                
            
            lidar_depth_frames = torch.cat(lidar_depth_frames, dim=0)
            
            
            return {'event_frames': event_frame, 'lidar_depth_frames': lidar_depth_frames, 'steering_angle': torch.tensor(steering_angle, dtype=torch.float32)}
        
        else:
            
            start_idx = self.clip_indices[idx]
            
            # Prepare lists to hold the frames
            
            
            lidar_depth_frames = []
            
            # Load consecutive frames using cv2
            
            event_frame = cv2.imread(self.event_frame_paths[start_idx + self.clip_length - 1])
            
            lidar_depth_frames = [cv2.imread(self.lidar_depth_paths[start_idx + i], cv2.IMREAD_GRAYSCALE) for i in range(self.clip_length)]
            
            # get frame's steering angle
            
            steering_angle = self.steering_angles[start_idx + self.clip_length - 1]
            
            # Convert images to PyTorch tensors
            
            event_frame = torch.from_numpy(event_frame).permute(2, 0, 1).float() / 255.0
            
            lidar_depth_frames = [torch.from_numpy(img).unsqueeze(0).float() / 255.0 for img in lidar_depth_frames]
            
            if self.transform:
                
                event_frame = self.transform(event_frame)
                
                lidar_depth_frames = [normalize_depth_map(frame, self.mean, self.std) for frame in lidar_depth_frames]
                
            
            lidar_depth_frames = torch.cat(lidar_depth_frames, dim=0)
            
            
            return {'event_frames': event_frame, 'lidar_depth_frames': lidar_depth_frames, 'steering_angle': torch.tensor(steering_angle, dtype=torch.float32)}
            
    




def normalize_depth_map(depth, mean, std):

    # Mask out the black points, create a mask for valid pixels
    mask = depth > 0	# Assuming black points are 0
    
    
    # Create a copy of the image to avoid modifying the original
    masked_depth = torch.clone(depth)
    
    # Normalize the entire image, but we'll resotre back black points later
    normalized_depth = transforms.Normalize([mean, ], [std, ])(masked_depth)
    # Restore black points, set them back to 0
    normalized_depth[~mask] = 0
    
    return normalized_depth

