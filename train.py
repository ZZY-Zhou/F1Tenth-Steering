"""training loop
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model import EveLidNet
from dataset import EveLidSteeringAngleDataset
import argparse
import os

import logging

import random
import numpy as np

def validate_save(model, criterion, val_dataloader, epoch, experiment_dir):
    model.eval()
    
    total_loss = 0.0
    
    output_file_dir = os.path.join(experiment_dir, 'output')
    os.makedirs(output_file_dir, exist_ok=True)
    output_file_path = os.path.join(output_file_dir, f'{epoch}_output.txt')
    
    
    with torch.no_grad(), open(output_file_path, 'w') as f:
        
        for batch in val_dataloader:
            event_frames = batch.get('event_frames', None)
            
            lidar_depth_frames = batch.get('lidar_depth_frames', None)
            steering_angle = batch.get('steering_angle', None)
            
            if event_frames is None:
                print("event_frames are missing in the batch.")
            
            if lidar_depth_frames is None:
                print("lidar_depth_frames are missing in the batch.")
            if steering_angle is None:
                print("steering_angle are missing in the batch.")
            
            # Transfer data to GPU
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            event_frames = event_frames.to(device)
            
            lidar_depth_frames = lidar_depth_frames.to(device)
            steering_angle = steering_angle.to(device)
            
            
            ### KL Loss ###
            output, _, _ = model(event_frames, lidar_depth_frames)
            
            
            steering_angle = steering_angle.unsqueeze(1)
            
            
            loss = criterion(output, steering_angle)
            
            total_loss += loss.item()	# Accumulate MSE loss
            
            
            # Iterate over batch and save ground truth and predictions to file
            for gt_angle, pred_angle in zip(steering_angle, output):
                # Conver to scalar if necessary, and write to file
                gt_angle_scalar = gt_angle.item()
                pred_angle_scalar = pred_angle.item()
                f.write(f"{gt_angle_scalar:.6f}, {pred_angle_scalar:.6f}\n")
    
    avg_mse_loss = total_loss / len(val_dataloader)	# Average MSE over the validation set
    rmse_loss = avg_mse_loss ** 0.5	# Convert MSE to RMSE
    return rmse_loss
    

def train(args):
    # Create the experiment directory
    experiment_dir = os.path.join(args.save_dir, args.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Log file
    log_file_path = os.path.join(experiment_dir, 'training_log.txt')
    
    logging.basicConfig(filename=log_file_path, level=logging.INFO)
    
    # Path to the root directories containing subfolders for training and validation
    train_root_dir = args.train_data
    val_root_dir = args.test_data
    
    
    # Define any transformations if necessary
    transform = transforms.Compose([
        
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize datasets and dataloaders
    
    train_dataset = EveLidSteeringAngleDataset(train_root_dir, transform=transform, clip_length=args.clip_length, train=True, flip_prob=args.flip_prob)
    
    val_dataset = EveLidSteeringAngleDataset(val_root_dir, transform=transform, clip_length=args.clip_length, train=False, flip_prob=0.0)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Initialize model, loss function, and optimizer
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EveLidNet().to(device)
    criterion = nn.MSELoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    
    ### Learning Rate Scheduler ###
    
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=0)
    
    # Initialize the previous learning rate variable
    prev_lr = optimizer.param_groups[0]['lr']
    
    
    # Initialize tracking variables
    best_val_loss = float('inf')
    best_eopch = -1
    
    # Training loop
    
    for epoch in range(1, args.epochs + 1):
    
        # Set model to training mode
        model.train()
        
        running_loss = 0.0
        
        logging.info(f"Epoch [{epoch}/{args.epochs}]")
        
        
        for batch in train_dataloader:
            
            event_frames = batch.get('event_frames', None)
            
            lidar_depth_frames = batch.get('lidar_depth_frames', None)
            steering_angle = batch.get('steering_angle', None)
            
            if event_frames is None:
                print("event_frames are missing in the batch.")
            
            if lidar_depth_frames is None:
                print("lidar_depth_frames are missing in the batch.")## else:
            
            if steering_angle is None:
                print("steering_angle are missing in the batch.")
            
            
            # Transfer data to GPU
            event_frames = event_frames.to(device)
            
            lidar_depth_frames = lidar_depth_frames.to(device)
            steering_angle = steering_angle.to(device)
            
            # Reset gradients
            optimizer.zero_grad()
            
            # Forward pass
            
            ### KL Loss ###
            output, kl_loss_event, kl_loss_lidar = model(event_frames, lidar_depth_frames)
            
            
            steering_angle = steering_angle.unsqueeze(1)
            
            # Compute loss
            loss = criterion(output, steering_angle)
            
            
            ### KL Loss ###
            
            total_loss = loss + 0.25 * kl_loss_event + 0.25 * kl_loss_lidar
            
            
            total_loss.backward()
            
            
            optimizer.step()
            
            
            ### KL Loss ###
            running_loss += total_loss.item()
            
        
        avg_train_loss = running_loss / len(train_dataloader)
        
        
        avg_val_loss = validate_save(model, criterion, val_dataloader, epoch, experiment_dir)
        
        
        ### Learning Rate Scheduler ###
        scheduler.step()	# Update the learning rate
        
        
        # Get the current learning rate
        current_lr = scheduler.get_last_lr()[0]
        
        # Check if the learning rate has changed
        if current_lr != prev_lr:
            logging.info(f"Epoch [{epoch}/{args.epochs}], Learning Rate changed to = {current_lr:.6f}")
            prev_lr = current_lr	# Update the previous learning rate
        
        
        # Log the losses
        
        logging.info(f"Epoch [{epoch}/{args.epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}\n")
        
        # Check if this is the best model so far based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(experiment_dir, 'checkpoint_best.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model at epoch {epoch} with loss {best_val_loss:.4f}")
        
        # Save the last model checkpoint
        last_model_path = os.path.join(experiment_dir, 'checkpoint_last.pth')
        torch.save(model.state_dict(), last_model_path)
        print(f"Saved last model at epoch {epoch}")
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Train a steering angle prediction model based on event frames and lidar depth frames.')
    
    parser.add_argument('--train_data', type=str, default='../EveLidAngle/training', help='Path to the training data directory')
    parser.add_argument('--test_data', type=str, default='../EveLidAngle/testing', help='Path to the testing data directory')
    
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training and validation')
    
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-2, help='weight decay for the optimizer')
    
    parser.add_argument('--epochs', type=int, default=300, help='Number of training epochs')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--experiment_name', type=str, default='experiment_name', help='Name of the experiment for saving logs and checkpoints')
    parser.add_argument('--save_dir', type=str, default='experiments', help='Directory where experiments will be saved')
    
    
    parser.add_argument('--clip_length', type=int, default=2, help='Number of frames in each sequence')
    parser.add_argument('--flip_prob', type=float, default=0.3, help='Probability of applying horizontazl flip')
    
    args = parser.parse_args()
    
    train(args)

