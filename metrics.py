import numpy as np
import argparse
import os

# Define functions for calculating metrics
def calculate_rmse(gt, pred):
    return np.sqrt(np.mean((np.array(gt) - np.array(pred)) ** 2))

def calculate_mae(gt, pred):
    return np.mean(np.abs(np.array(gt) - np.array(pred)))

def calculate_eva(gt, pred):
    residual_variance = np.var(np.array(gt) - np.array(pred))
    gt_variance = np.var(gt)
    return 1 - (residual_variance / gt_variance)

# Read data from the file
def read_file(file_path):
    gt = []
    pred = []
    with open(file_path, 'r') as file:
        for line in file:
            gt_value, prediction = map(float, line.split(','))
            gt.append(gt_value)
            pred.append(prediction)
    return gt, pred

def main():
    parser = argparse.ArgumentParser(description="Calculate Metrics: RMSE, MAE, EVA.")
    parser.add_argument("file_name", help="Input .txt file.")
    
    args = parser.parse_args()
    
    print(f"Calculate Metrics: RMSE, MAE, EVA; from {args.file_name}")
    
    file_path = os.path.join("./outputs", args.file_name)
    
    # Main fuction to compute and display the results
    gt, pred = read_file(file_path)
    
    rmse = calculate_rmse(gt, pred)
    mae = calculate_mae(gt, pred)
    eva = calculate_eva(gt, pred)
    
    print(f"RMSE: {rmse}; MAE: {mae}; EVA: {eva}")
    
if __name__ == '__main__':
    main()

