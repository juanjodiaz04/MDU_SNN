import os
import shutil
import random
from pathlib import Path

def split_dataset(dataset_dir, output_dir, split_ratio=0.8, seed=42):
    """
    Splits a dataset into train and test folders.
    
    Args:
        dataset_dir (str or Path): Path to the original dataset directory.
        output_dir (str or Path): Path where the train and test directories will be created.
        split_ratio (float): Ratio of files to put in the training set (default: 0.8).
        seed (int): Random seed for reproducibility.
    """
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    
    train_dir = output_dir / 'train'
    test_dir = output_dir / 'test'
    
    # Create output directories if they don't exist
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    random.seed(seed)
    
    # Iterate over each species folder
    for species_dir in dataset_dir.iterdir():
        if species_dir.is_dir():
            species_name = species_dir.name
            
            # Create species folders in train and test directories
            (train_dir / species_name).mkdir(exist_ok=True)
            (test_dir / species_name).mkdir(exist_ok=True)
            
            # Get all files in the current species folder
            files = [f for f in species_dir.iterdir() if f.is_file()]
            
            # Shuffle files randomly
            random.shuffle(files)
            
            # Calculate split index
            split_idx = int(len(files) * split_ratio)
            
            # Split files
            train_files = files[:split_idx]
            test_files = files[split_idx:]
            
            # Copy files to their respective train/test folders
            for f in train_files:
                shutil.copy2(f, train_dir / species_name / f.name)
                
            for f in test_files:
                shutil.copy2(f, test_dir / species_name / f.name)
                
            print(f"Species '{species_name}': {len(train_files)} to train, {len(test_files)} to test.")

if __name__ == "__main__":
    
    
    DATASET_PATH = "../../Birds_Balanced" 
    OUTPUT_PATH = "../../Birds_Split"
    
    split_dataset(DATASET_PATH, OUTPUT_PATH, split_ratio=0.8, seed=42)
