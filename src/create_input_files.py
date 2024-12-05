from . import utils as ut
import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description='Create input files')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the csv file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--input_data_dir', type=str, required=True, help='Path to the input data directory')
    parser.add_argument('--n_mfcc', type=int, default=50, help='Number of MFCC features')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size')
    parser.add_argument('--val_size', type=float, default=0.2, help='Validation size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
     
    return parser.parse_args()

def create_input_files(csv_file, data_dir, input_data_dir, n_mfcc, test_size, val_size, seed):
    """
    Create input files for the Urban Sound Classification project.

    Args:
        csv_file (str): Path to the CSV file.
        data_dir (str): Path to the data directory.
        input_data_dir (str): Path to the input data directory.
        n_mfcc (int): Number of MFCC features.
        test_size (float): Test size.
        val_size (float): Validation size.
        seed (int): Random seed.
    """
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Split the data into train, test, and validation sets
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed)
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=seed)
    
    # Ensure the classes are the same in train, test, and validation sets
    assert train_df['class'].nunique() == test_df['class'].nunique() == val_df['class'].nunique(), 'Classes are not the same in train, test and validation sets'
    
    # Create class folders for train, validation, and test sets
    ut.create_class_folders(base_dir=os.path.join(data_dir, 'train'), classes=train_df['class'].unique())
    ut.create_class_folders(base_dir=os.path.join(data_dir, 'val'), classes=val_df['class'].unique(), save_classes=False)
    ut.create_class_folders(base_dir=os.path.join(data_dir, 'test'), classes=test_df['class'].unique(), save_classes=False)
    
    # Save features for the train, validation, and test sets
    ut.save_features(input_data_dir, train_df, os.path.join(data_dir, 'train'), n_mfcc=n_mfcc)
    ut.save_features(input_data_dir, val_df, os.path.join(data_dir, 'val'), n_mfcc=n_mfcc)
    ut.save_features(input_data_dir, test_df, os.path.join(data_dir, 'test'), n_mfcc=n_mfcc)

def main():
    """
    Main function to parse arguments and create input files.
    """
    args = parse_args()
    create_input_files(args.csv_file, args.data_dir, args.input_data_dir, args.n_mfcc, args.test_size, args.val_size, args.seed)

if __name__ == "__main__":
    main()