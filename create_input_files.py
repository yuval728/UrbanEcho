import utils as ut
import argparse
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser(description='Create input files')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the csv file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the data directory')
    parser.add_argument('--input_data_dir', type=str, required=True, help='Path to the input data directory')
    parser.add_argument('--n_mfcc', type=int, default=50, help='Number of MFCC features')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size')
    parser.add_argument('--val_size', type=float, default=0.2, help='Validation size')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
     
    return parser.parse_args()
    

if __name__ == "__main__":
    args = parse_args()
    
    df = pd.read_csv(args.csv_file)
    train_df, test_df = train_test_split(df, test_size=args.test_size, random_state=args.seed)
    train_df, val_df = train_test_split(train_df, test_size=args.val_size, random_state=args.seed)
    
    assert train_df['class'].nunique()==test_df['class'].nunique()==val_df['class'].nunique(), 'Classes are not the same in train, test and validation sets'
    
    ut.create_class_folders(base_dir=os.path.join(args.data_dir,'train'), classes=train_df['class'].unique())
    ut.create_class_folders(base_dir=os.path.join(args.data_dir,'val'), classes=val_df['class'].unique(), save_classes=False)
    ut.create_class_folders(base_dir=os.path.join(args.data_dir,'test'), classes=test_df['class'].unique(), save_classes=False)
    
    ut.save_features(args.input_data_dir, train_df, os.path.join(args.data_dir,'train'), n_mfcc=args.n_mfcc)
    
    ut.save_features(args.input_data_dir, val_df, os.path.join(args.data_dir,'val'), n_mfcc=args.n_mfcc)
    
    ut.save_features(args.input_data_dir, test_df, os.path.join(args.data_dir,'test'), n_mfcc=args.n_mfcc)
    
    print('Data split and features saved to', args.data_dir)