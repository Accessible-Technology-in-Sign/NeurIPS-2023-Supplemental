import os
import pandas as pd
import numpy as np
from glob import glob
from p_tqdm import p_map
import json
import sys
import itertools
from tqdm import tqdm
import shutil

EBISU_DATA_LOCATION = '.'
PROCESSED_DATA_SAVE_LOCATION = './processed_250_signs_hands'
if os.path.exists(PROCESSED_DATA_SAVE_LOCATION):
    shutil.rmtree(PROCESSED_DATA_SAVE_LOCATION)

def get_processed_right_hand_data(frame, frame_dict):
    right_hand_data = frame_dict['landmarks']['1']
    processed_right_hand_dict = {
        'frame': [],
        'row_id': [],
        'type': [],
        'landmark_index': [],
        'x': [],
        'y': [],
        'z': [],
    }
    for keypoint in range(0, 21):
        processed_right_hand_dict['frame'].append(frame)
        processed_right_hand_dict['row_id'].append(
            f'{frame}-right_hand-{keypoint}'
        )
        processed_right_hand_dict['type'].append('right_hand')
        processed_right_hand_dict['landmark_index'].append(keypoint)
        
        keypoint = str(keypoint)
        if keypoint not in right_hand_data:
            processed_right_hand_dict['x'].append(np.nan)
            processed_right_hand_dict['y'].append(np.nan)
            processed_right_hand_dict['z'].append(np.nan)
        else:
            processed_right_hand_dict['x'].append(right_hand_data[keypoint][0])
            processed_right_hand_dict['y'].append(right_hand_data[keypoint][1])
            processed_right_hand_dict['z'].append(right_hand_data[keypoint][2])
    return processed_right_hand_dict
    
def get_processed_left_hand_data(frame, frame_dict):
    left_hand_data = frame_dict['landmarks']['0']
    processed_left_hand_dict = {
        'frame': [],
        'row_id': [],
        'type': [],
        'landmark_index': [],
        'x': [],
        'y': [],
        'z': [],
    }
    for keypoint in range(0, 21):
        processed_left_hand_dict['frame'].append(frame)
        processed_left_hand_dict['row_id'].append(
            f'{frame}-left_hand-{keypoint}'
        )
        processed_left_hand_dict['type'].append('left_hand')
        processed_left_hand_dict['landmark_index'].append(keypoint)
        
        keypoint = str(keypoint)
        if keypoint not in left_hand_data:
            processed_left_hand_dict['x'].append(np.nan)
            processed_left_hand_dict['y'].append(np.nan)
            processed_left_hand_dict['z'].append(np.nan)
        else:
            processed_left_hand_dict['x'].append(left_hand_data[keypoint][0])
            processed_left_hand_dict['y'].append(left_hand_data[keypoint][1])
            processed_left_hand_dict['z'].append(left_hand_data[keypoint][2])
    return processed_left_hand_dict

def get_frame_dfs(data):
    frame_dfs = []    
    
    # Iterate over each frame and append the resulting processed df into a list
    for frame in data.keys():
        frame_dict = data[frame]
        
        processed_right_hand_dict = get_processed_right_hand_data(frame, frame_dict)
        processed_left_hand_dict = get_processed_left_hand_data(frame, frame_dict)
    
        processed_right_hand_df = pd.DataFrame.from_dict(processed_right_hand_dict)
        processed_left_hand_df = pd.DataFrame.from_dict(processed_left_hand_dict)
        
        # Concatenate the dataframes
        combined_df = pd.concat(
            [processed_right_hand_df, processed_left_hand_df],
            axis=0
        )
        
        frame_dfs.append(combined_df)
        
    return frame_dfs

def process_file(file, save_location):
    try:
        sign = file.split('/')[-1].split('.')[2]
        
        with open(file, 'r') as f:
            # Load in .data file
            data = json.load(f)
            
        # Process the data
        frame_dfs = get_frame_dfs(data)
        
        # Concatenate the dataframes
        combined_df = pd.concat(frame_dfs, axis=0)
        
        # Get name of the file
        filename = file.split('/')[-1]
        
        # Change dtype of landmark_index to int and frame to int and x, y, z to float
        combined_df['frame'] = combined_df['frame'].astype(int)
        combined_df['landmark_index'] = combined_df['landmark_index'].astype(int)
        combined_df['x'] = combined_df['x'].astype(float)
        combined_df['y'] = combined_df['y'].astype(float)
        combined_df['z'] = combined_df['z'].astype(float)
        
        # Change dtype of row_id and type to str
        combined_df['row_id'] = combined_df['row_id'].astype(str)
        combined_df['type'] = combined_df['type'].astype(str)
        
        # Sort the dataframe by frame, type, and landmark_index
        combined_df = combined_df.sort_values(by=['frame', 'type', 'landmark_index'])
        
        # Reset the index
        combined_df = combined_df.reset_index(drop=True)

        # Replace the .data extension with .parquet and save the dataframe by sign into the correct folder
        combined_df.to_parquet(f'{save_location}/{sign}/{filename.replace(".data", ".parquet")}')
    except:
        print(f"{file} resulted in an exception")

# Get the path of each data file recursively in data location
all_data_files = glob(os.path.join(EBISU_DATA_LOCATION, '**', '*.data'), recursive=True)

print(all_data_files[0])

# Create a train, test, validation folder
if os.path.exists(f"{PROCESSED_DATA_SAVE_LOCATION}/train"):
    shutil.rmtree(f"{PROCESSED_DATA_SAVE_LOCATION}/train")
    
if os.path.exists(f"{PROCESSED_DATA_SAVE_LOCATION}/validation"):
    shutil.rmtree(f"{PROCESSED_DATA_SAVE_LOCATION}/validation")

if os.path.exists(f"{PROCESSED_DATA_SAVE_LOCATION}/test"):
    shutil.rmtree(f"{PROCESSED_DATA_SAVE_LOCATION}/test")

unique_signs = set()

for file in all_data_files:
    sign = file.split('/')[-1].split('.')[2]
    unique_signs.add(sign)

for unique_sign in unique_signs:
    os.makedirs(f"{PROCESSED_DATA_SAVE_LOCATION}/train/{unique_sign}")
    os.makedirs(f"{PROCESSED_DATA_SAVE_LOCATION}/validation/{unique_sign}")
    os.makedirs(f"{PROCESSED_DATA_SAVE_LOCATION}/test/{unique_sign}")

# Create an array of PROCESSED_DATA_SAVE_LOCATION/train the length of all_data_files
save_locations = [f"{PROCESSED_DATA_SAVE_LOCATION}/train" for _ in all_data_files]

# Change save_locations if any of these strings are in the file
validation_users = ["4a.7002", "4a.8005", "4a.8013", "4a.8018", "4a.8025", "4a.8047", "4a.8015", "4a.8051"]
for i, file in enumerate(all_data_files):
    if any([user in file for user in validation_users]):
        save_locations[i] = f"{PROCESSED_DATA_SAVE_LOCATION}/validation"

# Change save_locations to test if string has review_4 and not of user 4a.8031 or 4a.8040
test_users = ["4a.8032", "4a.8033", "4a.8031", "4a.8035", "4a.8036", "4a.8037", "4a.8038", "4a.8039"]
for i, file in enumerate(all_data_files):
    if any([user in file for user in test_users]):
        save_locations[i] = f"{PROCESSED_DATA_SAVE_LOCATION}/test"

# If using a windows computer, you need to use tqdm and call process_file on each file
platform = sys.platform
if platform == 'win32':
    for file in tqdm(all_data_files):
        process_file(file)
# We can use p_tqdm otherwise
else:
    p_map(process_file, all_data_files, save_locations, num_cpus=os.cpu_count())
