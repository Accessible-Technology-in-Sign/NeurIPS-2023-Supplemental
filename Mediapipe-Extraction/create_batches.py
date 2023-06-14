import argparse
import os
import glob
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_batches', required=True, type=int)
    parser.add_argument('--use_hands', action='store_true')
    
    args = parser.parse_args()
    return args

def create_batches(num_batches, use_hands, raw_video_location='/data/sign_language_videos', batch_output_location='/data/sign_language_videos/batches'):
    if not os.path.exists(batch_output_location):
        os.makedirs(batch_output_location)
    all_videos = glob.glob(f'{raw_video_location}/**/*.mp4', recursive=True)
    all_videos += glob.glob(f'{raw_video_location}/**/*.mov', recursive=True)
    all_videos += glob.glob(f'{raw_video_location}/**/*.mkv', recursive=True)
    batches = np.array_split(np.array(all_videos), num_batches)
    for i, batch in enumerate(batches):
        if use_hands:
            filename = f'{batch_output_location}/batch_{i+1}_hands.txt'
        else:
            filename = f'{batch_output_location}/batch_{i+1}.txt'
        with open(filename, 'w') as f:
            for video in batch: 
                f.write(f'{video}\n')

if __name__ == "__main__":
    args = parse_args()
    print("Args: ", args)
    
    print(f"Creating {args.num_batches} batches...")
    create_batches(args.num_batches, use_hands=args.use_hands)
    print(f"Finished creating batches")