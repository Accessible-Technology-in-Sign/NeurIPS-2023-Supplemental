# %% [markdown]
# # 250 Sign Dataset EDA and Training

# %%
# Imports
SEED = 11

import tensorflow as tf
import os
import numpy as np
import random

def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

# Call the above function with seed value
set_global_determinism(seed=SEED)

import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import plotly.graph_objects as go
import pandas as pd
from glob import glob
from tqdm import tqdm
from p_tqdm import p_map
import re
from functools import partial
from tensorflow.python.client import device_lib
from functools import partial

tf.get_logger().setLevel("ERROR")

# %%
# Constants
EBISU_FILES_DIR = "./processed_250_signs_hands"

# Define train_model model order
MODEL_ORDER = ["lstm"]

# Generate the preprocessing combinations
trial_nos = [0]

# Define the save location
DATA_LOCATION = "./set_250_hands"
SAVE_LOCATIONS = [f"{DATA_LOCATION}/{trial}" for trial in trial_nos]
LOSS_FIG_SAVE_LOCATIONS = [
    f"{SAVE_LOCATIONS[trial]}/loss_graphs" for trial in trial_nos
]
ACCURACY_FIG_SAVE_LOCATIONS = [
    f"{SAVE_LOCATIONS[trial]}/accuracy_graphs" for trial in trial_nos
]
CONFUSION_MATRIX_SAVE_LOCATIONS = [
    f"{SAVE_LOCATIONS[trial]}/confusion_matrices" for trial in trial_nos
]

for (
    loss_fig_save_location,
    accuracy_fig_save_location,
    confusion_matrix_save_location,
) in zip(
    LOSS_FIG_SAVE_LOCATIONS,
    ACCURACY_FIG_SAVE_LOCATIONS,
    CONFUSION_MATRIX_SAVE_LOCATIONS,
):
    if not os.path.exists(loss_fig_save_location):
        os.makedirs(loss_fig_save_location)
        for model in MODEL_ORDER:
            os.makedirs(f"{loss_fig_save_location}/{model}")
    if not os.path.exists(accuracy_fig_save_location):
        os.makedirs(accuracy_fig_save_location)
        for model in MODEL_ORDER:
            os.makedirs(f"{accuracy_fig_save_location}/{model}")
    if not os.path.exists(confusion_matrix_save_location):
        os.makedirs(confusion_matrix_save_location)
        for model in MODEL_ORDER:
            os.makedirs(f"{confusion_matrix_save_location}/{model}")

# %%
# Check the number of unique signs via glob
train_files = glob(f"{EBISU_FILES_DIR}/train/*/*.parquet", recursive=True)
test_files = glob(f"{EBISU_FILES_DIR}/test/*/*.parquet", recursive=True)
val_files = glob(f"{EBISU_FILES_DIR}/validation/*/*.parquet", recursive=True)

print(f"Number of files in train: {len(train_files)}")
print(f"Number of files in test: {len(test_files)}")
print(f"Number of files in val: {len(val_files)}")

train_labels = [file.split('/')[3] for file in train_files]
test_labels = [file.split('/')[3] for file in test_files]
val_labels = [file.split('/')[3] for file in val_files]

train_unique_signs = np.unique(train_labels)
test_unique_signs = np.unique(test_labels)
val_unique_signs = np.unique(val_labels)

print(f"Number of unique signs in train: {len(train_unique_signs)}")
print(f"Number of unique signs in test: {len(test_unique_signs)}")
print(f"Number of unique signs in val: {len(val_unique_signs)}")

# %% [markdown]
# ## Data Extraction
# 

# %%
# helper function to extract relevant frames (for wrist landmarks) from the parquet file
def get_features(parquet_file, return_columns=False):
    # Change dtypes
    parquet_file["frame"] = parquet_file["frame"].astype(int)
    parquet_file["landmark_index"] = parquet_file["landmark_index"].astype(int)
    parquet_file["x"] = parquet_file["x"].astype(float)
    parquet_file["y"] = parquet_file["y"].astype(float)
    parquet_file["z"] = parquet_file["z"].astype(float)
    parquet_file["row_id"] = parquet_file["row_id"].astype(str)
    parquet_file["type"] = parquet_file["type"].astype(str)

    # Get all the hand and pose landmarks for all frames
    right_hand = parquet_file[parquet_file["type"].str.contains("right_hand")]
    left_hand = parquet_file[parquet_file["type"].str.contains("left_hand")]
    pose = parquet_file[parquet_file["type"].str.contains("pose")]
    face = parquet_file[parquet_file["type"].str.contains("face")]

    # Get a list of face landmarks by landmark index
    face_landmarks = []
    for i in face["landmark_index"].unique():
        curr_face_landmark = face[face["landmark_index"] == i].copy()
        curr_face_landmark.rename(
            columns={
                "x": "x_face_" + str(i),
                "y": "y_face_" + str(i),
                "z": "z_face_" + str(i),
            },
            inplace=True,
        )
        curr_face_landmark.drop(
            ["row_id", "type", "landmark_index"], axis=1, inplace=True
        )
        curr_face_landmark.reset_index(drop=True, inplace=True)
        curr_face_landmark.set_index("frame", inplace=True)
        face_landmarks.append(curr_face_landmark)

    # Merge all of the face landmarks into one dataframe based on frame
    merged_face = pd.concat(face_landmarks, axis=1) if len(face_landmarks) > 0 else None

    # Get a list of all the right hand landmarks by landmark index
    right_hand_landmarks = []
    for i in right_hand["landmark_index"].unique():
        curr_hand_landmark = right_hand[right_hand["landmark_index"] == i].copy()
        curr_hand_landmark.rename(
            columns={
                "x": "x_right_hand_" + str(i),
                "y": "y_right_hand_" + str(i),
                "z": "z_right_hand_" + str(i),
            },
            inplace=True,
        )
        curr_hand_landmark.drop(
            ["row_id", "type", "landmark_index"], axis=1, inplace=True
        )
        curr_hand_landmark.reset_index(drop=True, inplace=True)
        curr_hand_landmark.set_index("frame", inplace=True)
        right_hand_landmarks.append(curr_hand_landmark)

    # Merge all of the right hand landmarks into one dataframe based on frame
    merged_right_hand = pd.concat(right_hand_landmarks, axis=1)

    # Get a list of all the left hand landmarks by landmark index
    left_hand_landmarks = []
    for i in left_hand["landmark_index"].unique():
        curr_hand_landmark = left_hand[left_hand["landmark_index"] == i].copy()
        curr_hand_landmark.rename(
            columns={
                "x": "x_left_hand_" + str(i),
                "y": "y_left_hand_" + str(i),
                "z": "z_left_hand_" + str(i),
            },
            inplace=True,
        )
        curr_hand_landmark.drop(
            ["row_id", "type", "landmark_index"], axis=1, inplace=True
        )
        curr_hand_landmark.reset_index(drop=True, inplace=True)
        curr_hand_landmark.set_index("frame", inplace=True)
        left_hand_landmarks.append(curr_hand_landmark)

    # Merge all of the left hand landmarks into one dataframe based on frame
    merged_left_hand = pd.concat(left_hand_landmarks, axis=1)

    # Get a list of all the pose landmarks by landmark index
    pose_landmarks = []
    for i in pose["landmark_index"].unique():
        curr_pose_landmark = pose[pose["landmark_index"] == i].copy()
        curr_pose_landmark.rename(
            columns={
                "x": "x_pose_" + str(i),
                "y": "y_pose_" + str(i),
                "z": "z_pose_" + str(i),
            },
            inplace=True,
        )
        curr_pose_landmark.drop(
            ["row_id", "type", "landmark_index"], axis=1, inplace=True
        )
        curr_pose_landmark.reset_index(drop=True, inplace=True)
        curr_pose_landmark.set_index("frame", inplace=True)
        pose_landmarks.append(curr_pose_landmark)

    # Merge all of the pose landmarks into one dataframe based on frame
    merged_pose = pd.concat(pose_landmarks, axis=1) if len(pose_landmarks) > 0 else None

    # Check which hand has more NaN values
    count_left_hand_nans = merged_left_hand.isna().sum().sum()
    count_right_hand_nans = merged_right_hand.isna().sum().sum()

    # If the left hand has more NaNs, then that means the left hand is not visible
    if count_left_hand_nans >= count_right_hand_nans:
        # Rename the right hand columns to just say "hand" instead of "right_hand"
        merged_right_hand.columns = merged_right_hand.columns.str.replace("_right", "")
        if merged_face is None or merged_pose is None:
            return merged_right_hand

        merged_pose_hand_face = pd.concat(
            [merged_face, merged_pose, merged_right_hand], axis=1
        )
        if return_columns:
            return merged_pose_hand_face, merged_pose_hand_face.columns
        return merged_pose_hand_face

    else:
        # Flip the left hand coordinates along the vertical axis
        middle_axis = 0.5
        x_columns_left = [col for col in merged_left_hand.columns if "x" in col]
        for col in x_columns_left:
            merged_left_hand[col] = middle_axis - (merged_left_hand[col] - middle_axis)

        # Rename the left hand columns to just say "hand" instead of "left_hand"
        merged_left_hand.columns = merged_left_hand.columns.str.replace("_left", "")

        if merged_face is None or merged_pose is None:
            return merged_left_hand

        merged_pose_hand_face = pd.concat(
            [merged_face, merged_pose, merged_left_hand], axis=1
        )

        if return_columns:
            return merged_pose_hand_face, merged_pose_hand_face.columns

        return merged_pose_hand_face

# %%
def extract_features_ebisu(filename, sign, files_dir, save_location, user):
    file_save_location = filename.replace(files_dir, f"{save_location}/{user}_files")

    # Identify whether train, validation, or test is in file_save_location
    is_train = "train" in file_save_location
    is_validation = "validation" in file_save_location
    is_test = "test" in file_save_location
    
    curr_group_assignment = None
    if is_train:
        curr_group_assignment = "train"
    elif is_validation:
        curr_group_assignment = "validation"
    elif is_test:
        curr_group_assignment = "test"

    # Remove "/validation" or "/train" or "/test"
    file_save_location = file_save_location.replace("/validation", "")
    file_save_location = file_save_location.replace("/train", "")
    file_save_location = file_save_location.replace("/test", "")
    
    # If the extracted feature exists, then no need to re-process it again
    if os.path.exists(file_save_location.replace(".parquet", ".pkl")):
        file_save_location = file_save_location.replace(".parquet", ".pkl")
        time_series_coordinates = pd.read_pickle(file_save_location)
        return file_save_location, sign, time_series_coordinates, curr_group_assignment

    # Only add parquet file if features are able to be extracted
    parquet_file = pd.read_parquet(filename)
    time_series_coordinates = get_features(parquet_file)
    if type(time_series_coordinates) != type(None):
        file_save_location = file_save_location.replace(".parquet", ".pkl")
        time_series_coordinates.to_pickle(file_save_location)
        return file_save_location, sign, time_series_coordinates, curr_group_assignment

# %%
files_dir = EBISU_FILES_DIR
save_location = DATA_LOCATION
user = "ebisu"

# %%
# Extract features from Ebisu Data
ebisu_file_list = glob(f"{files_dir}/**/*.parquet", recursive=True)

# Remove any files from ebisu_file_list that are 4a.8031 and 4a.8040
# ebisu_file_list = [file for file in ebisu_file_list if "4a.8031" not in file and "4a.8040" not in file]

all_signs = [file.split('/')[-1].split('.')[2] for file in ebisu_file_list]
unique_signs = set(np.unique(all_signs).tolist())

# print("Train_Test_Validation")
# print(train_test_validation)

# Make directories for signs
for sign in unique_signs:
    file_save_location = f"{save_location}/{user}_files/{sign}"
    if not os.path.exists(file_save_location): 
        os.makedirs(file_save_location)

ebisu_new_filenames, ebisu_signs, ebisu_pkl_files, ebisu_train_test_validation = zip(
    *p_map(
        partial(
            extract_features_ebisu,
            files_dir=files_dir,
            save_location=save_location,
            user=user,
        ),
        ebisu_file_list,
        all_signs,
        num_cpus=os.cpu_count(),
        desc="Extracting Ebisu Features",
    )
)

ebisu_filename_label_dict = {k: v for k, v in zip(ebisu_new_filenames, ebisu_signs)}

ebisu_labels = np.array(ebisu_signs)

# Create list of users for each file
ebisu_users = []
user_regex = r"4a\.([0-9]*?)\."
for file in ebisu_new_filenames:
    user = re.search(user_regex, file).group(1)
    ebisu_users.append(user)

pkl_files = ebisu_pkl_files
labels = ebisu_labels
users = ebisu_users
train_test_validation = ebisu_train_test_validation

# %% [markdown]
# ### Examine Class Imbalance between Train, Test, Validation before preprocessing

# %%
print(f"Length of unique signs: {len(set(unique_signs))}")

# %%
# Calculate the number of unique labels in train, test, validation
train_labels = labels[np.array(train_test_validation) == "train"]
test_labels = labels[np.array(train_test_validation) == "test"]
validation_labels = labels[np.array(train_test_validation) == "validation"]
train_unique_labels = set(train_labels)
test_unique_labels = set(test_labels)
validation_unique_labels = set(validation_labels)

print(f"Number of unique train labels: {len(train_unique_labels)}")
print(f"Number of unique test labels: {len(test_unique_labels)}")
print(f"Number of unique validation labels: {len(validation_unique_labels)}")

# Confirm that when you add the sets, you get the total number of unique labels
total_unique_labels = train_unique_labels | test_unique_labels | validation_unique_labels
print(f"Number of total unique labels: {len(total_unique_labels)}")

# Print the size of each dataset
print(f"Number of train samples: {len(train_labels)}")
print(f"Number of test samples: {len(test_labels)}")
print(f"Number of validation samples: {len(validation_labels)}")

# %%
# Find how many signs overlap from the validation and train set
validation_overlap = set(train_labels) & set(validation_labels)
test_overlap = set(train_labels) & set(test_labels)
print(f"Number of validation overlap: {len(validation_overlap)}")
print(f"Number of test overlap: {len(test_overlap)}")

# %%
# Create a dictionary of unique labels to their count for train, test, validation
train_label_count_dict = {label: 0 for label in train_unique_labels}
test_label_count_dict = {label: 0 for label in test_unique_labels}
validation_label_count_dict = {label: 0 for label in validation_unique_labels}
for label in train_labels:
    train_label_count_dict[label] += 1
for label in test_labels:
    test_label_count_dict[label] += 1
for label in validation_labels:
    validation_label_count_dict[label] += 1

# %%
# Create a Plotly Bar Graph of the label counts for train
fig = go.Figure(
    data=[
        go.Bar(
            x=list(train_label_count_dict.keys()),
            y=list(train_label_count_dict.values()),
        )
    ]
)
fig.update_layout(
    title="Train Label Counts",
    xaxis_title="Labels",
    yaxis_title="Count",
    font=dict(size=18),
)

# fig.show()

# %%
# Create a Plotly Bar Graph of the label counts for validation
fig = go.Figure(
    data=[
        go.Bar(
            x=list(validation_label_count_dict.keys()),
            y=list(validation_label_count_dict.values()),
        )
    ]
)
fig.update_layout(
    title="Validation Label Counts",
    xaxis_title="Labels",
    yaxis_title="Count",
    font=dict(size=18),
)
# fig.show()


# %%
# Create a Plotly Bar Graph of the label counts for test
fig = go.Figure(
    data=[
        go.Bar(
            x=list(test_label_count_dict.keys()),
            y=list(test_label_count_dict.values()),
        )
    ]
)
fig.update_layout(
    title="Test Label Counts",
    xaxis_title="Labels",
    yaxis_title="Count",
    font=dict(size=18),
)
# fig.show()

# %%
# Create a Plotly Violin Graph of the number of frames in each video in the dataset
num_frames_per_video = []
for file in pkl_files:
    num_frames_per_video.append(len(file))
num_frames_per_video = np.array(num_frames_per_video)

# Plot violin plot figure showing distribution of number of frames
frames_distribution = go.Figure(
    data=go.Violin(
        y=num_frames_per_video,
        name=""
    )
)
frames_distribution.update_layout(
    title="Distribution of Number of Frames",
    xaxis_title="Videos of all signs",
    yaxis_title="Number of Frames",
)
frames_distribution.show()

# %%
# Create a Plotly Violin Graph of the number of frames in each video for if it's in train/val/test
num_frames_per_video_train = []
num_frames_per_video_test = []
num_frames_per_video_validation = []
for file, group in zip(pkl_files, train_test_validation):
    if group == "train":
        num_frames_per_video_train.append(len(file))
    elif group == "test":
        num_frames_per_video_test.append(len(file))
    elif group == "validation":
        num_frames_per_video_validation.append(len(file))

# Plot violin plot figure showing distribution of number of frames
frames_train_test_val_distribution = go.Figure(
    data=[
        go.Violin(
            y=num_frames_per_video_train,
            name="Train"
        ),
        go.Violin(
            y=num_frames_per_video_test,
            name="Test"
        ),
        go.Violin(
            y=num_frames_per_video_validation,
            name="Validation"
        )
    ]
)

frames_train_test_val_distribution.update_layout(
    title="Distribution of Number of Frames",
    xaxis_title="Videos of all signs",
    yaxis_title="Number of Frames",
)

frames_train_test_val_distribution.show()

# %% [markdown]
# ## Preprocessing

# %%
# Feature Manipulation - Skip for Hands
print("Doing feature manipulation")
pkl_files = pkl_files

# Feature Dropping - Skip for Hands
print("Doing feature dropping")
pkl_files = pkl_files

# Z-Value Dropping - Drop all Z-Values
pkl_files = [df.drop(columns=[col for col in df.columns if "z" in col]) for df in tqdm(pkl_files, desc="Dropping Z-Values")]

# Drop all face keypoints
pkl_files = [df.drop(columns=[col for col in df.columns if 'face' in col]) for df in tqdm(pkl_files, desc="Dropping Face Keypoints")]

# For each pkl file, remove all the rows that have NaN values
pkl_files = [df.dropna() for df in tqdm(pkl_files, desc="Dropping NaN Rows")]

# Remove all pkl files and labels that have 0 frames
indices_to_keep = np.where([len(df) > 0 for df in pkl_files])[0]
pkl_files = [pkl_files[i] for i in indices_to_keep]
labels = np.array(labels)[indices_to_keep].tolist()
users = np.array(users)[indices_to_keep].tolist()
train_test_validation = np.array(train_test_validation)[indices_to_keep].tolist()

# %%
max_frames = 60

def left_pad_df_negative_1(df, max_frames):
    num_frames = len(df)
    num_rows_to_add = max_frames - num_frames
    negative_1_arr = np.zeros(df.shape[1]) - 1
    negative_1_df = pd.DataFrame(negative_1_arr.reshape(1, -1), columns=list(df.columns))
    new_df = df.append([negative_1_df] * num_rows_to_add, ignore_index=True)
    return new_df

def left_pad_df(df, max_frames):
    num_frames = len(df)
    num_rows_to_add = max_frames - num_frames
    new_df = df.append([df.iloc[-1]] * num_rows_to_add, ignore_index=True)
    return new_df

pkl_files = [left_pad_df_negative_1(df, max_frames) if len(df) < max_frames else df for df in tqdm(pkl_files, desc="Left Padding with -1")]
# pkl_files = [left_pad_df(df, max_frames) if len(df) < max_frames else df for df in tqdm(pkl_files, desc="Left Padding with Last Frame")]

def take_middle_frames_df(df, max_frames):
    num_frames = len(df)
    start_index = (num_frames - max_frames) // 2
    end_index = start_index + max_frames
    new_df = df[start_index:end_index]
    return new_df

pkl_files = [take_middle_frames_df(df, max_frames) if len(df) > max_frames else df for df in tqdm(pkl_files, desc="Taking Middle Frames")]

# %%
# Convert labels to categorical encoding
print("Converting to categorial encoding")
original_labels = labels.copy()
labels = pd.Categorical(labels)

# Convert labels to one-hot encoding
print("Converting to one-hot encoding")
labels = pd.get_dummies(labels)

# Get sign to one-hot encoding mapping
label_to_one_hot = {}
for i in range(len(original_labels)):
        label_to_one_hot[original_labels[i]] = labels.iloc[i].to_numpy()
        
print(f"Length of pkl_files: {len(pkl_files)}")
print(f"Length of labels: {len(labels)}")
print(f"Length of users: {len(users)}")
print(f"Length of train_test_validation: {len(train_test_validation)}")

# %% [markdown]
# ## Explorative Data Analysis

# %% [markdown]
# ## Model Training

# %%
# Create X_train, X_val, X_test, y_train, y_val, y_test
X_train = [pkl_files[i] for i in range(len(pkl_files)) if train_test_validation[i] == "train"]
X_val = [pkl_files[i] for i in range(len(pkl_files)) if train_test_validation[i] == "validation"]
X_test = [pkl_files[i] for i in range(len(pkl_files)) if train_test_validation[i] == "test"]

y_train = [labels.loc[i] for i in range(len(labels)) if train_test_validation[i] == "train"]
y_val = [labels.loc[i] for i in range(len(labels)) if train_test_validation[i] == "validation"]
y_test = [labels.loc[i] for i in range(len(labels)) if train_test_validation[i] == "test"]

def create_lstm_model(X_train, max_frames, signs):
    model = tf.keras.Sequential()

    model.add(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                128,
                return_sequences=True,
            ),
            input_shape=(max_frames, len(X_train[0].columns)),
        )
    )
    model.add(
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                128,
            )
        )
    )
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(len(list(signs)), activation="softmax"))
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )
    return model

def train_architecture(model, gpu_device, X_train, y_train, X_test=None, y_test=None, epochs_to_use=None, return_model=False):   
    if epochs_to_use is None:
        epochs_to_use = 1
    
    if X_test is not None and y_test is not None:
        with tf.device(gpu_device):
            history = model.fit(
                np.array(X_train),
                np.array(y_train),
                epochs=epochs_to_use,
                batch_size=1024,
                validation_data=(np.array(X_test), np.array(y_test)),
                callbacks=tf.keras.callbacks.EarlyStopping(
                    patience=5,
                    monitor="val_loss",
                    restore_best_weights=True,
                ),
                verbose=1
            )

            if return_model:
                return model, history
            
            return history
    else:
        with tf.device(gpu_device):
            history = model.fit(
                np.array(X_train),
                np.array(y_train),
                epochs=epochs_to_use,
                batch_size=64,
                callbacks=tf.keras.callbacks.EarlyStopping(
                    patience=5,
                    monitor="loss",
                    restore_best_weights=True,
                ),
                verbose=1
            )

            if return_model:
                return model, history
            
            return history

# %%
# tf.keras.backend.clear_session()
# tf.get_logger().setLevel("ERROR")
def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']
gpus = get_available_gpus()

signs = list(label_to_one_hot.keys())

# Create models
lstm_model = create_lstm_model(X_train, max_frames, signs)

# Train models
lstm_model_history = train_architecture(lstm_model, gpus[0], X_train, y_train, X_val, y_val, 40)

# Get the number of epochs from the best model
epochs_used_lstm_model = len(lstm_model_history.history["accuracy"])

models = (lstm_model)
histories = (lstm_model_history)
epochs_used = (epochs_used_lstm_model)


# %%
# Save models
def save_model(save_location, model, folder_name):
    # Save model in SavedModel format inside the model directory
    if not os.path.exists(f"{save_location}/{folder_name}"):
        os.makedirs(f"{save_location}/{folder_name}")

    model.save(f"{save_location}/{folder_name}")

    # Save model in TensorFlow Lite format
    if not os.path.exists(f"{save_location}/{folder_name}/tflite"):
        os.makedirs(f"{save_location}/{folder_name}/tflite")

    converter = tf.lite.TFLiteConverter.from_saved_model(
        f"{save_location}/{folder_name}"
    )
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
    ]
    tflite_model = converter.convert()

    with open(f"{save_location}/{folder_name}/tflite/model.tflite", "wb") as fout:
        fout.write(tflite_model)

if not os.path.exists("models"):
    os.makedirs("models")

save_model("models", lstm_model, "lstm_model")

# %%
def create_loss_accuracy_figure(history):
    # Plot the loss and accuracy curves for training and validation
    loss_train = history.history["loss"]
    loss_val = history.history["val_loss"]

    accuracy_train = history.history["accuracy"]
    accuracy_val = history.history["val_accuracy"]

    loss_fig = go.Figure()
    loss_fig.add_trace(
        go.Scatter(
            x=np.arange(1, len(loss_train) + 1), y=loss_train, name="Training loss"
        )
    )
    loss_fig.add_trace(
        go.Scatter(
            x=np.arange(1, len(loss_val) + 1), y=loss_val, name="Validation loss"
        )
    )
    loss_fig.update_layout(title="Loss", xaxis_title="Epoch", yaxis_title="Loss")

    accuracy_fig = go.Figure()
    accuracy_fig.add_trace(
        go.Scatter(
            x=np.arange(1, len(accuracy_train) + 1),
            y=accuracy_train,
            name="Training accuracy",
        )
    )
    accuracy_fig.add_trace(
        go.Scatter(
            x=np.arange(1, len(accuracy_val) + 1),
            y=accuracy_val,
            name="Validation accuracy",
        )
    )
    accuracy_fig.update_layout(
        title="Accuracy", xaxis_title="Epoch", yaxis_title="Accuracy"
    )

    return loss_fig, accuracy_fig


def create_confusion_matrix(model, X_test, y_test, signs, label_to_one_hot):
    # Display confusion matrix
    predictions = np.argmax(model.predict(np.array(X_test), verbose=0), axis=1)
    ground_truth = np.argmax(np.array(y_test), axis=1)
    confusion_matrix = tf.math.confusion_matrix(
        ground_truth, predictions, num_classes=len(signs)
    )
    confusion_matrix = confusion_matrix.numpy()

    # Normalize confusion matrix
    confusion_matrix = (
        confusion_matrix.astype("float") / confusion_matrix.sum(axis=1)[:, np.newaxis]
    )
    
    # For each label_to_one_hot key, convert the one_hot encoding to a single integer
    # This is so that the confusion matrix can be displayed with the correct labels
    label_to_int = {}
    for key in label_to_one_hot.keys():
        label_to_int[key] = np.argmax(label_to_one_hot[key])
    
    # Sort the label_to_int dictionary by the integer value
    label_to_int = {k: v for k, v in sorted(label_to_int.items(), key=lambda item: item[1])}
    
    # Get the sorted list of signs
    ordered_signs = list(label_to_int.keys())

    # Invert confusion matrix so that diagonal is down and right
    confusion_matrix = confusion_matrix[::-1]

    fig = go.Figure(
        data=go.Heatmap(z=confusion_matrix, x=list(ordered_signs), y=list(ordered_signs)[::-1])
    )
    fig.update_layout(
        title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual"
    )

    return fig, confusion_matrix, ordered_signs

def create_confusion_matrix_from_array(confusion_matrix, signs):
    fig = go.Figure(
        data=go.Heatmap(z=confusion_matrix, x=list(signs), y=list(signs)[::-1])
    )
    fig.update_layout(
        title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual"
    )
    return fig

def calculate_accuracy(model, X_test, y_test):
    # Calculate the accuracy using the predictions and ground truth
    predictions = np.argmax(model.predict(np.array(X_test), verbose=0), axis=1)
    ground_truth = np.argmax(np.array(y_test), axis=1)
    accuracy = np.sum(predictions == ground_truth) / len(predictions)
    return accuracy

def calculate_failed_mediapipe_adjusted_accuracy(model, X_test, y_test):
    # Calculate the accuracy but divide by total videos in the dataset
    predictions = np.argmax(model.predict(np.array(X_test), verbose=0), axis=1)
    ground_truth = np.argmax(np.array(y_test), axis=1)
    accuracy = np.sum(predictions == ground_truth) / 165294
    return accuracy

def calculate_accuracy_including_homosigns(model, X_test, y_test, label_to_one_hot):
    # Calculate the accuracy but count homosigns together
    predictions = np.argmax(model.predict(np.array(X_test), verbose=0), axis=1)
    ground_truth = np.argmax(np.array(y_test), axis=1)

    # Convert from one hot to int
    label_to_int = {}
    for key in label_to_one_hot.keys():
        label_to_int[key] = np.argmax(label_to_one_hot[key])
    int_to_label = {v:k for k, v in label_to_int.items()}

    # Generate known homosigns
    homosigns = {
        'awake': 'wake',
        'glasswindow': 'tooth',
        'beside': 'person',
        'finger': 'wait',
        'chin': 'lips',
        'police': 'hungry',
        'into': 'elephant'
    }

    # Convert predictions from int to label
    predictions = [int_to_label[pred] for pred in predictions]
    
    # Convert ground truth from int to label
    ground_truth = [int_to_label[truth] for truth in ground_truth]

    # Iterate over predictions and change one sign to the homosign if it is a homosign
    new_predictions = []
    for pred in predictions:
        if pred in homosigns:
            new_predictions.append(homosigns[pred])
        else:
            new_predictions.append(pred)
    predictions = new_predictions
    
    new_ground_truth = []
    for truth in ground_truth:
        if truth in homosigns:
            new_ground_truth.append(homosigns[truth])
        else:
            new_ground_truth.append(truth)
    ground_truth = new_ground_truth

    total_correct = 0
    for pred, gt in zip(predictions, ground_truth):
        if pred == gt:
            total_correct += 1

    accuracy = total_correct / len(predictions)
    return accuracy
    #accuracy = np.sum(predictions == ground_truth) / len(predictions)

def calculate_accuracy_including_homosigns_adjusted_to_failed(model, X_test, y_test, label_to_one_hot):
    # Calculate the accuracy but count homosigns together
    predictions = np.argmax(model.predict(np.array(X_test), verbose=0), axis=1)
    ground_truth = np.argmax(np.array(y_test), axis=1)

    # Convert from one hot to int
    label_to_int = {}
    for key in label_to_one_hot.keys():
        label_to_int[key] = np.argmax(label_to_one_hot[key])
    int_to_label = {v:k for k, v in label_to_int.items()}

    # Generate known homosigns
    homosigns = {
        'awake': 'wake',
        'glasswindow': 'tooth',
        'beside': 'person',
        'finger': 'wait',
        'chin': 'lips',
        'police': 'hungry',
        'into': 'elephant'
    }

    # Convert predictions from int to label
    predictions = [int_to_label[pred] for pred in predictions]

    # Convert ground truth from int to label
    ground_truth = [int_to_label[truth] for truth in ground_truth]
    
    # Iterate over predictions and change one sign to the homosign if it is a homosign
    # Iterate over predictions and change one sign to the homosign if it is a homosign
    new_predictions = []
    for pred in predictions:
        if pred in homosigns:
            new_predictions.append(homosigns[pred])
        else:
            new_predictions.append(pred)
    predictions = new_predictions
    
    new_ground_truth = []
    for truth in ground_truth:
        if truth in homosigns:
            new_ground_truth.append(homosigns[truth])
        else:
            new_ground_truth.append(truth)
    ground_truth = new_ground_truth
    
    total_correct = 0
    for pred, gt in zip(predictions, ground_truth):
        if pred == gt:
            total_correct += 1
    
    accuracy = total_correct / 165294
    return accuracy
    #accuracy = np.sum(predictions == ground_truth) / 165294

# %%
test_accuracy = calculate_accuracy(lstm_model, X_test, y_test)
val_accuracy = calculate_accuracy(lstm_model, X_val, y_val)

print(f"Test Accuracy: {test_accuracy}")
print(f"Val Accuracy: {val_accuracy}")

test_accuracy_adjusted = calculate_failed_mediapipe_adjusted_accuracy(lstm_model, X_test, y_test)
val_accuracy_adjusted = calculate_failed_mediapipe_adjusted_accuracy(lstm_model, X_val, y_val)

print(f"Test Accuracy Adjusted: {test_accuracy_adjusted}")
print(f"Val Accuracy Adjusted: {val_accuracy_adjusted}")

test_accuracy_homosign = calculate_accuracy_including_homosigns(lstm_model, X_test, y_test, label_to_one_hot)
val_accuracy_homosign = calculate_accuracy_including_homosigns(lstm_model, X_val, y_val, label_to_one_hot)

print(f"Test Accuracy Homosign: {test_accuracy_homosign}")
print(f"Val Accuracy Homosign: {val_accuracy_homosign}")

test_accuracy_homosign_adjusted = calculate_accuracy_including_homosigns_adjusted_to_failed(lstm_model, X_test, y_test, label_to_one_hot)
val_accuracy_homosign_adjusted = calculate_accuracy_including_homosigns_adjusted_to_failed(lstm_model, X_val, y_val, label_to_one_hot)

print(f"Test Accuracy Adjusted Homosign: {test_accuracy_homosign_adjusted}")
print(f"Val Accuracy Adjusted Homosign: {val_accuracy_homosign_adjusted}")

# %%
test_fig, test_confusion_matrix, test_ordered_signs = create_confusion_matrix(lstm_model, X_test, y_test, signs, label_to_one_hot)

# %%
# all(test_confusion_matrix <= 1)

# %%
# For each row in the confusion matrix, find the sign that it's most confused with
most_confused_signs = []
row_signs = test_ordered_signs[::-1]
column_signs = test_ordered_signs

for i in range(len(row_signs)):
    row = test_confusion_matrix[i]
    max_index = np.argmax(row)
    max_confusion = row[max_index]
    curr_row_sign = row_signs[i]
    curr_column_sign = column_signs[max_index]
    if curr_row_sign != curr_column_sign:
        most_confused_signs.append((curr_row_sign, curr_column_sign, max_confusion))

# %%
most_confused_signs

# %%
# Remove all signs that have NaN values
drop_nan_confused_signs = [sign for sign in most_confused_signs if not np.isnan(sign[2])]

# Sort the most confused signs by the confusion value
sorted_most_confused_signs = sorted(drop_nan_confused_signs, key=lambda x: x[2], reverse=True)

# %%
print(sorted_most_confused_signs)


