import shutil
import os
import pandas as pd

DATA_DIR = os.getcwd() + "/"
MODEL_DIR = DATA_DIR + "ressep/"
filenameCSV = MODEL_DIR + "EDV2performances.csv"
data_train = DATA_DIR + "train/"
data_valid = DATA_DIR + "valid/"

# Load dataset with IDs
df_train = pd.read_csv(DATA_DIR + "train_concepts.csv", sep=",")
df_valid = pd.read_csv(DATA_DIR + "valid_concepts.csv", sep=",")

df_train

df_valid

df_train["image_path"] = data_train + df_train.ID + ".jpg"
df_valid["image_path"] = data_valid + df_valid.ID + ".jpg"

cuis_list = []
for (i, row) in df_train.iterrows():
    for cui in row["CUIs"].split(";"):
        if not cui in cuis_list:
            cuis_list.append(cui)

valid_cuis_list = []
for (i, row) in df_valid.iterrows():
    for cui in row["CUIs"].split(";"):
        if cui not in valid_cuis_list:
            valid_cuis_list.append(cui)

# Load the generated CSV file
df1 = pd.read_csv("/path_to_your_csv_file/train_filt.csv")

# Get a list of unique IDs present in the CSV file
relevant_ids = df1['ID'].tolist()

# Original directory containing all train images
original_image_dir = "/path_to_folder/train/"

# Directory where the relevant images will be stored
relevant_image_dir = "/path_to_folder/train_new_folder/"

# Create the directory if it does not exist
os.makedirs(relevant_image_dir, exist_ok=True)

# Iterate over the image files in the original directory
for filename in os.listdir(original_image_dir):
    # Get the image ID from the filename
    image_id = filename.split('.')[0]
    # If the ID is in the list of relevant IDs, copy the image to the new directory
    if image_id in relevant_ids:
        shutil.copy(os.path.join(original_image_dir, filename), relevant_image_dir)

# Load the generated CSV file
df2 = pd.read_csv("/path_to_your_csv_file/valid_filt.csv")

# Get a list of unique IDs present in the CSV file
relevant_ids = df2['ID'].tolist()

# Original directory containing all valid images
original_image_dir = "/path_to_folder/valid/"

# Directory where the relevant images will be stored
relevant_image_dir = "/path_to_folder/valid_new_folder/"

# Create the directory if it does not exist
os.makedirs(relevant_image_dir, exist_ok=True)

# Iterate over the image files in the original directory
for filename in os.listdir(original_image_dir):
    # Get the image ID from the filename
    image_id = filename.split('.')[0]
    # If the ID is in the list of relevant IDs, copy the image to the new directory
    if image_id in relevant_ids:
        shutil.copy(os.path.join(original_image_dir, filename), relevant_image_dir)
