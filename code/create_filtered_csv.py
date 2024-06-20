# import necessary packages
import pandas as pd
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Set directories for models and data
MODEL_DIR = "path_to/directory/"
filenameCSV = MODEL_DIR + "filename_to_save_performances.csv"
DATA_DIR = "/"
data_train = DATA_DIR + "/path_to/train/"
data_valid = DATA_DIR + "/path_to/valid/"

# Load dataset with IDs
df_train = pd.read_csv(DATA_DIR + "path_to/train_concepts.csv", sep="\t")
df_valid = pd.read_csv(DATA_DIR + "path_to/valid_concepts.csv", sep="\t")

# Split the 'ID,CUIs' column into separate 'ID' and 'CUIs' columns
df_train[['ID', 'CUIs']] = df_train['ID,CUIs'].str.split(',', expand=True)
df_valid[['ID', 'CUIs']] = df_valid['ID,CUIs'].str.split(',', expand=True)

# Add a new column 'image_path' with the path to the image file for each ID
df_train["image_path"] = data_train + df_train.ID + ".jpg"
df_valid["image_path"] = data_valid + df_valid.ID + ".jpg"

# Create a list of unique CUIs in the training set
cuis_list = []
for (i, row) in df_train.iterrows():
    for cui in row["CUIs"].split(";"):
        if cui not in cuis_list:
            cuis_list.append(cui)

# Create a list of unique CUIs in the validation set
valid_cuis_list = []
for (i, row) in df_valid.iterrows():
    for cui in row["CUIs"].split(";"):
        if cui not in valid_cuis_list:
            valid_cuis_list.append(cui)


# Filter rows that contain the specific CUI for training
filtered_df_train = df_train[df_train['CUIs'].str.contains('C0000001')]

# Select only the necessary 'ID' and 'CUIs' columns before saving
filtered_df_train = filtered_df_train[['ID', 'CUIs']]

# Save the new training CSV file with the filtered rows
filtered_df_train.to_csv("/path_to/train_popular_cui.csv", index=False)

# Filter rows that contain the specific CUI for validation
filtered_df_valid = df_valid[df_valid['CUIs'].str.contains('C0000001')]

# Select only the necessary 'ID' and 'CUIs' columns before saving
filtered_df_valid = filtered_df_valid[['ID', 'CUIs']]

# Save the new validation CSV file with the filtered rows
filtered_df_valid.to_csv("/path_to/valid_popular_cui.csv", index=False)


df1 = pd.read_csv("/path_to/train_popular_cui.csv", sep="\t")

# Split the "ID,CUIs" column into "ID" and "CUIs"
df1[['ID', 'CUIs']] = df1['ID,CUIs'].str.split(',', expand=True)

# Remove the specific CUI ("C0000001") from all cells
df1['CUIs'] = df1['CUIs'].apply(lambda x: ';'.join([cui for cui in x.split(';') if cui != 'C0000001']))

# Remove rows where an ID has no CUIs and the "CUIs" column is empty
df1 = df1[(df1['CUIs'].notnull()) & (df1['CUIs'] != '')]

# Select only the necessary 'ID' and 'CUIs' columns before saving
df1 = df1[['ID', 'CUIs']]

# Save the new DataFrame to a CSV file
df1.to_csv("/path_to/train_without_popular_cui.csv", index=False)


df2 = pd.read_csv("/path_to/valid_popular_cui.csv", sep="\t")

# Split the "ID,CUIs" column into "ID" and "CUIs"
df2[['ID', 'CUIs']] = df2['ID,CUIs'].str.split(',', expand=True)

# Remove the specific CUI ("C0000001") from all cells
df2['CUIs'] = df2['CUIs'].apply(lambda x: ';'.join([cui for cui in x.split(';') if cui != 'C0000001']))

# Remove rows where an ID has no CUIs and the "CUIs" column is empty
df2 = df2[(df2['CUIs'].notnull()) & (df2['CUIs'] != '')]

# Select only the necessary 'ID' and 'CUIs' columns before saving
df2 = df2[['ID', 'CUIs']]

# Save the new DataFrame to a CSV file
df2.to_csv("/path_to/valid_without_popular_cui.csv", index=False)





# List of CUIs you want to keep
desired_cuis = ["C0000001", "C0000001", "C0000001", "C0000001", "C0000001",
                "C0000001", "C0000001", "C0000001", "C0000001", "C0000001"]

# Load the original training DataFrame
df_train = pd.read_csv(DATA_DIR + "/train_without_popular_cui.csv", sep="\t")

# Split the 'ID,CUIs' column into 'ID' and 'CUIs'
df_train[['ID', 'CUIs']] = df_train['ID,CUIs'].str.split(',', expand=True)

# Filter the DataFrame to include only rows with desired CUIs
def filter_cuis(row):
    cuis = row["CUIs"].split(";")
    cuis_filtered = [cui for cui in cuis if cui in desired_cuis]
    return ';'.join(cuis_filtered)

df_train_filtered = df_train.copy()
df_train_filtered['CUIs'] = df_train_filtered.apply(filter_cuis, axis=1)

# Remove rows where there are no CUIs after filtering
df_train_filtered = df_train_filtered[df_train_filtered['CUIs'] != '']

# Select only the 'ID' and 'CUIs' columns
df_train_filtered = df_train_filtered[['ID', 'CUIs']]

# Save the filtered DataFrame to a new CSV file
filenameCSV_train_filtered = MODEL_DIR + "train_filt.csv"
df_train_filtered.to_csv(filenameCSV_train_filtered, index=False)


df_valid = pd.read_csv(DATA_DIR + "/valid_without_popular_cui.csv", sep="\t")

# Split the 'ID,CUIs' column into 'ID' and 'CUIs'
df_valid[['ID', 'CUIs']] = df_valid['ID,CUIs'].str.split(',', expand=True)

# Filter the DataFrame to include only rows with desired CUIs
def filter_cuis(row):
    cuis = row["CUIs"].split(";")
    cuis_filtered = [cui for cui in cuis if cui in desired_cuis]
    return ';'.join(cuis_filtered)

df_valid_filtered = df_valid.copy()
df_valid_filtered['CUIs'] = df_valid_filtered.apply(filter_cuis, axis=1)

# Remove rows where there are no CUIs after filtering
df_valid_filtered = df_valid_filtered[df_valid_filtered['CUIs'] != '']

# Select only the 'ID' and 'CUIs' columns
df_valid_filtered = df_valid_filtered[['ID', 'CUIs']]

# Save the filtered DataFrame to a new CSV file
filenameCSV_valid_filtered = MODEL_DIR + "valid_filt.csv"
df_valid_filtered.to_csv(filenameCSV_valid_filtered, index=False)



