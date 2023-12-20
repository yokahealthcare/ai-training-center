import os
from datetime import datetime

import pandas as pd

folder_path = 'yolov8_extracted_angel'

# Get a list of all files in the folder
files = os.listdir(folder_path)

# Filter only CSV files with the target character in the filename
csv_files = [file for file in files if file.endswith('.csv')]

# Initialize an empty dataframe to store the combined data
combined_data = pd.DataFrame()

# Read each CSV file and append it to the combined dataframe
for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    df = pd.read_csv(file_path)
    combined_data = pd.concat([combined_data, df], ignore_index=True)

# drop any zero on values
combined_data = combined_data[combined_data.drop('class', axis=1).ne(0).all(axis=1)]

# Get the current date
current_date = datetime.now().strftime("%d%m%Y")

filename = f"{current_date}_{folder_path}.csv"
# Export the combined dataframe to a new CSV file
combined_data.to_csv(f'combined_result/{filename}', index=False)

print(f"Combined CSV file exported successfully to '{filename}'")
