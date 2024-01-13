import os
import re

folder_path = "saved_modelsv4"

# Function to extract iteration number from a filename
def extract_iteration_number(filename):
    match = re.search(r"iteration_(\d+)\.(npy|pickle|h5)", filename)
    if match:
        return int(match.group(1))
    return -1

# Get list of files in the folder
files = os.listdir(folder_path)

# Extract iteration numbers
iteration_numbers = [extract_iteration_number(file) for file in files]

# Find the highest iteration number
max_iteration = max(iteration_numbers)

# Delete files with iteration numbers less than the highest found,
# except for iteration.txt
for file, iteration_number in zip(files, iteration_numbers):
    if file != "iteration.txt" and iteration_number < max_iteration:
        os.remove(os.path.join(folder_path, file))
        print(f"Deleted: {file}")
