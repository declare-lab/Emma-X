import os

# Define the directory containing the files
directory = "./"  # Change this to your target directory

# Loop through all files in the directory
for filename in os.listdir(directory):
    # Check if the filename starts with 'e2' and ends with '.mp4'
    if filename.startswith("pred") and filename.endswith(".mp4"):
        # Extract the desired parts
        parts = filename.split("_")
        if len(parts) >= 4:  # Ensure the filename has the expected structure
            x_y = "_".join(parts[-2:])  # Extract 'tx' and 'ry'
            new_filename = f"{x_y}"

            # Construct full paths
            old_filepath = os.path.join(directory, filename)
            new_filepath = os.path.join(directory, new_filename)

            # Rename the file
            os.rename(old_filepath, new_filepath)
            print(f"Renamed: {filename} -> {new_filename}")
