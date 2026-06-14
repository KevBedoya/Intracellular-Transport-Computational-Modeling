
import os


def extract_csv_and_png_paths(destination_dirs):
    """
    Given a list of directory paths, return two lists:
    - csv_file_locations: paths to all .csv files found in the directories
    - png_file_locations: paths to all .png files found in the directories
    """
    csv_file_locations = []
    png_file_locations = []

    for dir_path in destination_dirs:
        if not os.path.isdir(dir_path):
            continue  # Skip if not a valid directory

        for filename in os.listdir(dir_path):
            full_path = os.path.join(dir_path, filename)
            if os.path.isfile(full_path):
                if filename.lower().endswith(".csv"):
                    csv_file_locations.append(full_path)
                elif filename.lower().endswith(".png"):
                    png_file_locations.append(full_path)

    return csv_file_locations, png_file_locations
