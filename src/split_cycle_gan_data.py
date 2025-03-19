import os
import random
import shutil

def sample_images_randomly(
    base_dir="data_for_cycle_gan", 
    fraction=0.2, 
    seed=42
):
    """
    This function will randomly select a given fraction of images from
    the 'rgb' and 'sentinel2' folders inside base_dir and copy them to
    new folders named 'rgb_sampled' and 'sentinel2_sampled' respectively.
    For each saved image, a notification is printed.
    """

    # Print information about the process
    print("Starting the sampling process...")
    print(f"Base directory: {base_dir}")
    print(f"Sampling fraction: {fraction * 100}%")
    
    # Set the random seed for reproducibility
    random.seed(seed)

    # Define paths for rgb and sentinel2 folders
    rgb_dir = os.path.join(base_dir, "rgb")
    sentinel2_dir = os.path.join(base_dir, "sentinel2")

    # Define paths for the output directories
    rgb_output_dir = os.path.join(base_dir, "rgb_sampled")
    sentinel2_output_dir = os.path.join(base_dir, "sentinel2_sampled")

    # Create output directories if they do not exist
    os.makedirs(rgb_output_dir, exist_ok=True)
    os.makedirs(sentinel2_output_dir, exist_ok=True)

    # List all files in the rgb and sentinel2 directories
    # (Assuming all files in these folders are images)
    rgb_files = [f for f in os.listdir(rgb_dir) if os.path.isfile(os.path.join(rgb_dir, f))]
    sentinel2_files = [f for f in os.listdir(sentinel2_dir) if os.path.isfile(os.path.join(sentinel2_dir, f))]

    # Calculate how many files we need to sample
    num_rgb_to_sample = int(len(rgb_files) * fraction)
    num_sentinel2_to_sample = int(len(sentinel2_files) * fraction)

    # Randomly sample the files
    rgb_sampled = random.sample(rgb_files, num_rgb_to_sample)
    sentinel2_sampled = random.sample(sentinel2_files, num_sentinel2_to_sample)

    # Copy the sampled files to the new directories and print a notification for each
    for file_name in rgb_sampled:
        source_path = os.path.join(rgb_dir, file_name)
        target_path = os.path.join(rgb_output_dir, file_name)
        shutil.copy2(source_path, target_path)
        # Print a notification for each saved file
        print(f"Saved file from 'rgb': {file_name} -> {target_path}")
    
    for file_name in sentinel2_sampled:
        source_path = os.path.join(sentinel2_dir, file_name)
        target_path = os.path.join(sentinel2_output_dir, file_name)
        shutil.copy2(source_path, target_path)
        # Print a notification for each saved file
        print(f"Saved file from 'sentinel2': {file_name} -> {target_path}")

    # Print the result of the sampling process
    print(f"Sampled and copied {len(rgb_sampled)} files from 'rgb' to '{rgb_output_dir}'")
    print(f"Sampled and copied {len(sentinel2_sampled)} files from 'sentinel2' to '{sentinel2_output_dir}'")
    print("Sampling process completed successfully!")

# If you want to run this script directly, you can uncomment the line below:
sample_images_randomly(base_dir="data_for_cycle_gan", fraction=0.2, seed=42)
