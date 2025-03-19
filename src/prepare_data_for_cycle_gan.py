import os
import sys
import tifffile as tiff
import numpy as np
import cv2
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def init_worker():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    else:
        sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

# Input and output directories
input_dir = "/media/kursat/TOSHIBA EXT48/projects/satellite/YENI/visual-language-model/mixture_of_experts/SSL4EO-S12-main/src/download_data/data"  
base_dir = "/media/kursat/TOSHIBA EXT48/projects/satellite/YENI/visual-language-model/mixture_of_experts/data_for_cycle_gan"

output_rgb_dir = os.path.join(base_dir, "rgb")
output_sentinel2_dir = os.path.join(base_dir, "sentinel2")
os.makedirs(output_rgb_dir, exist_ok=True)
os.makedirs(output_sentinel2_dir, exist_ok=True)

# Exclude output directories from processing
exclude_dirs = {output_rgb_dir, output_sentinel2_dir}

def band_key(filename):
    """
    Returns a float key based on the band string extracted from the filename.
    """
    band_str = filename[1:-4]
    try:
        return float(band_str.replace("A", ".1"))
    except:
        return 0

def resample_band(band, target_shape):
    """
    Resamples the given 2D band to the target shape.
    Uses cv2.INTER_AREA for downscaling and cv2.INTER_LINEAR for upscaling.
    """
    interp = cv2.INTER_AREA if (band.shape[0] > target_shape[0] or band.shape[1] > target_shape[1]) else cv2.INTER_LINEAR
    return cv2.resize(band, (target_shape[1], target_shape[0]), interpolation=interp)

def read_band(file_path):
    """
    Reads a single TIFF file and returns a numpy array of type float32.
    """
    return tiff.imread(file_path).astype(np.float32)

def process_folder(args):
    folder_index, root, tif_files = args
    print(f"{folder_index}. Processing folder: {root}", flush=True)
    
    # Sort files by band key
    tif_files.sort(key=band_key)
    
    # Read TIFF files in parallel (I/O bound, so ThreadPoolExecutor is beneficial)
    file_paths = [os.path.join(root, f) for f in tif_files]
    with ThreadPoolExecutor() as executor:
        bands = list(executor.map(read_band, file_paths))
    
    # Create mapping from band name (without .tif) to index
    band_indices = {f[:-4]: idx for idx, f in enumerate(tif_files)}
    
    # Use band "B2" as reference if available; otherwise, use the first band's shape.
    ref_shape = bands[band_indices["B2"]].shape if "B2" in band_indices else bands[0].shape
    
    # Resample bands to the reference shape if necessary
    bands = [resample_band(b, ref_shape) if b.shape != ref_shape else b for b in bands]
    multi_band = np.stack(bands, axis=0)
    multi_band_transposed = np.transpose(multi_band, (1, 2, 0))
    
    # Create a clean relative path name for output files.
    relative_path = os.path.relpath(root, input_dir)
    relative_path_clean = relative_path.replace(os.sep, '_')
    if relative_path_clean == '.':
        relative_path_clean = 'root'
    
    # Save multispectral image if it doesn't already exist.
    output_sentinel2_path = os.path.join(output_sentinel2_dir, f"{relative_path_clean}_sentinel2.tif")
    if not os.path.exists(output_sentinel2_path):
        tiff.imwrite(output_sentinel2_path, multi_band_transposed)
        print(f"   Saved multispectral image: {output_sentinel2_path}", flush=True)
    else:
        print(f"   Multispectral image already exists: {output_sentinel2_path}. Skipping multispectral save...", flush=True)
    
    # If B2, B3, and B4 bands are available, create an RGB image.
    if all(b in band_indices for b in ["B2", "B3", "B4"]):
        blue = bands[band_indices["B2"]]
        green = bands[band_indices["B3"]]
        red = bands[band_indices["B4"]]
        
        # Normalize images to the 0-1 range
        blue_norm = cv2.normalize(blue, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        green_norm = cv2.normalize(green, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        red_norm = cv2.normalize(red, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
        
        rgb = np.dstack((red_norm, green_norm, blue_norm))
        rgb_uint8 = (rgb * 255).astype(np.uint8)
        output_rgb_path = os.path.join(output_rgb_dir, f"{relative_path_clean}_rgb.png")
        if not os.path.exists(output_rgb_path):
            # Convert RGB to BGR for saving using OpenCV.
            cv2.imwrite(output_rgb_path, cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR))
            print(f"   Saved RGB image: {output_rgb_path}", flush=True)
        else:
            print(f"   RGB image already exists: {output_rgb_path}. Skipping RGB save...", flush=True)
    else:
        print(f"   WARNING: B2, B3, or B4 bands not found in folder: {root}", flush=True)
    return

def main():
    tasks = []
    folder_index = 1
    # Walk through all directories and add tasks for folders containing TIFF files
    for root, dirs, files in os.walk(input_dir):
        # Exclude output directories
        if any(os.path.commonpath([root, ex]) == ex for ex in exclude_dirs):
            continue
        
        tif_files = [f for f in files if f.lower().endswith('.tif') and f.startswith('B')]
        if not tif_files:
            continue
        
        # Create a clean relative path name for output file checks.
        relative_path = os.path.relpath(root, input_dir)
        relative_path_clean = relative_path.replace(os.sep, '_')
        if relative_path_clean == '.':
            relative_path_clean = 'root'
        
        # Determine output paths
        output_sentinel2_path = os.path.join(output_sentinel2_dir, f"{relative_path_clean}_sentinel2.tif")
        has_rgb = any(f.startswith("B2") for f in tif_files) and \
                  any(f.startswith("B3") for f in tif_files) and \
                  any(f.startswith("B4") for f in tif_files)
        output_rgb_path = os.path.join(output_rgb_dir, f"{relative_path_clean}_rgb.png") if has_rgb else None

        # If the multispectral file exists and (if applicable) the RGB file exists, skip this folder.
        if os.path.exists(output_sentinel2_path):
            if not has_rgb or (has_rgb and output_rgb_path is not None and os.path.exists(output_rgb_path)):
                print(f"Task for folder {root} already processed (files exist). Skipping...", flush=True)
                continue
        
        tasks.append((folder_index, root, tif_files))
        print(f"Task {folder_index} added for directory: {root}", flush=True)
        folder_index += 1
    
    print(f"Total tasks added: {len(tasks)}", flush=True)
    
    max_workers = 8  # Adjust based on your system capabilities
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:
        list(executor.map(process_folder, tasks))
    
    print("Completed.", flush=True)

if __name__ == "__main__":
    main()
