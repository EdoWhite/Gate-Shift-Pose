import pandas as pd
import numpy as np
import cv2
import argparse

parser = argparse.ArgumentParser(description="GSF testing with saved logits")

parser.add_argument('--file', type=str, help='Path to the CSV file containing gaze data')
parser.add_argument('--frame_path', type=str, help='Path to the frame image')

args = parser.parse_args()

# Load the gaze data from the CSV file
gaze_csv_path = args.file
gaze_data = pd.read_csv(gaze_csv_path)

# Filter the gaze data for the specific frame
#frame_gaze_data = gaze_data[gaze_data['frame'] == args.frame]

# Initialize the attention map with the frame dimensions 1920x1080
frame_width, frame_height = 1920, 1080
attention_map = np.zeros((frame_height, frame_width), dtype=np.float32)

# Function to apply a Gaussian blur centered at the gaze point
def apply_gaussian(x, y, map, sigma=15):
    # Create a 2D Gaussian kernel
    kernel_size = sigma * 3
    gauss_kernel = cv2.getGaussianKernel(kernel_size, sigma) * cv2.getGaussianKernel(kernel_size, sigma).T
    gauss_kernel /= gauss_kernel.max()

    # Define the bounds of the kernel in the map
    x_start, x_end = int(x - kernel_size / 2), int(x + kernel_size / 2)
    y_start, y_end = int(y - kernel_size / 2), int(y + kernel_size / 2)

    # Adjust the size of the kernel for the edges
    x_start_kernel, y_start_kernel = 0, 0
    x_end_kernel, y_end_kernel = kernel_size, kernel_size

    if x_start < 0:
        x_start_kernel -= x_start
        x_start = 0
    if y_start < 0:
        y_start_kernel -= y_start
        y_start = 0
    if x_end > map.shape[1]:
        x_end_kernel -= x_end - map.shape[1]
        x_end = map.shape[1]
    if y_end > map.shape[0]:
        y_end_kernel -= y_end - map.shape[0]
        y_end = map.shape[0]

    # Apply the kernel to the attention map
    map[y_start:y_end, x_start:x_end] += gauss_kernel[y_start_kernel:y_end_kernel, x_start_kernel:x_end_kernel]

# Apply Gaussian blur for each gaze point
for index, row in gaze_data.iterrows():
    apply_gaussian(row['x_pixel_coord'], row['y_pixel_coord'], attention_map)

    # Normalize the attention map
    attention_map /= attention_map.max()

    # Convert to an 8-bit image (scale from 0 to 255)
    attention_map_img = (attention_map * 255).astype(np.uint8)
    attention_map_color = cv2.applyColorMap(attention_map_img, cv2.COLORMAP_JET)

    # Load the original frame image
    frame_image = cv2.imread(args.frame_path)

    # Blend the attention map with the original frame
    overlayed_image = cv2.addWeighted(frame_image, 1, attention_map_color, 0.5, 0)

    # Display the overlayed image
    cv2.imshow('Attention Overlay', overlayed_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()