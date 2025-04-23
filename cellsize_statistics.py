from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import measure, morphology
from skimage.morphology import skeletonize, dilation, closing
from scipy.ndimage import distance_transform_edt
import cv2
import os
import pandas as pd
import tifffile as tiff
from matplotlib.pyplot import imsave


def compute_declination(point1, point2):
    """Compute the declination angle (slope) between two points."""
    x1, y1 = point1
    x2, y2 = point2
    return np.arctan2(y2 - y1, x2 - x1)  # Compute the slope in radians

def find_middle_area(rotated_rect, box, binary_image):
    """
    Identify the middle third region of the detected object.
    Ensures selected long edges match the bounding box's orientation.
    """

    # Extract bounding box properties
    (center_x, center_y), (width, height), angle = rotated_rect
    length = max(width, height)
    # Ensure angle corresponds to the **major axis** (longest side)
    if width < height:
        angle -= 90  # Adjust angle to match the longest side

    # Compute distances and store them with their corresponding point pairs
    distances = [
        (np.linalg.norm(box[0] - box[1]), box[0], box[1]),
        (np.linalg.norm(box[0] - box[2]), box[0], box[2]),
        (np.linalg.norm(box[0] - box[3]), box[0], box[3]),
        (np.linalg.norm(box[1] - box[2]), box[1], box[2]),
        (np.linalg.norm(box[1] - box[3]), box[1], box[3]),
        (np.linalg.norm(box[2] - box[3]), box[2], box[3])
    ]
    # Extract only the point pairs from the sorted distances
    long_edge_1 = (distances[2][1], distances[2][2])  # Extract (p1, p2) from the 3rd longest distance
    long_edge_2 = (distances[3][1], distances[3][2])  # Extract (p1, p2) from the 4th longest distance


    # Compute middle third region using the selected long edges
    major_axis_vector1 = (long_edge_1[1] - long_edge_1[0]) / np.linalg.norm(long_edge_1[1] - long_edge_1[0])
    major_axis_vector2 = (long_edge_2[1] - long_edge_2[0]) / np.linalg.norm(long_edge_2[1] - long_edge_2[0])

    start_offset1 = major_axis_vector1 * (max(width, height) / 3)
    end_offset1 = major_axis_vector1 * (2 * max(width, height) / 3)
    start_offset2 = major_axis_vector2 * (max(width, height) / 3)
    end_offset2 = major_axis_vector2 * (2 * max(width, height) / 3)

    # Compute the middle third region
    middle_box = np.array([
        long_edge_1[0] + start_offset1,
        long_edge_1[0] + end_offset1,
        long_edge_2[0] + start_offset2,
        long_edge_2[0] + end_offset2
    ], dtype=np.int32)

    print(f"Selected Long Edges:\n  {long_edge_1}\n  {long_edge_2}")
    print(f"Middle Box Coordinates:\n{middle_box}")

    middle_box = middle_box[np.argsort(middle_box[:, 0])]  # Sort by x-coordinates
    if middle_box[0, 1] > middle_box[1, 1]:
        middle_box[[0, 1]] = middle_box[[1, 0]]  # Swap to correct order
    if middle_box[2, 1] < middle_box[3, 1]:
        middle_box[[2, 3]] = middle_box[[3, 2]]  # Swap to correct order
    # Convert the image to BGR for colored overlays
    output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    # # Draw the bounding box in green
    # cv2.drawContours(output_image, [box], 0, (255,0,  0), 1)
    #
    # # Draw the middle third region in red
    # cv2.drawContours(output_image, [middle_box], 0, ( 0,255, 0), 1)
    #
    # # Display the result
    # plt.figure(figsize=(6, 6))
    # plt.imshow(output_image, cmap="gray")
    # plt.title("Middle 1/3 Region")
    # plt.axis("off")
    # plt.show()

    return middle_box

def find_middle_bounding_box(binary_image, middle_box):
    """
    Extracts the middle third region and finds the bounding box of the white part inside it.
    Computes the new area based on the number of white pixels inside the new bounding box.
    """

    # Create a mask for the middle region
    mask = np.zeros_like(binary_image, dtype=np.uint8)
    cv2.fillPoly(mask, [middle_box], 255)

    # Extract the masked region
    middle_region = cv2.bitwise_and(binary_image, binary_image, mask=mask)

    # Find contours within the middle region
    contours, _ = cv2.findContours(middle_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No white region found in the middle box.")
        return 0, 0  # Return zero area and length if no white region is detected

    # Find the largest contour (assuming it's the main white region)
    largest_contour = max(contours, key=cv2.contourArea)

    # Compute the minimum area bounding box using `cv2.minAreaRect()`
    new_rotated_rect = cv2.minAreaRect(largest_contour)
    new_box = cv2.boxPoints(new_rotated_rect)
    new_box = np.array(new_box).astype(np.int64)  # Convert to integer

    # Create a mask for the new bounding box
    new_box_mask = np.zeros_like(binary_image, dtype=np.uint8)
    cv2.fillPoly(new_box_mask, [new_box], 255)

    # Count the white pixels in the new bounding box
    new_area = np.count_nonzero(cv2.bitwise_and(middle_region, middle_region, mask=new_box_mask))

    # Extract dimensions of the bounding box
    (new_width, new_height) = new_rotated_rect[1]
    new_length = max(new_width, new_height)  # Major axis length
    if new_length == 0:
        print("Error: new_length is 0, cannot proceed.")
        sys.exit()
    else:
        ave_width = new_area / new_length

    ave_width = new_area / new_length
    return new_box, ave_width

def find_middle2_area(rotated_rect, box, binary_image):
    """
    Identify the middle third region of the detected object.
    Ensures selected long edges match the bounding box's orientation.
    """

    # Extract bounding box properties
    (center_x, center_y), (width, height), angle = rotated_rect
    length = max(width, height)
    # Ensure angle corresponds to the **major axis** (longest side)
    if width < height:
        angle -= 90  # Adjust angle to match the longest side

    # Compute distances and store them with their corresponding point pairs
    distances = [
        (np.linalg.norm(box[0] - box[1]), box[0], box[1]),
        (np.linalg.norm(box[0] - box[2]), box[0], box[2]),
        (np.linalg.norm(box[0] - box[3]), box[0], box[3]),
        (np.linalg.norm(box[1] - box[2]), box[1], box[2]),
        (np.linalg.norm(box[1] - box[3]), box[1], box[3]),
        (np.linalg.norm(box[2] - box[3]), box[2], box[3])
    ]
    # Sort by the first column (distance) in descending order
    distances_sorted = sorted(distances, key=lambda x: x[0], reverse=True)
    distances = distances_sorted
    # Extract only the point pairs from the sorted distances
    long_edge_1 = (distances[2][1], distances[2][2])  # Extract (p1, p2) from the 3rd longest distance
    # Extract the remaining two points
    all_points = {tuple(box[i]) for i in range(4)}  # Convert to set for quick lookup
    used_points = {tuple(long_edge_1[0]), tuple(long_edge_1[1])}
    remaining_points = list(all_points - used_points)  # Get the two points not in long_edge_1

    # Ensure remaining points exist
    if len(remaining_points) != 2:
        print(f"Box:{box}, Edge_1:{long_edge_1}, Remaining:{remaining_points}")
        output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        # cv2.drawContours(output_image, [box], 0, (255,0,  0), 1)
        plt.figure(figsize=(6, 6))
        plt.imshow(output_image, cmap="gray")
        plt.title("Box")
        plt.axis("off")
        plt.show()
        raise ValueError("Error: Unable to find two remaining points for the second edge.")

    # Form the second long edge
    long_edge_2 = (np.array(remaining_points[0]), np.array(remaining_points[1]))
    # Compute major axis unit vectors
    major_axis_vector1 = (long_edge_1[1] - long_edge_1[0]) / np.linalg.norm(long_edge_1[1] - long_edge_1[0])
    major_axis_vector2 = (long_edge_2[1] - long_edge_2[0]) / np.linalg.norm(long_edge_2[1] - long_edge_2[0])

    # Ensure both vectors are aligned in the same direction
    if np.dot(major_axis_vector1, major_axis_vector2) < 0:
        # Reverse the order of points in long_edge_2
        long_edge_2 = (long_edge_2[1], long_edge_2[0])
        major_axis_vector2 = -major_axis_vector2  # Reverse the vector

    # Compute middle third region using the selected long edges
    major_axis_vector1 = (long_edge_1[1] - long_edge_1[0]) / np.linalg.norm(long_edge_1[1] - long_edge_1[0])
    major_axis_vector2 = (long_edge_2[1] - long_edge_2[0]) / np.linalg.norm(long_edge_2[1] - long_edge_2[0])

    start_offset1 = major_axis_vector1 * (max(width, height) / 4)
    middle_offset1 = major_axis_vector1 * (2 * max(width, height) / 4)
    end_offset1 = major_axis_vector1 * (3 * max(width, height) / 4)
    start_offset2 = major_axis_vector2 * (max(width, height) / 4)
    middle_offset2 = major_axis_vector2 * (2 * max(width, height) / 4)
    end_offset2 = major_axis_vector2 * (3 * max(width, height) / 4)

    # Compute the middle tw 25% regions
    middle_box1 = np.array([
        long_edge_1[0] + start_offset1,
        long_edge_1[0] + middle_offset1,
        long_edge_2[0] + start_offset2,
        long_edge_2[0] + middle_offset2
    ], dtype=np.int32)

    # print(f"Selected Long Edges:\n  {long_edge_1}\n  {long_edge_2}")
    # print(f"Middle Two 25% Boxes Coordinates:\n{middle_box1}")

    middle_box1 = middle_box1[np.argsort(middle_box1[:, 0])]  # Sort by x-coordinates
    if middle_box1[0, 1] > middle_box1[1, 1]:
        middle_box1[[0, 1]] = middle_box1[[1, 0]]  # Swap to correct order
    if middle_box1[2, 1] < middle_box1[3, 1]:
        middle_box1[[2, 3]] = middle_box1[[3, 2]]  # Swap to correct order

    # Compute the middle third region
    middle_box2 = np.array([
        long_edge_1[0] + middle_offset1,
        long_edge_1[0] + end_offset1,
        long_edge_2[0] + middle_offset2,
        long_edge_2[0] + end_offset2
    ], dtype=np.int32)

    # print(f"Middle Two 25% Boxes Coordinates:\n{middle_box2}")

    middle_box2 = middle_box2[np.argsort(middle_box2[:, 0])]  # Sort by x-coordinates
    if middle_box2[0, 1] > middle_box2[1, 1]:
        middle_box2[[0, 1]] = middle_box2[[1, 0]]  # Swap to correct order
    if middle_box2[2, 1] < middle_box2[3, 1]:
        middle_box2[[2, 3]] = middle_box2[[3, 2]]  # Swap to correct order
    # # Convert the image to BGR for colored overlays
    # output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    #
    # # Draw the bounding box in green
    # cv2.drawContours(output_image, [box], 0, (255,0,  0), 1)
    #
    # cv2.drawContours(output_image, [middle_box1], 0, (0,255,  0), 1)
    #
    # # Draw the middle third region in red
    # cv2.drawContours(output_image, [middle_box2], 0, ( 0, 0,255), 1)
    #
    # # Display the result
    # plt.figure(figsize=(6, 6))
    # plt.imshow(output_image, cmap="gray")
    # plt.title("Middle 25% Regions")
    # plt.axis("off")
    # plt.show()
    # print('Find 25%')
    return middle_box1, middle_box2
def cell_volume(ave_width, target_length):
    scale= 0.078
    ave_width = ave_width * scale
    target_length = target_length * scale
    vol_cylinder = np.pi * (ave_width/2.0) **2 * target_length + 4/3.0 * np.pi * (ave_width/2.0) **3
    vol_rectangle =  ave_width * ave_width * target_length + 4 / 3.0 * np.pi * (ave_width/2.0) ** 3

    return ave_width,target_length, vol_cylinder,vol_rectangle
def cell_intensity(i, binary_image):
##############################################################################====
    # dataset_folder = 'datasets/Practice_Files'
    # ori_img_file = f'Image {i:02d}.tif'  # Format filenames like 'Image 01.tif'
    # ori_img_path = os.path.join(dataset_folder, ori_img_file)
    # or_im = tiff.imread(ori_img_path)

    #Using raw data to calculate the stacked image
    data_folder = 'datasets/DataPackage/Raw data/'
    ori_img_file = f'Image {i}.tif'  # Format filenames like 'Image 01.tif'
    ori_img_path = os.path.join(data_folder, ori_img_file)
    or_im = tiff.imread(ori_img_path)
    or_im = np.mean(or_im, axis=0)

    plt.figure()
    plt.imshow(or_im, cmap='gray', vmin=0, vmax=or_im.max())
    plt.show()
    region_values = or_im[binary_image]
    average_intensity = np.mean(region_values) if region_values.size > 0 else 0
    # print("Average Intensity in the selected region:", average_intensity)
    return average_intensity

##############
"""Calculating cell size and average intensity"""
# Ensure the results directory exists
os.makedirs("results/cell size", exist_ok=True)

# Create a list to store the results
mask_folder = 'results/Statistic/cell size'
os.makedirs(mask_folder, exist_ok=True)
for i in range(1, 38):  # Loop from 1 to 38
    results_list = []
    img_file = f'Image {i:02d}'  # Format filenames like 'Image 01.tif' results/cell size/{img_file}_mask_individual.tiff

    img_name = f'{img_file}_masks_individually.tiff'
    image_path = os.path.join(mask_folder, img_name)
    image = Image.open(image_path)
    # Check the number of frames in the image
    try:
        print(f"Number of frames: {image.n_frames}")
    except Exception as e:
        print(f"Error reading frames: {e}")

    # Iterate through each frame of the TIFF
    for frame_index in range(image.n_frames):
        print(f"Image:{i}, Frame: {frame_index}")
        # Set the current frame
        image.seek(frame_index)

        # Convert the current frame to grayscale
        image_gray = np.array(image.convert('L'))

        # # Binarize the image (thresholding to get white areas corresponding to the cell)
        binary_image = image_gray > 200  # Adjust threshold as needed




        # Find contours
        binary_image = (binary_image * 255).astype(np.uint8)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour (assuming it's the cell)
        largest_contour = max(contours, key=cv2.contourArea)

        # Compute the minimum area bounding box
        rotated_rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rotated_rect)
        box = np.array(box).astype(np.int64)  # Convert to integer
        (center_x, center_y), (width, height), angle = rotated_rect
        if min(width, height) == 0 or max(width, height) < 4:
            continue
        length = max(width, height)
        short_length =  min(width, height)

        # Find middle 25%+25% regions
        middle_box1, middle_box2 = find_middle2_area(rotated_rect, box, binary_image)
        new_box1, new_width1 = find_middle_bounding_box(binary_image, middle_box1)
        # print('Find middle bounding for box1')
        # print(
        #     f"new_box1: {new_box1}, type: {type(new_box1)}, shape: {new_box1.shape if hasattr(new_box1, 'shape') else 'N/A'}")

        new_box2, new_width2 = find_middle_bounding_box(binary_image, middle_box2)
        # print('Find middle bounding for box2')
        # print(
        #     f"new_box2: {new_box2}, type: {type(new_box2)}, shape: {new_box2.shape if hasattr(new_box2, 'shape') else 'N/A'}")

    ###output images
        # output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        # cv2.drawContours(output_image, [box], 0, (255, 0, 0), 2)
        # plt.figure(figsize=(6, 6))
        # plt.imshow(output_image, cmap="gray")
        # # plt.title("Middle 25% Regions")
        # plt.axis("off")
        #
        # cell_size_folder = 'results/Statistic/cell size/Frame'
        # os.makedirs(cell_size_folder, exist_ok=True)
        # img_name = f'{img_file}_{frame_index}_box.tiff'
        # img_path = os.path.join(cell_size_folder, img_name)
        # imsave(img_path, output_image, format='tiff', cmap='gray')
        # #plt.show()
        #
        # output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        # cv2.drawContours(output_image, [new_box1], 0, (0, 255, 0), 2)
        # cv2.drawContours(output_image, [new_box2], 0, ( 0, 0, 255), 2)
        # plt.figure(figsize=(6, 6))
        # plt.imshow(output_image, cmap="gray")
        # # plt.title("Middle 25% Regions")
        # plt.axis("off")
        #
        # img_name = f'{img_file}_{frame_index}_middleboxs.tiff'
        # img_path = os.path.join(cell_size_folder, img_name)
        # imsave(img_path, output_image, format='tiff', cmap='gray')
        # #plt.show()


        ave_width = np.average([new_width1, new_width2])
        target_length = length - ave_width
        ave_width,target_length,vol_cylinder, vol_rectangle = cell_volume(ave_width, target_length)
        average_intensity = cell_intensity(i, binary_image)
        # Store results in a list
        results_list.append([
            frame_index, average_intensity, vol_rectangle, vol_cylinder, target_length, ave_width
        ])

    # Convert results list to DataFrame
    df_results = pd.DataFrame(results_list, columns=[
        "frame_index", "average_intensity", "rectangle_volume",
        "cylinder_volume", "target_length", "ave_width"
    ])
    df_results = df_results.astype(float)  # Convert all columns to float to avoid dtype issues
    # Save the DataFrame to Excel, overwriting if it exists

    # Verify by saving a backup CSV file for debugging
    csv_filename = os.path.join(mask_folder, f"{img_file}_real_intensity.csv")
    df_results.to_csv(csv_filename, index=False)
    print(f'Saved results to: {csv_filename}')

print("All images are processed")

# print(f"Large bounding: Length: {length:.2f}, Width: {short_length:.2f}, Angle: {angle:.2f}°")
# print(f"Middle 2x25% bounding ave width: {ave_width:.2f}, Target_length:{target_length:.2f}")
# print(f"cylinder_volume: {vol_cylinder:.2f}, rectangle_volume: {vol_rectangle:.2f}")
# print(f"average_intensity: {average_intensity:.2f}")
# # output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
# # cv2.drawContours(output_image, [box], 0, (255, 0, 0), 1)
# # cv2.drawContours(output_image, [new_box1], 0, (0, 255, 0), 1)
# # cv2.drawContours(output_image, [new_box2], 0, (0, 0, 255), 1)
# # # Display the result
# # plt.figure(figsize=(6, 6))
# # plt.imshow(output_image, cmap="gray")
# # # plt.title(f"Width:{ave_width:.2f}, Target Length:{target_length:.2f}")
# # plt.axis("off")
# # plt.show()
        # # Draw the bounding box on the image
        # output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        # cv2.drawContours(output_image, [box], 0, (0, 255, 0), 2)
        #
        # # Display the result
        # plt.figure(figsize=(6, 6))
        # plt.imshow(output_image)
        # plt.title(f"Bounding Box ")
        # plt.axis("off")
        # plt.show()

        # # Find middle region
        # middle_box = find_middle_area(rotated_rect, box, binary_image)
        #
        # new_box, ave_width1 = find_middle_bounding_box(binary_image, middle_box, box)
        # target_length1= length - ave_width1
        # vol_cylinder1, vol_rectangle1 = cell_infor(ave_width1, target_length1)
        # print(f"Large bounding: Length: {length:.2f}, Width: {short_length:.2f}, Angle: {angle:.2f}°")
        # print(f"Triple bounding ave width: {ave_width1:.2f}, Target_length:{target_length1:.2f}")
        # print(f"cylinder_volume: {vol_cylinder:.2f}, rectangle_volume: {vol_rectangle:.2f}")
        # # Convert the image to BGR for visualization
        # output_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        #
        # cv2.drawContours(output_image, [box], 0, (255, 0, 0), 1)
        # # # Draw the middle box in red
        # # cv2.drawContours(output_image, [middle_box], 0, (0,255, 0 ),1)
        #
        # # Draw the new bounding box inside the middle box in blue
        # cv2.drawContours(output_image, [new_box], 0, (0, 255, 0), 1)
        #
        # # Display the result
        # plt.figure(figsize=(6, 6))
        # plt.imshow(output_image, cmap="gray")
        # plt.title(f"Width:{ave_width:.2f}, Target Length:{target_length:.2f}")
        # plt.axis("off")
        # plt.show()

