from PIL import Image
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from bm3d import bm3d, BM3DProfile
from matplotlib.pyplot import imsave
import os
import cv2
from sklearn.decomposition import PCA
from skimage.morphology import skeletonize, dilation, remove_small_objects
import math
from scipy.ndimage import zoom
import random
from skimage import exposure
import pandas as pd
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

def cell_intensity(image, masks, window_size=3, min_area=100, max_area=1250):
    """
    Calculate intensity of masks using the center region, and plot overlays.
    - Red outline: full mask
    - Blue outline: center region used for calculation
    - Black background with extracted region shown
    """
    intensity_list = []
    valid_intensities = []  # For computing the average intensity from valid masks

    for idx, ann in enumerate(masks):
        mask = ann['segmentation']
        y_coords, x_coords = np.where(mask)
        mask_area = len(y_coords)

        if mask_area == 0:
            intensity_list.append(0)
            continue

        # Compute the center of the mask
        center_y = np.median(y_coords).astype(int)
        center_x = np.median(x_coords).astype(int)

        # Define a small window around the center
        half_size = window_size // 2
        y_min = max(center_y - half_size, 0)
        y_max = min(center_y + half_size + 1, image.shape[0])
        x_min = max(center_x - half_size, 0)
        x_max = min(center_x + half_size + 1, image.shape[1])

        # Create a black background
        region_image = np.zeros_like(image)

        if (y_max > y_min) and (x_max > x_min):  # Ensure valid region
            region_image[y_min:y_max, x_min:x_max] = image[y_min:y_max, x_min:x_max]
            region_values = image[y_min:y_max, x_min:x_max]
            intensity = np.mean(region_values) if region_values.size > 0 else 0
        else:
            intensity = 0  # No valid region

        intensity_list.append(intensity)

        # Only consider masks within the valid size range for average calculation
        if min_area < mask_area < max_area:
            valid_intensities.append(intensity)

        # Plot each mask individually
        # if intensity < 0.3:
        #     fig, ax = plt.subplots(figsize=(6, 6))
        #     ax.imshow(image, cmap='gray')  # Black background with intensity overlay
        #
        #     # Draw red mask outline
        #     mask_uint8 = mask.astype(np.uint8) * 255
        #     contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #     for contour in contours:
        #         ax.plot(contour[:, 0, 0], contour[:, 0, 1], color='red', linewidth=1.5)
        #
        #     # Draw blue box for the center calculation area
        #     ax.add_patch(
        #         plt.Rectangle((x_min, y_min), window_size, window_size, edgecolor='blue', facecolor='none', linewidth=1.5))
        #
        #     ax.set_title(f'Mask {idx + 1}: Intensity = {intensity:.3f}')
        #     # plt.colorbar(ax.imshow(region_image, cmap='inferno'), label='Intensity')
        #     plt.show()

    # Compute the average intensity only from valid masks
    avg_intensity = np.mean(valid_intensities) if valid_intensities else 0

    # print(f'Average Intensity (Valid Masks): {avg_intensity:.3f}')

    return avg_intensity, intensity_list



def get_sorted_anns(image, anns, min_size=10):
    # sort the mask area from large to small
    # delete too large or too small masks
    if len(anns) == 0:
        return None
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    filtered_anns = []
    avg_intensity, intensity_list = cell_intensity(image, sorted_anns)

    # mask_folder = 'results/Statistic/sam_huge2'
    # img_mask_tiff_path = f'{mask_folder}/{img_file}_masks_individually_Before_PP.tiff'
    # num_masks = len(sorted_anns)
    # masks_array = []
    # for mask in sorted_anns:
    #     masks_array.append(mask['segmentation'])
    # if masks_array:
    #     masks_array = np.stack(masks_array, axis=0)
    #     tiff.imwrite(img_mask_tiff_path, masks_array, photometric='minisblack')  # Save as grayscale TIFF
    #
    # img = Image.open(img_mask_tiff_path)
    # frame_index = 0
    # save_dir = os.path.join(mask_folder, f'{img_file}_frames')
    # os.makedirs(save_dir, exist_ok=True)
    # try:
    #     while True:
    #         frame = img.convert("L")  # Grayscale
    #
    #         # Display the frame
    #         plt.figure(figsize=(4, 4))
    #         plt.imshow(frame, cmap='gray')
    #         plt.title(f"{img_file} - Frame {frame_index}")
    #         plt.axis("off")
    #         plt.tight_layout()
    #         plt.show()
    #
    #         # Save the frame
    #         frame_save_path = os.path.join(save_dir, f"{img_file}_frame_{frame_index:03d}.png")
    #         frame.save(frame_save_path)
    #         print(f"Saved: {frame_save_path}")
    #
    #         frame_index += 1
    #         img.seek(img.tell() + 1)
    #
    # except EOFError:
    #     print(f"✅ Finished: {frame_index} frames extracted and saved.")


    for i, ann in enumerate(sorted_anns):
        mask = ann['segmentation']
        cleaned_mask = remove_small_objects(mask.astype(bool), min_size=min_size)
        # Only keep the annotation if there's still a significant segmentation left
        # img_mask = np.zeros_like(image, dtype=np.uint8)
        # img_mask[cleaned_mask] = 255
        # if intensity_list[i] < 0.35*avg_intensity or intensity_list[i] > 1.6*avg_intensity: #100 > cleaned_mask.sum():
        #     fig, ax = plt.subplots(figsize=(6, 6))
        #     ax.imshow(image, cmap='gray')  # Black background with intensity overlay
        #
        #     # Draw red mask outline
        #     mask_uint8 = mask.astype(np.uint8) * 255
        #     contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #     for contour in contours:
        #         ax.plot(contour[:, 0, 0], contour[:, 0, 1], color='red', linewidth=1.5)
        #
        #     ax.set_title(f'i: {i}, Intensity: {intensity_list[i]:.3f}, Avg: {avg_intensity:.3f}', y=1.03)
        #     # plt.colorbar(ax.imshow(region_image, cmap='inferno'), label='Intensity')
        #     plt.axis('off')
        #     plt.show()
        if  100 < cleaned_mask.sum() < 50 ** 2 / 2 and 0.35* avg_intensity < intensity_list[i] < 1.6 * avg_intensity:
            ann['segmentation'] = cleaned_mask
            filtered_anns.append(ann)

    return filtered_anns

def show_anns(sorted_anns):
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    img_mask = np.zeros([img.shape[0], img.shape[1], 3])
    for ann in sorted_anns:
        m = ann['segmentation']
        color_arr = np.random.random(3)
        color_mask = np.concatenate([color_arr, [1]]) # np.concatenate([color_arr, [1]]) no transparency; 0.7 transparency
        img[m] = color_mask
        img_mask = img_mask + np.array([m] * 3).transpose([1, 2, 0]) * color_arr.reshape([1, 1, 3])
    ax.imshow(img)

    return img_mask

def show_anns_white(sorted_anns):
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.zeros((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1]))  # Black background
    img_mask = np.zeros_like(img)
    for ann in sorted_anns:
        m = ann['segmentation']
        img[m] = 255  # Set all mask pixels to white
        img_mask[m] = 255  # Store binary mask

    ax.imshow(img, cmap='gray')
    return img_mask



def get_center_lines(sorted_anns):

    imgs_cl = []
    for i in range(len(sorted_anns)):
        mask_binary = (sorted_anns[i]['segmentation'].astype(np.uint8) * 255).copy()

        coords = np.argwhere(mask_binary>0)
        x_min = coords[:, 0].min()
        x_max = coords[:, 0].max()
        y_min = coords[:, 1].min()
        y_max = coords[:, 1].max()

        # s = max(x_max - x_min, y_max - y_min)
        dia_kernel = mask_binary[x_min:x_max, y_min:y_max]

        h, w = dia_kernel.shape

        base_f = 0.4
        ratio = 2
        if h / w > ratio:
            fh = ratio*base_f*w/h
        else:
            fh = base_f

        if w / h > ratio:
            fw = ratio*base_f*h/w
        else:
            fw = base_f

        dia_kernel = zoom(dia_kernel, [fh, fw]) > 127

        if dia_kernel.max() <= 0:
            dia_kernel = np.ones([7, 7])>0
        # if x_max-x_min < s:
        #     x_min = (x_min+x_max)/2

        # plt.figure()
        # plt.imshow(mask_binary, cmap='gray')
        # plt.figure()
        # plt.imshow(dia_kernel, cmap='gray')

        # for ii in range(5):
        mask_binary = dilation(mask_binary, dia_kernel)
        # plt.figure()
        # plt.imshow(mask_binary, cmap='gray')
        # plt.show()
        skeleton = skeletonize(mask_binary)
        imgs_cl.append(skeleton)
    return imgs_cl

def is_touching_edge(mask):
    segmentation = mask['segmentation']  # Binary mask
    top_edge = segmentation[0:2, :].any()  # Any pixels in the first row?
    bottom_edge = segmentation[-2:-1, :].any()  # Any pixels in the last row?
    left_edge = segmentation[:, 0:2].any()  # Any pixels in the first column?
    right_edge = segmentation[:, -2:-1].any()  # Any pixels in the last column?
    return top_edge or bottom_edge or left_edge or right_edge

def show_anns_white_labled(sorted_anns):
    ax = plt.gca()
    ax.set_autoscale_on(False)

    height, width = sorted_anns[0]['segmentation'].shape
    img = np.zeros((height, width), dtype=np.uint8)  # Black background
    img_mask = np.zeros_like(img, dtype=np.uint8)    # Binary combined mask

    for i, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        img[m] = 255
        img_mask[m] = 255

        # Compute the centroid of the mask
        y_coords, x_coords = np.where(m)
        if len(y_coords) > 0 and len(x_coords) > 0:
            centroid_x = int(np.mean(x_coords))
            centroid_y = int(np.mean(y_coords))

            # Generate a random color except white
            text_color = (
                random.randint(0, 200) / 255,
                random.randint(0, 200) / 255,
                random.randint(0, 200) / 255
            )

            # Label starts from 1 to match the frame index
            ax.text(
                centroid_x,
                centroid_y,
                str(i + 1),  # FIX: Start labels from 1
                color=text_color,
                fontsize=10,
                ha='center',
                va='center',
                fontweight='bold'
            )

    ax.imshow(img, cmap='gray')
    return img_mask


def masks_task_output(img_file,im, sorted_anns, mask_folder):
    # 1. Plot all masks in color
    mask_path = f'{mask_folder}/{img_file}_masks_color.tiff'
    plt.figure()
    plt.imshow(im)
    img_mask = show_anns(sorted_anns)
    plt.axis('off')
    # plt.title(f"Masks for {img_file}")
    plt.savefig(mask_path, format='tiff', bbox_inches='tight', pad_inches=0, dpi=300)
    #plt.show()

    # 2. Plot masks' boundary
    mask_path = f'{mask_folder}/{img_file}_masks_boundary.tiff'
    plt.figure(figsize=(8, 8))  # Set figure size
    plt.imshow(im, cmap='gray')  # Display the original image as the background
    for mask in sorted_anns:# Overlay all mask outlines in red
        segmentation = mask['segmentation']  # Get segmentation mask
        plt.contour(segmentation, colors='red', linewidths=1.5)  # Draw red outline
    plt.axis('off')
    plt.savefig(mask_path, format='tiff', bbox_inches='tight', pad_inches=0, dpi=300)  # Save as .tiff
    #plt.show()


    # 3. plot individually masks
    img_mask_tiff_path = f'{mask_folder}/{img_file}_masks_individually.tiff'
    num_masks = len(sorted_anns)
    masks_array = []
    for mask in sorted_anns:
        masks_array.append(mask['segmentation'])
    if masks_array:
        masks_array = np.stack(masks_array, axis=0)
        tiff.imwrite(img_mask_tiff_path, masks_array, photometric='minisblack')  # Save as grayscale TIFF

    # 4. Index masks
    white_mask_path = f'{mask_folder}/{img_file}_masks_index.tiff'
    plt.figure()
    plt.imshow(im, cmap='gray')
    img_mask = show_anns_white_labled(sorted_anns)
    plt.axis('off')
    plt.savefig(white_mask_path, format='tiff', bbox_inches='tight', pad_inches=0, dpi=300)  # Save as .tiff
    #plt.show()

    # # 5. Un-index white masks
    # white_mask_path = f'results/masks/{img_file}_masks_white.tiff'
    # plt.figure()
    # plt.imshow(im, cmap='gray')
    # img_mask = show_anns_white(sorted_anns)
    # plt.axis('off')
    # plt.savefig(white_mask_path, format='tiff', bbox_inches='tight', pad_inches=0, dpi=300)  # Save as .tiff
    # #plt.show()
def cell_size_tasks_output(no_edge_masks,im,img_file, mask_folder):
    # 1. Plot all masks in color
    mask_folder = f'{mask_folder}/cell size'
    os.makedirs(mask_folder, exist_ok=True)

    mask_path= f'{mask_folder}/{img_file}_masks_color.tiff'
    plt.figure()
    plt.imshow(im)
    img_mask = show_anns(no_edge_masks)
    plt.axis('off')
    # plt.title(f"Masks for {img_file}")
    plt.savefig(mask_path, format='tiff', bbox_inches='tight', pad_inches=0, dpi=300)
    #plt.show()

    # 2. plot individually masks
    img_mask_tiff_path = f'{mask_folder}/{img_file}_masks_individually.tiff'
    num_masks = len(no_edge_masks)
    masks_array = []
    for mask in no_edge_masks:
        masks_array.append(mask['segmentation'])
    if masks_array:
        masks_array = np.stack(masks_array, axis=0)
        tiff.imwrite(img_mask_tiff_path, masks_array, photometric='minisblack')  # Save as grayscale TIFF

    # 3. Index masks
    white_mask_path = f'{mask_folder}/{img_file}_masks_index.tiff'
    plt.figure()
    plt.imshow(im, cmap='gray')
    img_mask = show_anns_white_labled(no_edge_masks)
    plt.axis('off')
    plt.savefig(white_mask_path, format='tiff', bbox_inches='tight', pad_inches=0, dpi=300)  # Save as .tiff
    #plt.show()

    # 4. Un-index white masks
    white_mask_path = f'{mask_folder}/{img_file}_masks_white.tiff'
    plt.figure()
    plt.imshow(im, cmap='gray')
    img_mask = show_anns_white(no_edge_masks)
    plt.axis('off')
    plt.savefig(white_mask_path, format='tiff', bbox_inches='tight', pad_inches=0, dpi=300)  # Save as .tiff
    #plt.show()



# 1️⃣ Function: Remove Fully Contained Masks
def remove_contained_masks(masks):
    """
    Removes smaller masks that are fully contained within larger masks.

    Args:
        masks (list of dicts): List of SAM-generated masks.

    Returns:
        List of filtered masks.
    """
    filtered_masks = []
    mask_arrays = [m['segmentation'] for m in masks]  # Extract binary masks

    for i, mask1 in enumerate(mask_arrays):
        keep = True
        for j, mask2 in enumerate(mask_arrays):
            if i != j and np.all((mask1 & mask2) == mask1):  # mask1 is inside mask2
                keep = False
                break
        if keep:
            filtered_masks.append(masks[i])  # Keep the full dictionary
    return filtered_masks


# 2️⃣ Function: Compute IoU Between Two Masks
def mask_iou(mask1, mask2):
    """Calculate IoU between two binary masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0


# 3️⃣ Function: Non-Maximum Suppression (NMS) for Masks
def nms_masks(masks, iou_threshold=0.3):
    """
    Apply Non-Maximum Suppression (NMS) to remove highly overlapping masks.

    Args:
        masks (list of dicts): List of SAM-generated masks.
        iou_threshold (float): IoU threshold for suppression.

    Returns:
        List of filtered masks.
    """
    kept_masks = []
    for i, mask1 in enumerate(masks):
        keep = True
        for j, mask2 in enumerate(masks):
            if i != j:
                iou = mask_iou(mask1['segmentation'], mask2['segmentation'])
                if iou > iou_threshold and mask1['area'] < mask2['area']:
                    # If mask1 is smaller and overlaps too much, remove it
                    keep = False
                    break
        if keep:
            kept_masks.append(mask1)
    return kept_masks


# 4️⃣ Function: Erode Mask for Partial Overlap Removal
def erode_mask(mask, kernel_size=3):
    """Erodes a binary mask to shrink its size slightly."""
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=1)


# 5️⃣ Function: Remove Partially Overlapping Masks Using Erosion
def remove_partial_overlaps(masks):
    """
    Further removes masks that partially overlap by using erosion.

    Args:
        masks (list of dicts): List of SAM-generated masks.

    Returns:
        List of filtered masks.
    """
    filtered_masks = []
    mask_arrays = [m['segmentation'] for m in masks]  # Extract binary masks

    for i, mask1 in enumerate(mask_arrays):
        keep = True
        eroded_mask1 = erode_mask(mask1)  # Shrink mask1 slightly

        for j, mask2 in enumerate(mask_arrays):
            if i != j and np.all((eroded_mask1 & mask2) == eroded_mask1):
                keep = False  # mask1 is still inside mask2 even after erosion
                break

        if keep:
            filtered_masks.append(masks[i])  # Keep this mask
    return filtered_masks


# 6️⃣ Function: Score-Based Filtering (Keeps Only the Best Mask Per Object)
def filter_by_score(masks):
    """
    Keeps only the highest confidence mask per overlapping region.

    Args:
        masks (list of dicts): List of SAM-generated masks.

    Returns:
        List of filtered masks.
    """
    masks.sort(key=lambda x: x['score'], reverse=True)  # Sort by highest score first
    final_masks = []

    for i, mask1 in enumerate(masks):
        keep = True
        for j, mask2 in enumerate(final_masks):
            if mask_iou(mask1['segmentation'], mask2['segmentation']) > 0.3:
                keep = False  # If another high-score mask exists for this region, remove
                break
        if keep:
            final_masks.append(mask1)

    return final_masks


# 7️⃣ Function: Full Pipeline to Apply All Filtering Steps
def process_masks(masks):
    """
    Apply all filtering steps to remove redundant, overlapping, or low-quality masks.

    Args:
        masks (list of dicts): List of SAM-generated masks.

    Returns:
        List of fully filtered masks.
    """
    print(f"Initial masks: {len(masks)}")

    # Step 1: Remove fully contained masks
    masks = remove_contained_masks(masks)
    print(f"After containment removal: {len(masks)}")

    # Step 2: Apply Non-Maximum Suppression (NMS)
    masks = nms_masks(masks, iou_threshold=0.3)
    print(f"After NMS: {len(masks)}")

    # Step 3: Remove partially overlapping masks using erosion
    masks = remove_partial_overlaps(masks)
    print(f"After partial overlap removal: {len(masks)}")

    # # Step 4: Use score-based filtering
    # masks = filter_by_score(masks)
    print(f"Final mask count: {len(masks)}")

    return masks

def apply_closing_to_masks(masks, kernel_size=5):
    """
    Applies morphological closing to each mask and ensures masks remain valid.

    Args:
        masks (list of dicts): List of SAM-generated masks.
        kernel_size (int): Size of the structuring element for closing.

    Returns:
        List of processed masks with closing applied.
    """
    closed_masks = []
    kernel = np.ones((kernel_size, kernel_size), np.uint8)  # Structuring element

    for mask in masks:
        original_mask = mask['segmentation'].astype(np.uint8)

        # Apply closing and ensure it does not erase the mask
        closed_mask = cv2.morphologyEx(original_mask, cv2.MORPH_CLOSE, kernel)

        # If the mask disappears, revert to the original
        if closed_mask.sum() == 0:
            closed_mask = original_mask

        # Ensure mask remains binary (0s and 1s)
        closed_mask = (closed_mask > 0).astype(bool)

        # Store the modified mask while keeping the original structure
        new_mask = mask.copy()
        new_mask['segmentation'] = closed_mask
        closed_masks.append(new_mask)


        # # Show original and closed masks side by side
        # fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        # axes[0].imshow(original_mask, cmap='gray')
        # axes[0].set_title(f"Original Mask {idx}")
        # axes[0].axis("off")
        #
        # axes[1].imshow(closed_mask, cmap='gray')
        # axes[1].set_title(f"Closed Mask {idx}")
        # axes[1].axis("off")
        #
        # #plt.show()

    return closed_masks

def origin_segment(im,img_file):
    # Generate masks
    mask_generator = SamAutomaticMaskGenerator(
        sam, points_per_side=64, box_nms_thresh=0.3, crop_nms_thresh=0.3
    )
    # mask_generator = SamAutomaticMaskGenerator(
    #     sam, points_per_side=32, box_nms_thresh=0.2, crop_nms_thresh=0.2
    # )
    masks = mask_generator.generate(im)
    masks = get_sorted_anns(masks)  # delete too small or too large objects
    print(f"Original masks: {len(masks)}")
    mask_path = f'test/{img_file}_Org_SAM.tiff'
    plt.figure()
    plt.imshow(im)
    img_mask = show_anns(masks)
    plt.axis('off')
    # plt.title(f"SAM {img_file}")
    plt.savefig(mask_path, format='tiff', bbox_inches='tight', pad_inches=0, dpi=300)
    #plt.show()

    # Example Usage
    # Assuming `masks` is the list of SAM-generated masks
    filtered_masks = process_masks(masks)
    print(f"Filtered masks: {len(filtered_masks)}")
    mask_path = f'test/{img_file}_Org_SAM_filtered.tiff'
    plt.figure()
    plt.imshow(im)
    img_mask = show_anns(filtered_masks)
    plt.axis('off')
    # plt.title(f"SAM Filtered {img_file}")
    plt.savefig(mask_path, format='tiff', bbox_inches='tight', pad_inches=0, dpi=300)
    #plt.show()

    # # mask tasks output images
    # masks_task_output(img_file, im, sorted_anns)

    # cell size tasks output images
    # Filter the cell masks on edges
    no_edge_masks = [mask for mask in filtered_masks if not is_touching_edge(mask)]
    # cell_size_tasks_output(no_edge_masks, im, img_file)
    mask_path = f'test/{img_file}_Org_SAM_NoEdge.tiff'
    plt.figure()
    plt.imshow(im)
    img_mask = show_anns(no_edge_masks)
    plt.axis('off')
    # plt.title(f"SAM NoEdge {img_file}")
    plt.savefig(mask_path, format='tiff', bbox_inches='tight', pad_inches=0, dpi=300)
    #plt.show()

    closed_masks = apply_closing_to_masks(no_edge_masks, kernel_size=5)

    mask_path = f'test/{img_file}_Org_SAM_NoEdge_Closing.tiff'
    plt.figure()
    plt.imshow(im)
    img_mask = show_anns(closed_masks)
    plt.axis('off')
    # plt.title(f"SAM NoEdge Closing {img_file}")
    plt.savefig(mask_path, format='tiff', bbox_inches='tight', pad_inches=0, dpi=300)
    #plt.show()
def mask_to_tuple(mask):
    """Helper to convert mask dict to a hashable tuple for comparison."""
    return tuple(mask['segmentation'].flatten())


# Define dataset and results folders
dataset_folder = 'sample_data/Image'

# Load SAM Model
# # ViT-Huge
sam = sam_model_registry["vit_h"](checkpoint="sam_models/sam_vit_h_4b8939.pth")

# ViT-Large
# sam = sam_model_registry["vit_l"](checkpoint="sam_models/sam_vit_l_0b3195.pth")

# # ViT-Base
# sam = sam_model_registry["vit_b"](checkpoint="sam_models/sam_vit_b_01ec64.pth")


sam.cuda()
mask_folder = 'sample_data/Results/Statistic/sam_huge2'
os.makedirs(mask_folder, exist_ok=True)
num_masks_list = []
# mask_path = f'{mask_folder}/{img_file}_masks_color.tiff'
# Process images from Image 01 to Image 38

for i in range(1, 2):  # Loop from 1 to 38

    img_file = f'Image {i:02d}'  # Format filenames like 'Image 01.tif'
    img_name = f'{img_file}.tif'
    img_path = os.path.join(dataset_folder, img_name)
    print(img_path)
    if not os.path.exists(img_path):
        print(f"Skipping {img_file} (not found)")
        continue

    print(f"Processing {img_file}...")

    # Load image
    im = Image.open(img_path)
    im = np.array(im, dtype=np.float32)
    im = (im - im.min()) / (im.max() - im.min())  # Normalize
    image = im
    img_path = f'{mask_folder}/{img_file}_original.tiff'
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.title(f"original {img_file}")
    imsave(img_path, im, format='tiff', cmap='gray')
    #plt.show()


    # Convert grayscale to 3-channel
    im = np.array([im] * 3).transpose([1, 2, 0])

    # segment on original data
    # Generate masks
    mask_generator = SamAutomaticMaskGenerator(
        sam, points_per_side=64, box_nms_thresh=0.3, crop_nms_thresh=0.3
    )
    masks_ori = mask_generator.generate(im)
    masks_ori = get_sorted_anns(image, masks_ori)  # delete too small or too large objects
    masks_ori = apply_closing_to_masks(masks_ori, kernel_size=5)
    no_edge_ori_masks = [mask for mask in masks_ori if not is_touching_edge(mask)]
    num_no_edge_ori_masks = len(no_edge_ori_masks)
    mask_path = f'{mask_folder}/{img_file}_ori_SAM.tiff'
    plt.figure()
    plt.imshow(im)
    img_mask = show_anns(no_edge_ori_masks)
    plt.axis('off')
    plt.title(f"Original_SAM {img_file}")
    plt.savefig(mask_path, format='tiff', bbox_inches='tight', pad_inches=0, dpi=300)
    # #plt.show()
    #
    white_mask_path = f'{mask_folder}/{img_file}_ori_SAM_index.tiff'
    plt.figure()
    plt.imshow(im, cmap='gray')
    img_mask = show_anns_white_labled(no_edge_ori_masks)
    plt.axis('off')
    plt.title(f"Original_SAM {img_file}")
    plt.savefig(white_mask_path, format='tiff', bbox_inches='tight', pad_inches=0, dpi=300)  # Save as .tiff
    # #plt.show()

    # Apply BM3D denoising
    sigma_est = 0.085
    im = bm3d(im, sigma_est)
    im = np.clip(im, 0, 1)  # Clip to [0,1]
    denoised_image = im
    # Save denoised image
    denoised_path = f'{mask_folder}/{img_file}_denoise.tiff'
    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.title(f"BM3D Denoised {img_file}")
    imsave(denoised_path, im, format='tiff', cmap='gray')
    # #plt.show()


    # Enhance contrast of the visualization image
    # im = exposure.rescale_intensity(im, in_range='image', out_range=(0, 1))

    # Generate masks
    mask_generator = SamAutomaticMaskGenerator(
        sam, points_per_side=64, box_nms_thresh=0.3, crop_nms_thresh=0.3
    )
    masks = mask_generator.generate(im)
    masks_o = masks
    masks = get_sorted_anns(denoised_image, masks) # delete too small or too large objects
    masks_r = masks
    # masks = apply_closing_to_masks(masks, kernel_size=5)
    no_edge_ori_denoised_masks = [mask for mask in masks if not is_touching_edge(mask)]
    masks3 = no_edge_ori_denoised_masks
    num_no_edge_ori_denoised_masks = len(no_edge_ori_denoised_masks)

    mask_path = f'{mask_folder}/{img_file}_ori_denoised_SAM_with_edge.tiff'
    plt.figure()
    plt.imshow(im)
    img_mask = show_anns(masks_r)
    plt.axis('off')
    # plt.title(f"Denoised SAM  with edge {img_file}")
    plt.savefig(mask_path, format='tiff', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()



    mask_path = f'{mask_folder}/{img_file}_ori_denoised_SAM.tiff'
    plt.figure()
    plt.imshow(im)
    img_mask = show_anns(no_edge_ori_denoised_masks)
    plt.axis('off')
    plt.title(f"Denoised SAM {img_file}")
    plt.savefig(mask_path, format='tiff', bbox_inches='tight', pad_inches=0, dpi=300)
    # plt.show()

    white_mask_path = f'{mask_folder}/{img_file}_ori_denoised_SAM_index.tiff'
    plt.figure()
    plt.imshow(im, cmap='gray')
    img_mask = show_anns_white_labled(no_edge_ori_denoised_masks)
    plt.axis('off')
    plt.title(f"Denoised SAM {img_file}")
    plt.savefig(white_mask_path, format='tiff', bbox_inches='tight', pad_inches=0, dpi=300)  # Save as .tiff
    # plt.show()

    # Post processing
    post_processed_masks = process_masks(masks)  # Apply all filtering steps to remove redundant, overlapping, or low-quality masks.
    no_edge_post_processed_masks = [mask for mask in post_processed_masks if not is_touching_edge(mask)]
    num_no_edge_post_processed_masks = len(no_edge_post_processed_masks)
    mask_path = f'{mask_folder}/{img_file}_ori_denoise_SAM_PP.tiff'
    plt.figure()
    plt.imshow(im)
    img_mask = show_anns(no_edge_post_processed_masks)
    plt.axis('off')
    plt.title(f"Denoised SAM PP {img_file}")
    plt.savefig(mask_path, format='tiff', bbox_inches='tight', pad_inches=0, dpi=300)
    # #plt.show()


    # masks_p = no_edge_post_processed_masks
    #
    # # Convert to sets
    # masks1_set = set(map(mask_to_tuple, masks_o))
    # masks2_set = set(map(mask_to_tuple, masks_r))
    # masks3_set = set(map(mask_to_tuple, masks_p))
    #
    # # Find removed masks
    # removed_masks_small_large_lowhigh_intensity = masks1_set - masks2_set
    # removed_masks_pp = masks2_set - masks3_set
    #
    # # Map back to original dicts in masks1
    # removed_masks1 = [m for m in masks_o if mask_to_tuple(m) in removed_masks_small_large_lowhigh_intensity]
    # filtered_mask_path = f'{mask_folder}/{img_file}_filtered_masks.tiff'
    # plt.figure()
    # plt.imshow(im)
    # _ = show_anns(removed_masks1)
    # plt.axis('off')
    # plt.title(f"Filtered Masks {img_file}")
    # plt.savefig(filtered_mask_path, format='tiff', bbox_inches='tight', pad_inches=0, dpi=300)
    # plt.show()
    #
    # removed_masks2 = [m for m in masks_r if mask_to_tuple(m) in removed_masks_pp]
    # filtered_mask_path = f'{mask_folder}/{img_file}_pp_masks.tiff'
    # plt.figure()
    # plt.imshow(im)
    # _ = show_anns(removed_masks2)
    # plt.axis('off')
    # plt.title(f"PP Masks {img_file}")
    # plt.savefig(filtered_mask_path, format='tiff', bbox_inches='tight', pad_inches=0, dpi=300)
    # plt.show()

    white_mask_path = f'{mask_folder}/{img_file}_ori_denoise_SAM_PP_index.tiff'
    plt.figure()
    plt.imshow(im, cmap='gray')
    img_mask = show_anns_white_labled(no_edge_post_processed_masks)
    plt.axis('off')
    plt.title(f"Denoised SAM PP {img_file}")
    plt.savefig(white_mask_path, format='tiff', bbox_inches='tight', pad_inches=0, dpi=300)  # Save as .tiff
    # #plt.show()
    num_masks_list.append([i, num_no_edge_ori_masks, num_no_edge_ori_denoised_masks, num_no_edge_post_processed_masks])

    # cell_size_tasks_output(no_edge_post_processed_masks, im, img_file, mask_folder)
    print(f"Saved results for {img_file}")
# Convert list to a DataFrame
df = pd.DataFrame(num_masks_list, columns=['Image', 'ori_num', 'denoised_num', 'pp_num'])

# Save as CSV file
df.to_csv(f'{mask_folder}/num_masks.csv', index=False)

print("Processing complete for all images.")


