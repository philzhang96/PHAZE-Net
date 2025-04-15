import numpy as np
import cv2

def is_valid_crop(crop, min_size=20, std_thresh=5):
    """
    Check if an ROI crop is valid based on size, content, and visual quality.

    Parameters:
        crop (np.ndarray): The cropped ROI image.
        min_size (int): Minimum width and height for a valid crop.
        std_thresh (float): Minimum standard deviation to avoid flat images.

    Returns:
        bool: True if the crop passes all checks, False otherwise.
    """
    if crop is None:
        return False
    if not isinstance(crop, np.ndarray):
        return False

    h, w = crop.shape[:2]
    if h < min_size or w < min_size:
        return False

    if crop.ndim == 2:  # Grayscale
        std = np.std(crop)
    elif crop.ndim == 3:  # Colour image
        std = np.std(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    else:
        return False

    if std < std_thresh:
        return False

    if np.mean(crop) < 5 or np.mean(crop) > 250:
        return False  # extremely dark or bright

    return True


def clean_roi_batch(roi_list_dict, verbose=False):
    """
    Filters out invalid crops in a dictionary of ROI sequences.

    Parameters:
        roi_list_dict (dict): {region_name: [roi_0, roi_1, ...]}
        verbose (bool): Whether to print reasons for rejected crops.

    Returns:
        dict: Cleaned {region_name: [valid_rois]}
    """
    cleaned = {}
    for region, crops in roi_list_dict.items():
        valid_crops = []
        for i, crop in enumerate(crops):
            if is_valid_crop(crop):
                valid_crops.append(crop)
            elif verbose:
                print(f"❌ Invalid crop for '{region}' at index {i}")
        if valid_crops:
            cleaned[region] = valid_crops
    return cleaned
