import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from scipy.ndimage import measurements, center_of_mass, binary_dilation, zoom, binary_fill_holes
import nibabel as nib
from tqdm.notebook import tqdm
from scipy.signal import find_peaks
import os


# put smallest dimension in axis 0
def get_axes_maps(tensor):
    """
    This function takes a tensor and returns a mapping of axes and the reverse mapping.
    """
    # Create a mapping of axes
    axes_map = {i: (len(tensor.shape) - 1 - i) for i in range(len(tensor.shape))}
    # Create the reverse mapping
    reverse_map = {v: k for k, v in axes_map.items()}
    return axes_map, reverse_map

def transpose_tensor(tensor, axes_map):
    """
    This function takes a tensor and an axes map, and returns the transposed tensor.
    """
    # Get the list of axes for transposition
    axes = [axes_map[i] for i in range(len(tensor.shape))]
    # Perform the transposition
    return np.transpose(tensor, axes=axes)



def lung_segmentation(path, output_path, experimental=False, shqiperia=False, left_adjust=5, right_adjust=5):

    #if dir does not exist, create it
    if not os.path.exists(output_path):
        print("sarai fesso, oh...")
        os.makedirs(output_path)
    files = sorted(os.listdir(path))

    for i in tqdm(range(len(files)), colour='red', desc='segmenting for you baby ;)'):
        file = files[i]
        
        file_path = os.path.join(path, file)
        # if it already exists, skip
        if shqiperia is True and os.path.exists(os.path.join(output_path, file)): 
            #print(f"Skipping {file}")
            continue
        print(f"file is {file}")

        case = nib.load(file_path)
        img = case.get_fdata()

        axes_map, reverse_map = get_axes_maps(img)
        img = transpose_tensor(img, axes_map)

        body_seg = np.logical_and(img >= -500, img <= 2000).astype(int)

        # Find the air segmentation by inverting the body segmentation
        air_seg = 1 - body_seg
        # also remove everything with axis2 < 100
        air_seg[:, :100, :] = 0
        # Label different components in the air segmentation
        air_seg_labeled = label(air_seg, background=0)

        # Find the largest connected component in the air segmentation (this should be the outside air)
        # and remove it from the segmentation
        largest_air_component = air_seg_labeled == np.argmax(np.bincount(air_seg_labeled.flat)[1:]) + 1
        air_seg_cleaned = air_seg - largest_air_component

        

        # Repeat the process to segment the lungs (which are now the largest air-filled areas in the image)
        lung_seg_labeled = label(air_seg_cleaned, background=0)
        largest_lung_component = lung_seg_labeled == np.argmax(np.bincount(lung_seg_labeled.flat)[1:]) + 1

        # dilate the lung segmentation
        largest_lung_component = binary_dilation(largest_lung_component, iterations=10)
        # fill holes in the lung segmentation
        largest_lung_component = binary_fill_holes(largest_lung_component)

        
        if experimental is True:
            activations_per_slice = np.array([np.sum(largest_lung_component[i, :, :]) for i in range(largest_lung_component.shape[0])])
            derivative = np.diff(activations_per_slice)
            argmax = np.argmax(derivative) - left_adjust
            #remove everything before argmax
            largest_lung_component[:argmax, :, :] = 0

            

        # give the mask the same orientation as the original image
        largest_lung_component = transpose_tensor(largest_lung_component, reverse_map)

        # save the mask
        mask_nii = nib.Nifti1Image(largest_lung_component, case.affine, case.header)
        # save the mask with the same name as the original image
        nib.save(mask_nii, os.path.join(output_path, file))

    print("done")

if __name__ == "__main__":
    path = r"D:\script_di_ale\output\imagesTr"
    output_path = r"D:\script_di_ale\output\lung_extraction"



    experimental = True
    shqiperia = True 
    left_adjust = 5
    right_adjust = 5

    lung_segmentation(path, output_path, experimental, shqiperia, left_adjust, right_adjust)