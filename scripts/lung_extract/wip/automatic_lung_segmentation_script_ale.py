import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from scipy.ndimage import measurements, center_of_mass, binary_dilation, zoom, binary_fill_holes
import nibabel as nib
import os
import argparse

default__threshold = -320
default_labels = 3
default_dilation_iterations = 5
default_remove_output = True
default_verbose = False

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

def mask_and_label(img, threshold):
    mask = img < threshold
    mask = np.vectorize(clear_border, signature='(m,n)->(m,n)')(mask)
    mask_labeled = np.vectorize(label, signature='(m,n)->(m,n)')(mask)
    return mask_labeled

def keep_top_n_labels(slice, n = 5):
    new_slice = np.zeros_like(slice)
    rps = regionprops(slice)
    areas = [rp.area for rp in rps]
    
    sorted_indices = np.argsort(areas)[::-1]
    for index in sorted_indices[:n]:
        new_slice[tuple(rps[index].coords.T)] = index + 1 # tuple(rps[index].coords.T) converts the coordinates to the right format
    return new_slice

def remove_trachea(slice, threshold = 0.007):
    new_slice = slice.copy()
    labels = label(slice, connectivity=1, background=0)
    rps = regionprops(labels)
    labels_areas = np.array([rp.area for rp in rps])
    total_area_of_slice = slice.shape[0] * slice.shape[1]
    indices_to_remove = np.where(labels_areas / total_area_of_slice < threshold)[0]
    for index in indices_to_remove:
        new_slice[tuple(rps[index].coords.T)] = 0
    return new_slice


def lung_segmentation(input_path, output_path,
                        threshold = default__threshold,
                        labels = default_labels,
                        dilation_iterations = default_dilation_iterations
                      ):
    ct = nib.load(input_path)
    img = ct.get_fdata()
    min_img, max_img = np.min(img), np.max(img)

    print(f"shape: {img.shape}")
    print(f"min: {min_img}, max: {max_img}")
    #minmaxscale the image
    img = (img - min_img) / (max_img - min_img)
    #threshold is -320 in the interval -1024, 3071, so we need to scale it, here though we are using the minmaxscaled image
    threshold = (threshold - (-1024)) / (3071 - (-1024))
    

    # need to put it in the right orientation
    axes_map, reverse_map = get_axes_maps(img)
    img = transpose_tensor(img, axes_map)

    mask_labeled = mask_and_label(img, threshold)
    mask_labeled = keep_top_n_labels(mask_labeled, labels)

    mask = mask_labeled != 0
    #fill holes in mask
    mask = np.vectorize(binary_fill_holes, signature='(m,n)->(m,n)')(mask)
    mask = np.vectorize(remove_trachea, signature='(m,n)->(m,n)')(mask)
    mask = binary_dilation(mask, iterations=dilation_iterations)

    mask = transpose_tensor(mask, reverse_map)
    mask_nii = nib.Nifti1Image(mask, ct.affine, ct.header)
    nib.save(mask_nii, output_path)

#function for parser
def parse_args():
    parser = argparse.ArgumentParser(description='Perform lung segmentation on NIfTI images.')
    parser.add_argument('input_path', type=str, help='Input directory containing NIfTI images')
    parser.add_argument('output_path', type=str, help='Output directory to save segmented images')
    parser.add_argument('-t', '--threshold', type=float, default=default__threshold, help='Threshold value for segmentation')
    parser.add_argument('-l', '--labels', type=int, default=default_labels, help='Number of top labels to retain')
    parser.add_argument('-d', '--dilation_iterations', type=int, default=default_dilation_iterations, help='Number of dilation iterations')
    parser.add_argument('-r', '--remove_output', type=bool, default=default_remove_output, help='Remove output file if it already exists')
    parser.add_argument('-v', '--verbose', type=bool, default=default_verbose, help='Print debug information')

    args = parser.parse_args()
    return args

def conditioal_print(message, verbose):
    if verbose is True:
        print(message)


if __name__ == "__main__":
    args = parse_args()
    input_path = args.input_path
    output_path = args.output_path
    threshold = args.threshold
    labels = args.labels
    dilation_iterations = args.dilation_iterations
    remove_output = args.remove_output
    verbose = args.verbose
  
    conditioal_print(f"input_path: {input_path}", verbose)
    conditioal_print(f"output_path: {output_path}", verbose)
    conditioal_print(f"threshold: {threshold}", verbose)
    conditioal_print(f"labels: {labels}", verbose)
    conditioal_print(f"dilation_iterations: {dilation_iterations}", verbose)
    conditioal_print(f"remove_output: {remove_output}", verbose)
    conditioal_print(f"verbose: {verbose}", verbose)


    files = os.listdir(input_path)
    for file in files:
        input_file = os.path.join(input_path, file)
        output_file = os.path.join(output_path, file)
        # see if output file already exists
        if os.path.exists(output_file):
            if remove_output is True:
                print(f"deleting {output_file}")
                os.remove(output_file)
            else:
                conditioal_print(f"skipping {file} as output file already exists", verbose)
                continue

        print(f"processing {file}")
        conditioal_print(f"output path for {file}: {output_file}", verbose)
        #lung_segmentation(input_file, output_file)
        try:
            lung_segmentation(input_file, output_file)
        except Exception as e:
            print(f"error processing {file}: {e}")
            continue
# example call

# python automatic_lung_segmentation_script_ale.py C:\Users\aless\Desktop\git\NSCLC\data\nnUNet_raw\Dataset001_Apm\imagesTr C:\Users\aless\Desktop\git\NSCLC\data\nnUNet_raw\Dataset001_Apm\segment_lung

# input D:\nsclc\data\nnUNet_raw\Dataset002_Rid\imagesTr
# output D:\nsclc\data\nnUNet_raw\Dataset002_Rid\segment_lung
# python automatic_lung_segmentation_script_ale.py D:\nsclc\data\nnUNet_raw\Dataset002_Rid\imagesTr D:\nsclc\data\nnUNet_raw\Dataset002_Rid\segment_lung

