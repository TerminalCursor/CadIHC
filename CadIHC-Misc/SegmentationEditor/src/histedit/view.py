import sys
import numpy as np
import cv2
import tifffile
from matplotlib import pyplot as plt

def point_dist(p1, p2):
    return (p1.x - p2.x)**2 + (p1.y - p2.y)**2

def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <image_file> <segmentation_file>")
        print("Example: python main.py data/HT-B5.tif data/HT-B5_seg.npy")
        return
        
    img_path = sys.argv[1]
    seg_path = sys.argv[2]
    
    # Load Image
    print(f"Loading image {img_path}...")
    if img_path.endswith('.tif') or img_path.endswith('.tiff'):
        img_data = tifffile.imread(img_path)
        if len(img_data.shape) == 3 and img_data.shape[0] in [1, 3, 4]:
            img_data = np.transpose(img_data, (1, 2, 0))
    else:
        img_data = cv2.imread(img_path)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        
    height, width = img_data.shape[:2]
    print(f"Image dimensions: {width}x{height}")
    
    # Load Segmentation
    print(f"Loading segmentation {seg_path}...")
    if seg_path.endswith('.npy'):
        seg_data = np.load(seg_path, allow_pickle=True)
        if seg_data.shape == (): # It's a dict
            seg_data = seg_data.item()['masks']
    else: # bin file
        seg_data = np.fromfile(seg_path, dtype=np.uint16)
        seg_data = seg_data.reshape((height, width))

    plt.imshow(img_data)
    plt.imshow(np.ma.masked_equal(seg_data, 0), alpha=0.5)
    plt.savefig("output.png")

if __name__ == "__main__":
    main()
