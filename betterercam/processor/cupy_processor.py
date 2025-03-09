import ctypes
import cupy as cp
from .base import Processor


class CupyProcessor(Processor):
    
    def __init__(self, color_mode):
        
        self.cvtcolor = None
        self.color_mode = color_mode
        if self.color_mode=='BGRA':
            self.color_mode = None
        else:
            # Precompute grayscale weights (do this once)
            self.weights = cp.array([0.114, 0.587, 0.299], dtype=cp.float32)
            rgba_indices = cp.array([2, 1, 0, 3], dtype=cp.int8)
            
            # Define optimized conversions
            self.color_mapping = {  #leaves h,w alone but modifies color channels
                #should use views(slices ::-1 instead of [2,1,0,3]) instead of "advanced indexing" which creates copies
                "RGB": lambda img: img[:, :, 2::-1],
                "RGBA": lambda img: img.take(rgba_indices, axis=-1),#cant really be avoided for this one i think but uses cupy funcs
                "BGR": lambda img: img[:, :, :3],
                "GRAY": self.grayscale_optimized,
            }
            
            self.cvtcolor = self.color_mapping[color_mode]
            
    def grayscale_optimized(self, img):
        # Optimized grayscale conversion using matrix multiplication
        gray = cp.dot(img[..., :3], self.weights)
        return gray[..., cp.newaxis]  # Add channel dimension without reshaping
    
    def process(self, rect, width, height, region, rotation_angle):
        pitch = int(rect.Pitch)
        pitch_pixels = pitch // 4  # 4 bytes per BGRA pixel
        
        # Calculate offset and dimensions based on rotation
        if rotation_angle in (0, 180):
            offset = (region[1] if rotation_angle == 0 else height - region[3]) * pitch
            target_height = region[3] - region[1]
            target_width = region[2] - region[0]
        else:
            offset = (region[0] if rotation_angle == 270 else width - region[2]) * pitch
            target_width = region[3] - region[1]
            target_height = region[2] - region[0]
        
        # Create buffer and image view
        buffer = (ctypes.c_char * (pitch * target_height)).from_address(
            ctypes.addressof(rect.pBits.contents) + offset
        )
        image = cp.frombuffer(buffer, dtype=cp.uint8).reshape(target_height, pitch_pixels, 4)
        
        # Apply color conversion
        if self.cvtcolor is not None:
            image = self.cvtcolor(image)
        
        # Apply rotation
        if rotation_angle == 90:
            image = cp.rot90(image, axes=(1, 0))  # Swap axes for 90/270
        elif rotation_angle == 180:
            image = cp.rot90(image, k=2)
        elif rotation_angle == 270:
            image = cp.rot90(image, axes=(0, 1))
        cropped_image = image[:target_height, :target_width, :]
        # Crop to target dimensions (handles pitch mismatch)
        
        #contiguous array is slower but allows for zero copy transfers for other things like pytorch
        return cp.ascontiguousarray(cropped_image)