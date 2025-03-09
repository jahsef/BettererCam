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
        bytes_per_pixel = 4
        pitch_pixels = pitch // bytes_per_pixel

        # Transform region coordinates based on rotation angle
        x1, y1, x2, y2 = region
        if rotation_angle == 0:
            x_start, y_start = x1, y1
            x_end, y_end = x2, y2
        elif rotation_angle == 90:
            x_start = y1
            y_start = width - x2
            x_end = y2
            y_end = width - x1
        elif rotation_angle == 180:
            x_start = width - x2
            y_start = height - y2
            x_end = width - x1
            y_end = height - y1
        elif rotation_angle == 270:
            x_start = height - y2
            y_start = x1
            x_end = height - y1
            y_end = x2
        else:
            raise ValueError("Unsupported rotation angle")

        # Calculate correct offset and dimensions
        offset = y_start * pitch + x_start * bytes_per_pixel
        target_width = x_end - x_start
        target_height = y_end - y_start

        # Create buffer and image view
        buffer_size = target_height * pitch
        buffer = (ctypes.c_char * buffer_size).from_address(
            ctypes.addressof(rect.pBits.contents) + offset
        )
        image = cp.frombuffer(buffer, dtype=cp.uint8).reshape(target_height, pitch_pixels, 4)

        # Apply color conversion
        if self.cvtcolor is not None:
            image = self.cvtcolor(image)

        # Apply rotation (if needed for display)
        if rotation_angle == 90:
            image = cp.rot90(image, axes=(1, 0))
        elif rotation_angle == 180:
            image = cp.rot90(image, k=2)
        elif rotation_angle == 270:
            image = cp.rot90(image, axes=(0, 1))

        # Crop to target dimensions
        cropped_image = image[:target_height, :target_width, :]

        return cp.ascontiguousarray(cropped_image)
        # return cropped_image