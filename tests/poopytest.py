import cupy as cp
import time
# Create a large BGRA image (4K resolution)
img = cp.random.randint(0, 256, (2160, 3840, 512), dtype=cp.uint8)

# Advanced indexing (copy)
start = time.perf_counter_ns()
img[:, :, [2, 1, 0]]  # ~3.2 ms (slower due to memory allocation)
print((time.perf_counter_ns()- start)/1e6)
# View-based approach (no copy)

start = time.perf_counter_ns()
img[:, :, 2::-1]       # ~0.1 ms (30x faster)

print((time.perf_counter_ns()- start)/1e6)