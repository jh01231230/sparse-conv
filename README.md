# sparse-conv
**Sparse Convolution and Deconvolution Class**

**Features:**

- Implement SparseConv2d() and SparseConvTranspose2d() to handle sparse input tensors (dense tensors with many zeros).
- The implementation skips computations for zero elements to reduce FLOPs.
- In SparseConvTranspose2d, the skipped computations are padded correctly to maintain output shape integrity.
- The function signatures and arguments match PyTorch’s nn.Conv2d() and nn.ConvTranspose2d().
- The I/O tensors support autograd, allowing proper gradient computation and backpropagation.

1. **Adaptive Execution Strategy:**
- Added a sparsity threshold (70%) to automatically use standard convolution for less sparse inputs
- This prevents performance loss when the input isn't sparse enough to benefit from sparse operations
2. **GPU-Optimized Operations:**
- Minimized CPU-GPU transfers by keeping operations on the GPU
- Improved batch processing by handling non-zero batches collectively
- Reduced memory allocations and improved tensor reuse
3. **Efficient Gradient Calculation:**
- Optimized backward pass implementation for both weight and input gradients
- Used PyTorch's optimized CUDA kernels when possible instead of manual element-wise operations
- Added specialized handling for grouped convolutions
4. **Structured Sparsity Support:**
- Added a convert_to_structured_sparse function that transforms random sparsity into structured block sparsity
- This better aligns with how GPUs process data in warps/blocks
5. **Performance Benchmarking:**
- Added a benchmark_conv function to compare sparse vs dense convolution performance
- Includes proper CUDA synchronization and timing
6. **Improved FLOP Counting:**
- Enhanced the FLOP calculation with batch-level granularity
- Added detailed reporting option to understand performance characteristics


**Usage:**
```
from sparse_conv import SparseConv2d, SparseConvTranspose2d

conv = SparseConv2d(in_channels, out_channels, kernel_size)
deconv = SparseConvTranspose2d(in_channels, out_channels, kernel_size)
```
