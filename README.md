# sparse-conv
**Sparse Convolution and Deconvolution Class**

**Features:**

- Implement SparseConv2d() and SparseConvTranspose2d() to handle sparse input tensors (dense tensors with many zeros).
- The implementation skips computations for zero elements to reduce FLOPs.
- In SparseConvTranspose2d, the skipped computations are padded correctly to maintain output shape integrity.
- The function signatures and arguments match PyTorch’s nn.Conv2d() and nn.ConvTranspose2d().
- The I/O tensors support autograd, allowing proper gradient computation and backpropagation.


**Usage:**
```
from sparse_conv import SparseConv2d, SparseConvTranspose2d

conv = SparseConv2d(in_channels, out_channels, kernel_size)
deconv = SparseConvTranspose2d(in_channels, out_channels, kernel_size)
```
