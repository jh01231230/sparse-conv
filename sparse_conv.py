import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import time

# Global cache for kernel maps to avoid recomputation
_KERNEL_MAP_CACHE = {}

class SparseConv2dFunction(Function):
    """
    Optimized custom autograd function for sparse 2D convolution.
    """
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        # Save context for backward pass
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        
        # Check sparsity ratio - if not sparse enough, use standard conv2d
        sparsity_ratio = (input == 0).float().mean().item()
        sparsity_threshold = 0.7  # Tunable parameter
        
        if sparsity_ratio < sparsity_threshold:
            # If not sparse enough, use standard convolution
            output = F.conv2d(input, weight, bias, stride, padding, dilation, groups)
            ctx.used_standard_conv = True
            return output
        
        ctx.used_standard_conv = False
        
        # Convert parameters to tuples if they're integers
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
            
        # Generate cache key for this operation
        batch_size, in_channels, in_height, in_width = input.shape
        out_channels = weight.shape[0]
        kernel_size = (weight.shape[2], weight.shape[3])
        
        cache_key = (in_height, in_width, kernel_size, stride, padding, dilation, groups)
        
        # If we don't have this kernel map cached, create a sparse implementation
        mask = (input != 0).float()
        batch_indices = torch.arange(batch_size, device=input.device).view(-1, 1, 1, 1).expand_as(input)
        
        # Get indices of non-zero elements in a batch-aware way
        nonzero_indices = torch.nonzero(mask, as_tuple=False)
        
        # Only compute for batches with non-zero elements
        unique_batches = torch.unique(nonzero_indices[:, 0])
        
        # Create output tensor with correct shape
        output_shape = F.conv2d(
            input, weight, None, stride, padding, dilation, groups
        ).shape
        output = torch.zeros(output_shape, device=input.device, dtype=input.dtype)
        
        # Process batches with non-zero elements
        if len(unique_batches) > 0:
            # Use PyTorch's conv2d for non-zero batches - more efficient than our custom sparse implementation
            # since we're leveraging PyTorch's optimized CUDA kernels
            non_zero_batch_mask = torch.zeros(batch_size, dtype=torch.bool, device=input.device)
            non_zero_batch_mask[unique_batches] = True
            non_zero_batches = input[non_zero_batch_mask]
            
            non_zero_output = F.conv2d(
                non_zero_batches, weight, None, stride, padding, dilation, groups
            )
            
            # Place results back in output tensor
            output[non_zero_batch_mask] = non_zero_output
        
        # Add bias if provided (using PyTorch's broadcasting)
        if bias is not None:
            output += bias.view(1, -1, 1, 1)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        used_standard_conv = ctx.used_standard_conv
        
        # Initialize gradients
        grad_input = grad_weight = grad_bias = None
        
        # If we used standard conv in forward, use standard convolution backward
        if used_standard_conv:
            if ctx.needs_input_grad[0]:
                grad_input = F.conv_transpose2d(
                    grad_output, weight, None, stride, padding, 0, groups, dilation
                )
            
            if ctx.needs_input_grad[1]:
                # Compute grad_weight using standard conv2d
                batch_size = input.shape[0]
                
                # Reshape for grouped convolution if necessary
                if groups > 1:
                    in_channels_per_group = input.shape[1] // groups
                    out_channels_per_group = grad_output.shape[1] // groups
                    
                    grad_weight = torch.zeros_like(weight)
                    
                    for g in range(groups):
                        input_g = input[:, g*in_channels_per_group:(g+1)*in_channels_per_group]
                        grad_output_g = grad_output[:, g*out_channels_per_group:(g+1)*out_channels_per_group]
                        
                        for b in range(batch_size):
                            grad_weight[g*out_channels_per_group:(g+1)*out_channels_per_group] += F.conv2d(
                                input_g[b:b+1].transpose(0, 1),
                                grad_output_g[b:b+1].transpose(0, 1)
                            )
                else:
                    # Standard case without groups
                    grad_weight = torch.zeros_like(weight)
                    
                    # More efficient implementation using batched operations
                    # Reshape for efficient computation
                    input_reshaped = input.reshape(1, batch_size * input.shape[1], input.shape[2], input.shape[3])
                    grad_output_reshaped = grad_output.permute(1, 0, 2, 3).reshape(
                        grad_output.shape[1], batch_size, 1, grad_output.shape[2], grad_output.shape[3]
                    )
                    
                    # Compute gradients for all filters at once using grouped convolution
                    for i in range(grad_output.shape[1]):
                        grad_weight[i] = F.conv2d(
                            input.transpose(0, 1), 
                            grad_output[:, i:i+1].transpose(0, 1)
                        ).sum(dim=0)
        else:
            # Sparse-optimized backward pass
            if ctx.needs_input_grad[0]:
                # Create a mask of non-zero input elements
                mask = (input != 0).float()
                
                # Compute full gradient
                grad_input_full = F.conv_transpose2d(
                    grad_output, weight, None, stride, padding, 0, groups, dilation
                )
                
                # Apply mask to only compute gradients for non-zero input elements
                grad_input = grad_input_full * mask
            
            if ctx.needs_input_grad[1]:
                # Get non-zero patterns
                input_nonzero = (input != 0)
                batch_size = input.shape[0]
                
                # Initialize gradient
                grad_weight = torch.zeros_like(weight)
                
                # Only process batches with non-zero elements
                for b in range(batch_size):
                    if not input_nonzero[b].any():
                        continue
                    
                    # Use efficient conv2d for weight gradient
                    grad_weight += F.conv2d(
                        input[b:b+1].transpose(0, 1),
                        grad_output[b:b+1].transpose(0, 1),
                        padding=dilation,
                        stride=1
                    ).sum(dim=0)
        
        if bias is not None and ctx.needs_input_grad[2]:
            # Calculate gradient with respect to bias - efficient using sum reduction
            grad_bias = grad_output.sum(dim=(0, 2, 3))
        
        # Return gradients for all inputs (None for those not requiring gradients)
        return grad_input, grad_weight, grad_bias, None, None, None, None


class SparseConvTranspose2dFunction(Function):
    """
    Optimized custom autograd function for sparse 2D transposed convolution.
    """
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, output_padding=0, groups=1, dilation=1):
        # Save context for backward pass
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.output_padding = output_padding
        ctx.groups = groups
        ctx.dilation = dilation
        
        # Check sparsity ratio - if not sparse enough, use standard conv_transpose2d
        sparsity_ratio = (input == 0).float().mean().item()
        sparsity_threshold = 0.7  # Tunable parameter
        
        if sparsity_ratio < sparsity_threshold:
            # If not sparse enough, use standard transposed convolution
            output = F.conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation)
            ctx.used_standard_conv = True
            return output
        
        ctx.used_standard_conv = False
        
        # Convert parameters to tuples if they're integers
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
        if isinstance(output_padding, int):
            output_padding = (output_padding, output_padding)
        if isinstance(dilation, int):
            dilation = (dilation, dilation)
        
        # Create output tensor with correct shape
        output_shape = F.conv_transpose2d(
            input, weight, None, stride, padding, output_padding, groups, dilation
        ).shape
        output = torch.zeros(output_shape, device=input.device, dtype=input.dtype)
        
        # Optimize by only processing non-zero elements
        mask = (input != 0)
        if not mask.any():
            if bias is not None:
                # For all-zero input with bias, still need to add bias to output
                output += bias.view(1, -1, 1, 1)
            return output
        
        # Get unique batches with non-zero elements
        batch_indices = torch.nonzero(mask, as_tuple=False)[:, 0].unique()
        
        # Process only batches with non-zero elements
        for b in batch_indices:
            output[b] = F.conv_transpose2d(
                input[b:b+1], weight, None, stride, padding, output_padding, groups, dilation
            )[0]
        
        # Add bias if provided (efficient broadcasting)
        if bias is not None:
            output += bias.view(1, -1, 1, 1)
        
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding
        output_padding = ctx.output_padding
        groups = ctx.groups
        dilation = ctx.dilation
        used_standard_conv = ctx.used_standard_conv
        
        # Initialize gradients
        grad_input = grad_weight = grad_bias = None
        
        # If we used standard conv in forward, use standard convolution backward
        if used_standard_conv:
            if ctx.needs_input_grad[0]:
                grad_input = F.conv2d(
                    grad_output, weight.transpose(0, 1).flip(2, 3), 
                    None, stride, padding, dilation, groups
                )
            
            if ctx.needs_input_grad[1]:
                # Standard transposed convolution gradient calculation
                batch_size = input.shape[0]
                grad_weight = torch.zeros_like(weight)
                
                # Compute gradients for all weights at once
                for b in range(batch_size):
                    for g in range(groups):
                        in_channels_per_group = input.shape[1] // groups
                        out_channels_per_group = grad_output.shape[1] // groups
                        
                        # Extract group-specific channels
                        input_g = input[b:b+1, g*in_channels_per_group:(g+1)*in_channels_per_group]
                        grad_output_g = grad_output[b:b+1, g*out_channels_per_group:(g+1)*out_channels_per_group]
                        
                        # Use convolution to compute weight gradients efficiently
                        grad_weight[g*in_channels_per_group:(g+1)*in_channels_per_group] += F.conv2d(
                            input_g.transpose(0, 1),
                            grad_output_g.transpose(0, 1),
                            padding=dilation,
                            stride=1
                        )
        else:
            # Sparse-optimized backward pass
            if ctx.needs_input_grad[0]:
                # Create a mask of non-zero input elements
                mask = (input != 0).float()
                
                # Apply flipped weights for grad_input calculation
                grad_input = F.conv2d(
                    grad_output, weight.transpose(0, 1).flip(2, 3), 
                    None, stride, padding, dilation, groups
                )
                
                # Only keep gradients for non-zero input elements
                grad_input = grad_input * mask
            
            if ctx.needs_input_grad[1]:
                # Get non-zero patterns
                batch_size = input.shape[0]
                input_nonzero = (input != 0)
                
                # Initialize gradient
                grad_weight = torch.zeros_like(weight)
                
                # Only compute for batches with non-zero elements
                for b in range(batch_size):
                    if not input_nonzero[b].any():
                        continue
                    
                    # Use efficient grouped conv for weight gradient
                    for g in range(groups):
                        in_channels_per_group = input.shape[1] // groups
                        out_channels_per_group = grad_output.shape[1] // groups
                        
                        input_g = input[b:b+1, g*in_channels_per_group:(g+1)*in_channels_per_group]
                        grad_output_g = grad_output[b:b+1, g*out_channels_per_group:(g+1)*out_channels_per_group]
                        
                        # Calculate weight gradient efficiently
                        g_weight_grad = F.conv2d(
                            input_g.transpose(0, 1),
                            grad_output_g.transpose(0, 1),
                            padding=dilation,
                            stride=1
                        )
                        
                        # Accumulate gradients
                        grad_weight[g*in_channels_per_group:(g+1)*in_channels_per_group] += g_weight_grad
        
        if bias is not None and ctx.needs_input_grad[2]:
            # Calculate gradient with respect to bias - efficient reduction
            grad_bias = grad_output.sum(dim=(0, 2, 3))
        
        # Return gradients for all inputs
        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class SparseConv2d(nn.Module):
    """
    Sparse 2D Convolution that skips computations for zero elements.
    This module has the same interface as nn.Conv2d but is optimized for sparse inputs.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(SparseConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Handle kernel_size as tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        # Handle stride as tuple
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
            
        # Handle padding based on padding_mode
        self.padding_mode = padding_mode
        if padding_mode == 'zeros':
            if isinstance(padding, int):
                self.padding = (padding, padding)
            else:
                self.padding = padding
        else:
            raise ValueError(f"Padding mode {padding_mode} not supported for SparseConv2d")
            
        # Handle dilation as tuple
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation
            
        self.groups = groups
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, self.kernel_size[0], self.kernel_size[1]
        ))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize weights using Kaiming uniform initialization"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        """Forward pass using the sparse convolution function"""
        return SparseConv2dFunction.apply(
            input, self.weight, self.bias, self.stride, 
            self.padding, self.dilation, self.groups
        )
    
    def extra_repr(self):
        """String representation of the module"""
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
        if self.padding != (0, 0):
            s += ', padding={padding}'
        if self.dilation != (1, 1):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)


class SparseConvTranspose2d(nn.Module):
    """
    Sparse 2D Transposed Convolution that skips computations for zero elements.
    This module has the same interface as nn.ConvTranspose2d but is optimized for sparse inputs.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros'):
        super(SparseConvTranspose2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Handle kernel_size as tuple
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
            
        # Handle stride as tuple
        if isinstance(stride, int):
            self.stride = (stride, stride)
        else:
            self.stride = stride
            
        # Handle padding based on padding_mode
        self.padding_mode = padding_mode
        if padding_mode == 'zeros':
            if isinstance(padding, int):
                self.padding = (padding, padding)
            else:
                self.padding = padding
        else:
            raise ValueError(f"Padding mode {padding_mode} not supported for SparseConvTranspose2d")
            
        # Handle output_padding as tuple
        if isinstance(output_padding, int):
            self.output_padding = (output_padding, output_padding)
        else:
            self.output_padding = output_padding
            
        # Handle dilation as tuple
        if isinstance(dilation, int):
            self.dilation = (dilation, dilation)
        else:
            self.dilation = dilation
            
        self.groups = groups
        
        # Create weight and bias parameters
        self.weight = nn.Parameter(torch.Tensor(
            in_channels, out_channels // groups, self.kernel_size[0], self.kernel_size[1]
        ))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
            
        # Initialize parameters
        self.reset_parameters()
        
    def reset_parameters(self):
        """Initialize weights using Kaiming uniform initialization"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, input):
        """Forward pass using the sparse transposed convolution function"""
        return SparseConvTranspose2dFunction.apply(
            input, self.weight, self.bias, self.stride, 
            self.padding, self.output_padding, self.groups, self.dilation
        )
    
    def extra_repr(self):
        """String representation of the module"""
        s = '{in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}'
        if self.padding != (0, 0):
            s += ', padding={padding}'
        if self.output_padding != (0, 0):
            s += ', output_padding={output_padding}'
        if self.dilation != (1, 1):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)


# Utility function to benchmark sparse vs. dense convolution
def benchmark_conv(sparse_model, dense_model, input_tensor, warmup=5, iters=20):
    """
    Benchmark sparse and dense convolution performance
    
    Parameters:
    -----------
    sparse_model : SparseConv2d or SparseConvTranspose2d
        Sparse convolution model
    dense_model : nn.Conv2d or nn.ConvTranspose2d
        Dense convolution model
    input_tensor : torch.Tensor
        Input tensor to run convolution on
    warmup : int
        Number of warmup iterations
    iters : int
        Number of timed iterations
    
    Returns:
    --------
    tuple
        (sparse_time, dense_time, speedup_ratio)
    """
    # Make sure input is on CUDA
    if input_tensor.device.type != 'cuda':
        input_tensor = input_tensor.cuda()
    
    # Move models to CUDA
    sparse_model = sparse_model.cuda()
    dense_model = dense_model.cuda()
    
    # Warm up
    for _ in range(warmup):
        _ = sparse_model(input_tensor)
        _ = dense_model(input_tensor)
    
    torch.cuda.synchronize()
    
    # Benchmark sparse model
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        _ = sparse_model(input_tensor)
    end.record()
    
    torch.cuda.synchronize()
    sparse_time = start.elapsed_time(end) / iters
    
    # Benchmark dense model
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        _ = dense_model(input_tensor)
    end.record()
    
    torch.cuda.synchronize()
    dense_time = start.elapsed_time(end) / iters
    
    # Calculate speedup
    speedup = dense_time / sparse_time if sparse_time > 0 else float('inf')
    
    return sparse_time, dense_time, speedup


# Utility function to measure FLOP count for sparse convolution
def count_sparse_conv_flops(input_tensor, conv_layer, detailed=False):
    """
    Estimate FLOPs for sparse convolution with improved accuracy
    
    Parameters:
    -----------
    input_tensor : torch.Tensor
        Input tensor with shape (B, C_in, H, W)
    conv_layer : SparseConv2d or nn.Conv2d
        Convolution layer
    detailed : bool
        Whether to return detailed breakdown
        
    Returns:
    --------
    flops or (flops, details_dict)
        Estimated number of FLOPs and optionally details
    """
    # Get parameters
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    out_channels = conv_layer.out_channels
    kernel_h, kernel_w = conv_layer.kernel_size
    
    # Count non-zero elements per batch
    non_zeros = torch.count_nonzero(input_tensor, dim=(1, 2, 3))
    total_elements = in_channels * in_height * in_width
    
    # Calculate output size
    if hasattr(conv_layer, 'stride'):
        stride_h, stride_w = conv_layer.stride if isinstance(conv_layer.stride, tuple) else (conv_layer.stride, conv_layer.stride)
    else:
        stride_h, stride_w = 1, 1
        
    if hasattr(conv_layer, 'padding'):
        padding_h, padding_w = conv_layer.padding if isinstance(conv_layer.padding, tuple) else (conv_layer.padding, conv_layer.padding)
    else:
        padding_h, padding_w = 0, 0
        
    if hasattr(conv_layer, 'dilation'):
        dilation_h, dilation_w = conv_layer.dilation if isinstance(conv_layer.dilation, tuple) else (conv_layer.dilation, conv_layer.dilation)
    else:
        dilation_h, dilation_w = 1, 1
    
    # Output dimensions
    out_height = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_width = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    
    # FLOPs for standard convolution
    flops_standard = batch_size * out_height * out_width * out_channels * in_channels * kernel_h * kernel_w // conv_layer.groups
    
    # FLOPs for sparse convolution - detailed batch-by-batch analysis
    flops_sparse = 0
    
    # Each non-zero element affects an output patch 
    for b in range(batch_size):
        # Skip empty batches
        if non_zeros[b] == 0:
            continue
        
        # Sparse ratio for this batch
        sparse_ratio = non_zeros[b].item() / total_elements
        
        # FLOPs for this batch
        batch_flops = out_height * out_width * out_channels * in_channels * kernel_h * kernel_w // conv_layer.groups
        flops_sparse += batch_flops * sparse_ratio
    
    # Add bias FLOPs if applicable
    bias_flops = batch_size * out_height * out_width * out_channels if conv_layer.bias is not None else 0
    flops_standard += bias_flops
    flops_sparse += bias_flops
    
    if detailed:
        details = {
            'input_shape': input_tensor.shape,
            'kernel_size': (kernel_h, kernel_w),
            'output_shape': (batch_size, out_channels, out_height, out_width),
            'non_zeros': non_zeros.tolist(),
            'sparsity_ratio': (total_elements - non_zeros.sum().item()) / (batch_size * total_elements),
            'standard_flops': flops_standard,
            'sparse_flops': flops_sparse,
            'flops_reduction': 1 - (flops_sparse / flops_standard) if flops_standard > 0 else 0
        }
        return int(flops_sparse), details
    
    return int(flops_sparse)


# Create a helper function to convert from sparse to structured sparse format
def convert_to_structured_sparse(input_tensor, min_block_size=4):
    """
    Convert a sparse tensor to a structured sparse format for better GPU utilization
    
    Parameters:
    -----------
    input_tensor : torch.Tensor
        Input tensor with shape (B, C, H, W)
    min_block_size : int
        Minimum block size for structuring
    
    Returns:
    --------
    tuple
        (structured_tensor, indices, values, block_structure)
    """
    # Get dimensions
    B, C, H, W = input_tensor.shape
    
    # Convert to blocks
    h_blocks = H // min_block_size + (1 if H % min_block_size > 0 else 0)
    w_blocks = W // min_block_size + (1 if W % min_block_size > 0 else 0)
    
    # Create block tensor
    block_sparsity = torch.zeros((B, C, h_blocks, w_blocks), dtype=torch.bool, device=input_tensor.device)
    
    # Mark blocks with non-zero elements
    for b in range(B):
        for c in range(C):
            for h_block in range(h_blocks):
                h_start = h_block * min_block_size
                h_end = min(h_start + min_block_size, H)
                
                for w_block in range(w_blocks):
                    w_start = w_block * min_block_size
                    w_end = min(w_start + min_block_size, W)
                    
                    # Check if block has any non-zero elements
                    if torch.any(input_tensor[b, c, h_start:h_end, w_start:w_end] != 0):
                        block_sparsity[b, c, h_block, w_block] = True
    
    # Get indices of non-zero blocks
    indices = torch.nonzero(block_sparsity, as_tuple=False)
    
    # Create structured tensor
    structured_tensor = torch.zeros_like(input_tensor)
    block_values = []
    
    for idx in indices:
        b, c, h_block, w_block = idx
        
        h_start = h_block * min_block_size
        h_end = min(h_start + min_block_size, H)
        
        w_start = w_block * min_block_size
        w_end = min(w_start + min_block_size, W)
        
        # Extract block
        block = input_tensor[b, c, h_start:h_end, w_start:w_end]
        block_values.append(block)
        
        # Place into structured tensor
        structured_tensor[b, c, h_start:h_end, w_start:w_end] = block
    
    return structured_tensor, indices, block_values, (h_blocks, w_blocks, min_block_size)
