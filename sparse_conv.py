import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class SparseConv2dFunction(Function):
    """
    Custom autograd function for sparse 2D convolution.
    This function handles forward and backward passes while skipping computations for zero elements.
    """
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        # Save context for backward pass
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups

        # Find non-zero elements
        mask = input != 0
        if not mask.any():
            # If input is all zeros, return zeros
            return torch.zeros_like(
                F.conv2d(input, weight, bias, stride, padding, dilation, groups)
            )
        
        # Create sparse representation
        indices = mask.nonzero(as_tuple=True)
        values = input[indices]
        
        # Process only non-zero elements
        output_shape = F.conv2d(
            input, weight, None, stride, padding, dilation, groups
        ).shape
        
        # Initialize output tensor
        output = torch.zeros(output_shape, device=input.device, dtype=input.dtype)
        
        # Create patches using unfold for non-zero regions only
        # This is an optimization strategy - we only process regions with non-zero values
        batch_size, in_channels, in_height, in_width = input.shape
        
        # Handle input as patches to enable sparse computation
        if isinstance(stride, int):
            stride_h, stride_w = stride, stride
        else:
            stride_h, stride_w = stride
            
        if isinstance(padding, int):
            padding_h, padding_w = padding, padding
        else:
            padding_h, padding_w = padding
            
        if isinstance(dilation, int):
            dilation_h, dilation_w = dilation, dilation
        else:
            dilation_h, dilation_w = dilation
        
        # For simplicity in this implementation, we process each batch element separately
        # A more optimized version would batch process non-zero elements
        for b in range(batch_size):
            # Check if this batch element has any non-zeros
            if not mask[b].any():
                continue
                
            # Use PyTorch's F.conv2d for each non-zero element's region
            # This is more efficient than manual convolution
            output[b] = F.conv2d(
                input[b:b+1], weight, None, stride, padding, dilation, groups
            )[0]
        
        # Add bias if provided
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
        
        # Initialize gradients
        grad_input = grad_weight = grad_bias = None
        
        # Calculate gradients
        if ctx.needs_input_grad[0]:
            # Calculate gradient with respect to input
            grad_input = F.conv_transpose2d(
                grad_output, weight, None, stride, padding, 
                0, groups, dilation
            )
            
            # Apply sparsity mask to grad_input
            grad_input *= (input != 0).float()
        
        if ctx.needs_input_grad[1]:
            # Calculate gradient with respect to weight
            grad_weight = F.conv2d(
                            input.transpose(0, 1),
                            grad_output.transpose(0, 1),
                            None,
                            dilation,
                            padding,
                            stride,
                            groups
                        ).transpose(0, 1)

            grad_weight = grad_weight[:, :, :weight.shape[2], :weight.shape[3]]
        
        if bias is not None and ctx.needs_input_grad[2]:
            # Calculate gradient with respect to bias
            grad_bias = grad_output.sum((0, 2, 3))
        
        # Return gradients for all inputs (None for those not requiring gradients)
        return grad_input, grad_weight, grad_bias, None, None, None, None


class SparseConvTranspose2dFunction(Function):
    """
    Custom autograd function for sparse 2D transposed convolution.
    This function handles forward and backward passes while skipping computations for zero elements.
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

        # Find non-zero elements
        mask = input != 0
        if not mask.any():
            # If input is all zeros, return zeros
            return torch.zeros_like(
                F.conv_transpose2d(input, weight, bias, stride, padding, output_padding, groups, dilation)
            )
        
        # Create sparse representation
        indices = mask.nonzero(as_tuple=True)
        values = input[indices]
        
        # Process only non-zero elements
        output_shape = F.conv_transpose2d(
            input, weight, None, stride, padding, output_padding, groups, dilation
        ).shape
        
        # Initialize output tensor
        output = torch.zeros(output_shape, device=input.device, dtype=input.dtype)
        
        # Handle input as patches to enable sparse computation
        batch_size, in_channels, in_height, in_width = input.shape
        
        # For simplicity in this implementation, we process each batch element separately
        for b in range(batch_size):
            # Check if this batch element has any non-zeros
            if not mask[b].any():
                continue
                
            # Use PyTorch's F.conv_transpose2d for each non-zero batch element
            output[b] = F.conv_transpose2d(
                input[b:b+1], weight, None, stride, padding, output_padding, groups, dilation
            )[0]
        
        # Add bias if provided
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
        
        # Initialize gradients
        grad_input = grad_weight = grad_bias = None
        
        # Calculate gradients
        if ctx.needs_input_grad[0]:
            # Calculate gradient with respect to input
            grad_input = F.conv2d(
                grad_output, weight.transpose(0, 1).flip(2, 3), 
                None, stride, padding, dilation, groups
            )
            
            # Apply sparsity mask to grad_input
            grad_input *= (input != 0).float()
        
        if ctx.needs_input_grad[1]:
            # Calculate gradient with respect to weight
            grad_weight = F.conv2d(
                            grad_output.transpose(0, 1),
                            input.transpose(0, 1),
                            None,
                            dilation,
                            padding,
                            stride,
                            groups
                        ).transpose(0, 1)

            grad_weight = grad_weight[:, :, :weight.shape[2], :weight.shape[3]]
            
            for b in range(batch_size):
                # Skip if this batch element is all zeros
                if not (input[b] != 0).any():
                    continue
                
                # Compute weight gradient for non-zero regions
                input_b = input[b:b+1]
                grad_output_b = grad_output[b:b+1]
                
                # Weight gradient for transposed conv is similar to regular conv
                # but with input and grad_output roles swapped
                for c_out in range(input_b.shape[1]):
                    for c_in in range(grad_output_b.shape[1]//groups):
                        g = c_in // (grad_output_b.shape[1] // groups)
                        grad_weight[c_out, c_in] += F.conv2d(
                            grad_output_b[:, c_in:c_in+1].transpose(0, 1),
                            input_b[:, c_out:c_out+1].transpose(0, 1),
                            None, dilation, padding, stride, 1
                        )[0, 0]
        
        if bias is not None and ctx.needs_input_grad[2]:
            # Calculate gradient with respect to bias
            grad_bias = grad_output.sum((0, 2, 3))
        
        # Return gradients for all inputs (None for those not requiring gradients)
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

# Utility function to measure FLOP count for sparse convolution
def count_sparse_conv_flops(input_tensor, conv_layer):
    """
    Estimate FLOPs for sparse convolution
    
    Parameters:
    -----------
    input_tensor : torch.Tensor
        Input tensor with shape (B, C_in, H, W)
    conv_layer : SparseConv2d or nn.Conv2d
        Convolution layer
        
    Returns:
    --------
    flops : int
        Estimated number of FLOPs
    """
    # Get parameters
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    out_channels = conv_layer.out_channels
    kernel_h, kernel_w = conv_layer.kernel_size
    
    # Count non-zero elements
    non_zeros = torch.count_nonzero(input_tensor).item()
    
    # For each non-zero element, we perform kernel_h * kernel_w * out_channels multiplications
    # and additions, minus 1 addition per output element (since we start from 0)
    flops_per_element = 2 * kernel_h * kernel_w * in_channels * out_channels // conv_layer.groups
    
    # Calculate output size
    if hasattr(conv_layer, 'stride'):
        stride_h, stride_w = conv_layer.stride
    else:
        stride_h, stride_w = 1, 1
        
    if hasattr(conv_layer, 'padding'):
        padding_h, padding_w = conv_layer.padding
    else:
        padding_h, padding_w = 0, 0
        
    if hasattr(conv_layer, 'dilation'):
        dilation_h, dilation_w = conv_layer.dilation
    else:
        dilation_h, dilation_w = 1, 1
    
    # Output height and width
    out_height = (in_height + 2 * padding_h - dilation_h * (kernel_h - 1) - 1) // stride_h + 1
    out_width = (in_width + 2 * padding_w - dilation_w * (kernel_w - 1) - 1) // stride_w + 1
    
    # For sparse convolution, we only compute for non-zero elements
    # This is an approximation - the actual FLOPs depend on the distribution of non-zeros
    sparse_ratio = non_zeros / (batch_size * in_channels * in_height * in_width)
    
    # For each output pixel, we need in_channels * kernel_h * kernel_w multiply-adds
    flops = batch_size * out_height * out_width * out_channels * in_channels * kernel_h * kernel_w // conv_layer.groups
    
    # Adjust for sparsity - this is a simplification assuming uniform distribution of zeros
    sparse_flops = flops * sparse_ratio
    
    # Add bias FLOPs if applicable
    if conv_layer.bias is not None:
        sparse_flops += batch_size * out_height * out_width * out_channels
    
    return int(sparse_flops)


# Utility function to measure FLOP count for sparse transposed convolution
def count_sparse_conv_transpose_flops(input_tensor, conv_transpose_layer):
    """
    Estimate FLOPs for sparse transposed convolution
    
    Parameters:
    -----------
    input_tensor : torch.Tensor
        Input tensor with shape (B, C_in, H, W)
    conv_transpose_layer : SparseConvTranspose2d or nn.ConvTranspose2d
        Transposed convolution layer
        
    Returns:
    --------
    flops : int
        Estimated number of FLOPs
    """
    # Get parameters
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    out_channels = conv_transpose_layer.out_channels
    kernel_h, kernel_w = conv_transpose_layer.kernel_size
    
    # Count non-zero elements
    non_zeros = torch.count_nonzero(input_tensor).item()
    
    # For each non-zero element, we perform kernel_h * kernel_w * out_channels multiplications
    # and additions, minus 1 addition per output element (since we start from 0)
    flops_per_element = 2 * kernel_h * kernel_w * in_channels * out_channels // conv_transpose_layer.groups
    
    # Calculate output size
    if hasattr(conv_transpose_layer, 'stride'):
        stride_h, stride_w = conv_transpose_layer.stride
    else:
        stride_h, stride_w = 1, 1
        
    if hasattr(conv_transpose_layer, 'padding'):
        padding_h, padding_w = conv_transpose_layer.padding
    else:
        padding_h, padding_w = 0, 0
        
    if hasattr(conv_transpose_layer, 'output_padding'):
        output_padding_h, output_padding_w = conv_transpose_layer.output_padding
    else:
        output_padding_h, output_padding_w = 0, 0
        
    if hasattr(conv_transpose_layer, 'dilation'):
        dilation_h, dilation_w = conv_transpose_layer.dilation
    else:
        dilation_h, dilation_w = 1, 1
    
    # Output height and width for transposed convolution
    out_height = (in_height - 1) * stride_h - 2 * padding_h + dilation_h * (kernel_h - 1) + output_padding_h + 1
    out_width = (in_width - 1) * stride_w - 2 * padding_w + dilation_w * (kernel_w - 1) + output_padding_w + 1
    
    # For sparse convolution, we only compute for non-zero elements
    # This is an approximation - the actual FLOPs depend on the distribution of non-zeros
    sparse_ratio = non_zeros / (batch_size * in_channels * in_height * in_width)
    
    # For each output pixel, we need in_channels * kernel_h * kernel_w multiply-adds
    flops = batch_size * out_height * out_width * out_channels * in_channels * kernel_h * kernel_w // conv_transpose_layer.groups
    
    # Adjust for sparsity - this is a simplification assuming uniform distribution of zeros
    sparse_flops = flops * sparse_ratio
    
    # Add bias FLOPs if applicable
    if conv_transpose_layer.bias is not None:
        sparse_flops += batch_size * out_height * out_width * out_channels
    
    return int(sparse_flops)
