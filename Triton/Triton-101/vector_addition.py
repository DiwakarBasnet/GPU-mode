import torch
import triton
import triton.language as tl

DEVICE = torch.device(f'cuda:{torch.cuda.current_device()}')


@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr    # tells triton that this value is known at compile time (constant)
):
    # Which chunk of data are we responsible for?
    PID = tl.program_id(axis=0)
    # vec of length 256
    # BLOCK_SIZE 64
    # PID 0 might process elements [0:64]
    # PID 1 might process elements [64:128]

    block_start = PID * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # vec of [0,1,2,...,BLOCK_SIZE-1] + block_start
    mask = offsets < n_elements  # handle case when n_elements is not a multiple of BLOCK_SIZE

    # load data from DRAM/VRAM/HBM to SRAM/on-chip memory
    x = tl.load(x_ptr + offsets, mask=mask, other=None)  # shape (BLOCK_SIZE,)
    y = tl.load(y_ptr + offsets, mask=mask, other=None)

    output = x + y

    # write data back to DRAM
    tl.store(output_ptr + offsets, output, mask=mask)


def add(x, y):
    # pre-allocate the output
    output = torch.empty_like(x)

    # check tensors are on same device (raise assertion error if not)
    assert x.device == DEVICE and y.device == DEVICE

    # defining our launch grid (no. of programs that will launch in parallel)
    n_elements = output.numel()  # total no. of elements in that tensor
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    return output


def test_add_kernel(size, atol=1e-3, rtol=1e-3, device=DEVICE):
    # create test data
    torch.manual_seed(0)
    x = torch.randn(size, device=DEVICE)
    y = torch.randn(size, device=DEVICE)
    # run tritom kernel & pytorch equivalent
    z_tri = add(x, y)
    z_ref = x + y
    # compare
    torch.testing.assert_close(z_tri, z_ref, atol=atol, rtol=rtol)
    print('passed')


if __name__ == '__main__':
    test_add_kernel(size=4096)
    test_add_kernel(size=4097)
    test_add_kernel(size=98432)
