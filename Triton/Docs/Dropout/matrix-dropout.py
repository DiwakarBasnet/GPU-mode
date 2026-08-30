import tabulate
import torch
import triton
import triton.language as tl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def _dropout(
    input_ptr,
    output_ptr,
    row_stride,
    col_stride,
    n_rows,
    n_cols,
    p,
    seeds_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step):
        # Row input
        row_start_ptr = input_ptr + row_idx * row_stride
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets * col_stride
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Seed calculation
        seed_ptr = seeds_ptr + row_idx * 1
        seed_for_row = tl.load(seed_ptr)
        # Random calculation
        random = tl.rand(seed_for_row, col_offsets)
        row_keep = random > p
        # Output calculation
        output = tl.where(row_keep, row / (1 - p), 0.0)
        output_row_start_ptr = output_ptr + row_idx * row_stride
        output_ptrs = output_row_start_ptr + col_offsets * col_stride
        tl.store(output_ptrs, output, mask=mask)


properties = triton.runtime.driver.active.get_device_properties(DEVICE.index)
NUM_SM = properties["multiprocessor_count"]
NUM_REGS = properties["max_num_regs"]
SIZE_SMEM = properties["max_shared_mem"]
WARP_SIZE = properties["warpSize"]
target = triton.runtime.driver.active.get_current_target()
kernels = {}


def dropout(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
