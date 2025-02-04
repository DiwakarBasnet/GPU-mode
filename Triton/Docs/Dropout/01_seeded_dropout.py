import torch
import tabulate
import triton
import triton.language as tl


@triton.jit
def _seeded_dropout(
    input_ptr, output_ptr,
    n_rows, n_cols,
    p, seed_ptr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    row_idx = pid
    col_idx =tl.arange(0, BLOCK_SIZE)

    if row_idx < n_rows:
        seed = tl.load(seed_ptr + row_idx)
        offsets = row_idx * n_cols + col_idx
        mask= col_idx < n_cols

        input = tl.load(input_ptr + offsets, mask=mask)

        # Random prune using row_specific seed
        random = tl.rand(seed, col_idx)
        x_keep = random > p
        output = tl.where(x_keep, input / (1 - p), 0.0)
        tl.store(output_ptr + offsets, output, mask=mask)


def seeded_dropout(x, p, seeds):
    output = torch.empty_like(x)
    assert x.is_contiguous()
    assert seeds.is_contiguous()
    n_rows, n_cols = x.shape
    grid = lambda meta: (n_rows, triton.cdiv(n_cols, meta['BLOCK_SIZE']), )
    _seeded_dropout[grid](x, output, n_rows, n_cols, p, seeds, BLOCK_SIZE=1024)
    return output


# Testing function
if __name__== '__main__':
    DEVICE = 'cuda'
    x = torch.randn(size=(3, 10000), device=DEVICE)
    seeds = torch.tensor([123, 456, 789], device=DEVICE)

    # Apply dropout with the same seeds
    output1 = seeded_dropout(x, p=0.5, seeds=seeds)
    output2 = seeded_dropout(x, p=0.5, seeds=seeds)

    # Apply dropout with different seeds
    seeds2 = torch.tensor([111, 222, 333], device=DEVICE)
    output3 = seeded_dropout(x, p=0.5, seeds=seeds2)

    print(
        tabulate.tabulate([
            ["input"] + x.tolist(),
            ["output (seeds = [123, 456, 789])"] + output1.tolist(),
            ["output (seeds = [123, 456, 789])"] + output2.tolist(),
            ["output (seeds = [111, 222, 333])"] + output3.tolist(),
        ])
    )
