"""The torch/Metal adapter must bind arguments by the device function's
parameter order, which SplitHostDevice sorts by name, not by PrimFunc order."""

import torch

import tilelang
import tilelang.language as T
import tilelang.testing

N = 256
THREADS = 64


def non_alphabetical_params():
    # PrimFunc order (X, B, Y) differs from the sorted device order (B, X, Y).
    @T.prim_func
    def main(
        X: T.Tensor((N,), T.float32),
        B: T.Tensor((N,), T.float32),
        Y: T.Tensor((N,), T.float32),
    ):
        with T.Kernel(T.ceildiv(N, THREADS), threads=THREADS) as bx:
            for i in T.Parallel(THREADS):
                idx = bx * THREADS + i
                Y[idx] = X[idx] * 2.0 + B[idx]

    return main


@tilelang.testing.requires_metal
def test_non_alphabetical_params_bind_correct_buffers():
    kernel = tilelang.compile(non_alphabetical_params(), target="metal", execution_backend="torch")
    x = torch.arange(N, dtype=torch.float32, device="mps")
    b = torch.full((N,), 100.0, device="mps")
    y = torch.zeros(N, device="mps")
    kernel(x, b, y)
    torch.mps.synchronize()
    torch.testing.assert_close(y, x * 2.0 + b)


if __name__ == "__main__":
    tilelang.testing.main()
