import numpy as np
import LLR_BP_CUDA
import time

def test_LLBP_CUDA():
    # 构造测试数据
    num_checks = int(2304 / 2)
    num_vars = 2304

    # 构造随机的 H 矩阵
    H = np.random.randint(0, 2, size=(num_checks, num_vars)).astype(np.int32)

    # 构造随机的 LLR_initial，received_signal，decoded_bits
    LLR_initial = np.random.randn(num_vars).astype(np.float32)
    received_signal = np.random.randn(num_vars).astype(np.float32)
    decoded_bits = np.zeros(num_vars, dtype=np.float32)

    max_iter = 25
    flag = False

    # 调用 CUDA 加速的 LLR_BP_CUDA 函数
    LLR_BP_CUDA.LLR_BP_CUDA(H, LLR_initial, received_signal, decoded_bits, flag, max_iter)

    print("Decoded bits:", decoded_bits)
    print("Decoded bits shape:", decoded_bits.shape)
    print("Decoding successful:", flag)

if __name__ == "__main__":
    t0 = time.time()
    test_LLBP_CUDA()
    print(f'用时：{time.time()-t0}')

