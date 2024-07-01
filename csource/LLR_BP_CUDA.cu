// LLR_BP_CUDA.cu

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

__global__ void update_LLRR_kernel(int* H, float* LLRQ, float* LLRR, float* LLR_initial, int num_checks, int num_vars) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < num_checks && j < num_vars) {
        if (H[i * num_vars + j] == 1) {
            float product = 1.0f;
            for (int k = 0; k < num_vars; ++k) {
                if (k != j && H[i * num_vars + k] == 1) {
                    product *= tanh(LLRQ[i * num_vars + k] / 2.0f);
                }
            }
            LLRR[i * num_vars + j] = 2.0f * atanh(product);
        }
    }
}

__global__ void update_LLRQ_kernel(int* H, float* LLRQ, float* LLRR, float* LLR_initial, int num_checks, int num_vars) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < num_checks && j < num_vars) {
        if (H[i * num_vars + j] == 1) {
            float sum_llr = 0.0f;
            for (int k = 0; k < num_checks; ++k) {
                if (k != i && H[k * num_vars + j] == 1) {
                    sum_llr += LLRR[k * num_vars + j];
                }
            }
            LLRQ[i * num_vars + j] = LLR_initial[j] + sum_llr;
        }
    }
}

__global__ void calculate_LQ_kernel(int* H, float* LLRQ, float* LLRR, float* LLR_initial, float* LQ, int num_checks, int num_vars) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (j < num_vars) {
        float sum_llr = 0.0f;
        for (int i = 0; i < num_checks; ++i) {
            if (H[i * num_vars + j] == 1) {
                sum_llr += LLRR[i * num_vars + j];
            }
        }
        LQ[j] = LLR_initial[j] + sum_llr;
    }
}

void LLR_BP_CUDA(py::array_t<int> H, py::array_t<float> LLR_initial, py::array_t<float> received_signal, py::array_t<float> decoded_bits, bool& flag, int max_iter) {
    auto buf_H = H.request(), buf_LLRI = LLR_initial.request(), buf_received = received_signal.request(), buf_decoded = decoded_bits.request();

    int num_checks = buf_H.shape[0];
    int num_vars = buf_H.shape[1];

    int *ptr_H = static_cast<int *>(buf_H.ptr);
    float *ptr_LLRI = static_cast<float *>(buf_LLRI.ptr);
    float *ptr_received = static_cast<float *>(buf_received.ptr);
    float *ptr_decoded = static_cast<float *>(buf_decoded.ptr);

    float *d_LLRI, *d_received, *d_decoded, *d_LLRQ, *d_LLRR, *d_LQ;
    int *d_H;

    cudaMalloc(&d_H, num_checks * num_vars * sizeof(int));
    cudaMalloc(&d_LLRI, num_vars * sizeof(float));
    cudaMalloc(&d_received, num_vars * sizeof(float));
    cudaMalloc(&d_decoded, num_vars * sizeof(float));
    cudaMalloc(&d_LLRQ, num_checks * num_vars * sizeof(float));
    cudaMalloc(&d_LLRR, num_checks * num_vars * sizeof(float));
    cudaMalloc(&d_LQ, num_vars * sizeof(float));

    cudaMemcpy(d_H, ptr_H, num_checks * num_vars * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_LLRI, ptr_LLRI, num_vars * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_received, ptr_received, num_vars * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_decoded, ptr_decoded, num_vars * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((num_checks + threadsPerBlock.x - 1) / threadsPerBlock.x, (num_vars + threadsPerBlock.y - 1) / threadsPerBlock.y);

    for (int iter = 0; iter < max_iter; ++iter) {
        update_LLRR_kernel<<<numBlocks, threadsPerBlock>>>(d_H, d_LLRQ, d_LLRR, d_LLRI, num_checks, num_vars);
        cudaDeviceSynchronize();

        update_LLRQ_kernel<<<numBlocks, threadsPerBlock>>>(d_H, d_LLRQ, d_LLRR, d_LLRI, num_checks, num_vars);
        cudaDeviceSynchronize();

        calculate_LQ_kernel<<<(num_vars + threadsPerBlock.x - 1) / threadsPerBlock.x, threadsPerBlock.x>>>(d_H, d_LLRQ, d_LLRR, d_LLRI, d_LQ, num_checks, num_vars);
        cudaDeviceSynchronize();

        cudaMemcpy(ptr_decoded, d_decoded, num_vars * sizeof(float), cudaMemcpyDeviceToHost);

        bool is_valid = true;
        for (int i = 0; i < num_checks; ++i) {
            int check_sum = 0;
            for (int j = 0; j < num_vars; ++j) {
                if (ptr_H[i * num_vars + j] == 1) {
                    check_sum += ptr_decoded[j];
                }
            }
            if (check_sum % 2 != 0) {
                is_valid = false;
                break;
            }
        }

        flag = is_valid;
        if (flag) {
            break;
        }
    }

    cudaFree(d_H);
    cudaFree(d_LLRI);
    cudaFree(d_received);
    cudaFree(d_decoded);
    cudaFree(d_LLRQ);
    cudaFree(d_LLRR);
    cudaFree(d_LQ);
}

PYBIND11_MODULE(LLR_BP_CUDA, m) {
m.def("LLR_BP_CUDA", &LLR_BP_CUDA, "LLR_BP_CUDA function with CUDA acceleration");
}

