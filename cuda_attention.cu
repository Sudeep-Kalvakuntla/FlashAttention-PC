/*
 * Computes standard scaled dot-product attention on the GPU:
 *     S = (Q @ K^T) / sqrt(d)
 *     P = softmax(S)
 *     O = P @ V
 *
 * One CUDA kernel per step. The full N x N score matrix S is written to
 * global memory, exactly like the naive Python version.
 *
 * Build: nvcc -O2 cuda_attention.cu -o cuda_attention
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define THREADS 16


/* ---------------- Kernels ---------------- */

// 1. S = (Q @ K^T) * scale
// One thread computes one entry S[i][j].
__global__ void scores_kernel(const float *Q, const float *K, float *S,
                              int N, int d, float scale)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < N) {
        float dot = 0.0f;
        for (int k = 0; k < d; k++) {
            dot += Q[i * d + k] * K[j * d + k];
        }
        S[i * N + j] = dot * scale;
    }
}

// 2. P = softmax(S)
// One thread handles one full row (numerically stable: subtract the max).
__global__ void softmax_kernel(float *S, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        float m = S[i * N];
        for (int j = 1; j < N; j++) {
            if (S[i * N + j] > m) {
                m = S[i * N + j];
            }
        }

        float row_sum = 0.0f;
        for (int j = 0; j < N; j++) {
            float e = expf(S[i * N + j] - m);
            S[i * N + j] = e;
            row_sum += e;
        }

        for (int j = 0; j < N; j++) {
            S[i * N + j] /= row_sum;
        }
    }
}

// 3. O = P @ V
// One thread computes one entry O[i][j].
__global__ void output_kernel(const float *P, const float *V, float *O,
                              int N, int d)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N && j < d) {
        float val = 0.0f;
        for (int k = 0; k < N; k++) {
            val += P[i * N + k] * V[k * d + j];
        }
        O[i * d + j] = val;
    }
}


/* ---------------- Host helpers ---------------- */

// Generate a flat rows x cols array of Gaussian random values.
void random_matrix(float *M, int rows, int cols, unsigned int seed)
{
    srand(seed);
    for (int i = 0; i < rows * cols; i++) {
        // Box-Muller transform on two uniform samples
        float u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
        float u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
        M[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    }
}

// Runs the three kernels and returns the elapsed GPU time in milliseconds.
float cuda_attention(const float *hQ, const float *hK, const float *hV,
                     float *hO, int N, int d)
{
    float scale = 1.0f / sqrtf((float)d);
    size_t size_nd = (size_t)N * d * sizeof(float);
    size_t size_nn = (size_t)N * N * sizeof(float);

    // Allocate device memory and copy the inputs over
    float *Q, *K, *V, *S, *O;
    cudaMalloc(&Q, size_nd);
    cudaMalloc(&K, size_nd);
    cudaMalloc(&V, size_nd);
    cudaMalloc(&S, size_nn);
    cudaMalloc(&O, size_nd);

    cudaMemcpy(Q, hQ, size_nd, cudaMemcpyHostToDevice);
    cudaMemcpy(K, hK, size_nd, cudaMemcpyHostToDevice);
    cudaMemcpy(V, hV, size_nd, cudaMemcpyHostToDevice);

    // Time the computation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    dim3 block(THREADS, THREADS);
    dim3 grid_nn((N + THREADS - 1) / THREADS, (N + THREADS - 1) / THREADS);
    dim3 grid_nd((d + THREADS - 1) / THREADS, (N + THREADS - 1) / THREADS);

    scores_kernel<<<grid_nn, block>>>(Q, K, S, N, d, scale);
    softmax_kernel<<<(N + 255) / 256, 256>>>(S, N);
    output_kernel<<<grid_nd, block>>>(S, V, O, N, d);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    // Copy the result back and free everything
    cudaMemcpy(hO, O, size_nd, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(S);
    cudaFree(O);

    return elapsed_ms;
}


int main()
{
    // Test different sequence lengths (N) and head dimensions (d)
    int sequence_lengths[] = {64, 128, 256, 512};
    int head_dimensions[] = {32, 64};
    int num_lengths = 4;
    int num_dims = 2;

    printf("%-14s | %-14s | %-14s | %-10s\n",
           "Seq Len (N)", "Head Dim (d)", "Output Shape", "Time (ms)");
    for (int i = 0; i < 65; i++) {
        printf("-");
    }
    printf("\n");

    for (int a = 0; a < num_dims; a++) {
        for (int b = 0; b < num_lengths; b++) {
            int d = head_dimensions[a];
            int N = sequence_lengths[b];

            // Generate random Q, K, V matrices
            float *Q = (float *)malloc((size_t)N * d * sizeof(float));
            float *K = (float *)malloc((size_t)N * d * sizeof(float));
            float *V = (float *)malloc((size_t)N * d * sizeof(float));
            float *O = (float *)malloc((size_t)N * d * sizeof(float));

            random_matrix(Q, N, d, 1);
            random_matrix(K, N, d, 2);
            random_matrix(V, N, d, 3);

            float elapsed_ms = cuda_attention(Q, K, V, O, N, d);

            char out_shape_str[32];
            sprintf(out_shape_str, "%d x %d", N, d);
            printf("%-14d | %-14d | %-14s | %-10.2f\n",
                   N, d, out_shape_str, elapsed_ms);

            free(Q);
            free(K);
            free(V);
            free(O);
        }
    }

    return 0;
}
