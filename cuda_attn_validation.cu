/*
 * Computes standard scaled dot-product attention on the GPU and compares 
 * it against a sequential CPU implementation for correctness validation.
 * 
 * Build: nvcc -O2 cuda_attention_val.cu -o cuda_attention_val
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define THREADS 16


/* ---------------- Kernels ---------------- */

// 1. S = (Q @ K^T) * scale
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
        float u1 = (rand() + 1.0) / (RAND_MAX + 2.0);
        float u2 = (rand() + 1.0) / (RAND_MAX + 2.0);
        M[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    }
}

// Sequential CPU implementation for correctness validation
void cpu_attention(const float *Q, const float *K, const float *V, 
                   float *O, float *P, int N, int d)
{
    float scale = 1.0f / sqrtf((float)d);
    
    for (int i = 0; i < N; i++) {
        // 1. S = (Q @ K^T) / sqrt(d)
        float m = -1e20f; // Track max for numerical stability
        for (int j = 0; j < N; j++) {
            float dot = 0.0f;
            for (int k = 0; k < d; k++) {
                dot += Q[i * d + k] * K[j * d + k];
            }
            P[i * N + j] = dot * scale;
            if (P[i * N + j] > m) {
                m = P[i * N + j];
            }
        }
        
        // 2. P = softmax(S)
        float row_sum = 0.0f;
        for (int j = 0; j < N; j++) {
            P[i * N + j] = expf(P[i * N + j] - m);
            row_sum += P[i * N + j];
        }
        for (int j = 0; j < N; j++) {
            P[i * N + j] /= row_sum;
        }
        
        // 3. O = P @ V
        for (int j = 0; j < d; j++) {
            float val = 0.0f;
            for (int k = 0; k < N; k++) {
                val += P[i * N + k] * V[k * d + j];
            }
            O[i * d + j] = val;
        }
    }
}

// Runs the three kernels, extracting both O and intermediate P for validation.
void cuda_attention(const float *hQ, const float *hK, const float *hV,
                    float *hO, float *hP, int N, int d)
{
    float scale = 1.0f / sqrtf((float)d);
    size_t size_nd = (size_t)N * d * sizeof(float);
    size_t size_nn = (size_t)N * N * sizeof(float);

    float *Q, *K, *V, *S, *O;
    cudaMalloc(&Q, size_nd);
    cudaMalloc(&K, size_nd);
    cudaMalloc(&V, size_nd);
    cudaMalloc(&S, size_nn);
    cudaMalloc(&O, size_nd);

    cudaMemcpy(Q, hQ, size_nd, cudaMemcpyHostToDevice);
    cudaMemcpy(K, hK, size_nd, cudaMemcpyHostToDevice);
    cudaMemcpy(V, hV, size_nd, cudaMemcpyHostToDevice);

    dim3 block(THREADS, THREADS);
    dim3 grid_nn((N + THREADS - 1) / THREADS, (N + THREADS - 1) / THREADS);
    dim3 grid_nd((d + THREADS - 1) / THREADS, (N + THREADS - 1) / THREADS);

    scores_kernel<<<grid_nn, block>>>(Q, K, S, N, d, scale);
    softmax_kernel<<<(N + 255) / 256, 256>>>(S, N);
    output_kernel<<<grid_nd, block>>>(S, V, O, N, d);

    cudaDeviceSynchronize();

    // Copy O and P (which is stored in S after softmax) back to host
    cudaMemcpy(hO, O, size_nd, cudaMemcpyDeviceToHost);
    cudaMemcpy(hP, S, size_nn, cudaMemcpyDeviceToHost);

    cudaFree(Q);
    cudaFree(K);
    cudaFree(V);
    cudaFree(S);
    cudaFree(O);
}


int main()
{
    // Configurations for the validation table
    int val_N[] = {64, 512, 64, 512};
    int val_d[] = {32, 32, 64, 64};
    int num_tests = 4;

    printf("%-5s %-5s %-15s %-20s\n", "N", "d", "Max abs error", "Rows of P sum to 1");
    for (int i = 0; i < 50; i++) printf("-");
    printf("\n");

    for (int t = 0; t < num_tests; t++) {
        int N = val_N[t];
        int d = val_d[t];

        size_t size_nd = (size_t)N * d * sizeof(float);
        size_t size_nn = (size_t)N * N * sizeof(float);

        float *Q = (float *)malloc(size_nd);
        float *K = (float *)malloc(size_nd);
        float *V = (float *)malloc(size_nd);
        
        float *O_gpu = (float *)malloc(size_nd);
        float *P_gpu = (float *)malloc(size_nn);
        
        float *O_cpu = (float *)malloc(size_nd);
        float *P_cpu = (float *)malloc(size_nn);

        // Same random Q, K, V (fixed seeds)[cite: 1]
        random_matrix(Q, N, d, 42);
        random_matrix(K, N, d, 43);
        random_matrix(V, N, d, 44);

        // Run GPU version
        cuda_attention(Q, K, V, O_gpu, P_gpu, N, d);

        // Run CPU version
        cpu_attention(Q, K, V, O_cpu, P_cpu, N, d);

        // 1. Every element of O compared; report largest absolute difference[cite: 1]
        float max_abs_error = 0.0f;
        for (int i = 0; i < N * d; i++) {
            float err = fabsf(O_gpu[i] - O_cpu[i]);
            if (err > max_abs_error) {
                max_abs_error = err;
            }
        }

        // 2. Extra check: every row of P sums to 1[cite: 1]
        int rows_sum_to_one = 1; // True
        for (int i = 0; i < N; i++) {
            float row_sum = 0.0f;
            for (int j = 0; j < N; j++) {
                row_sum += P_gpu[i * N + j];
            }
            // Allow for float32 accumulation error margins around 10^-6[cite: 1]
            if (fabsf(row_sum - 1.0f) > 1e-4f) {
                rows_sum_to_one = 0; // False
                break;
            }
        }

        printf("%-5d %-5d %-15.3e %-20s\n", 
               N, d, max_abs_error, rows_sum_to_one ? "Yes" : "No");

        free(Q); free(K); free(V);
        free(O_gpu); free(P_gpu);
        free(O_cpu); free(P_cpu);
    }

    return 0;
}