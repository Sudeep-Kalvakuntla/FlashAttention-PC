// Milestone 3: FlashAttention-style attention in CUDA (FP32), with split-KV.
//
// This one program contains:
//   1. The naive 3-kernel attention from Milestone 2 (our baseline).
//   2. A fused FlashAttention kernel (tiling + online softmax).
//   3. Split-KV: optionally split the keys across several blocks, then merge.
//   4. A small lookup-table "selector" that picks a configuration per shape.
//   5. Timing code and a CPU reference to check correctness.
//
// Tensor layouts (all contiguous float arrays):
//   Q   [B, H, M, D]   M = number of queries
//   K,V [B, H, N, D]   N = number of keys
//   O   [B, H, M, D]
// B and H are flattened into one "head" index: head = b*H + h.
//
// Example:
//   ./attention --mode fixed --config 8x16s4 --b 1 --h 4 --m 1 --n 2048 --d 64 --check

#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Stop with an error message if a CUDA call fails.
#define CUDA_CHECK(call)                                                 \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess)                                          \
            throw std::runtime_error(cudaGetErrorString(err));           \
    } while (0)

// ---------------------------------------------------------------------------
// Basic data types
// ---------------------------------------------------------------------------

// Size of one attention problem.
struct Shape {
    int b = 1;       // batch size
    int h = 4;       // number of heads
    int m = 128;     // number of queries
    int n = 1024;    // number of keys
    int d = 64;      // head dimension (32, 64 or 128)
    int causal = 1;  // 1 = causal mask, 0 = no mask
};

// One kernel configuration, written like "8x16" or "8x16s4".
struct Config {
    int rows = 4;    // query rows per block (= warps per block)
    int keys = 32;   // keys per K/V tile in shared memory
    int splits = 1;  // split-KV: how many blocks share the keys of one row
};

// GPU buffers used by the kernels.
struct DeviceBuffers {
    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* out = nullptr;
    float* scratch = nullptr;  // naive: score matrix, split-KV: partial results
};

// ---------------------------------------------------------------------------
// Warp helpers
// ---------------------------------------------------------------------------

// Sum a value across the 32 threads of a warp. Every thread gets the result.
__device__ float warp_sum(float value) {
    for (int offset = 16; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffff, value, offset);
    return __shfl_sync(0xffffffff, value, 0);
}

// Maximum of a value across the 32 threads of a warp. Every thread gets the result.
__device__ float warp_max(float value) {
    for (int offset = 16; offset > 0; offset /= 2)
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    return __shfl_sync(0xffffffff, value, 0);
}

// ---------------------------------------------------------------------------
// FlashAttention kernel
// ---------------------------------------------------------------------------
//
// Work split:
//   - One warp computes one query row.
//   - One block has ROWS warps, so it computes ROWS query rows.
//   - The block loads KEYS keys (and values) at a time into shared memory,
//     and all its warps reuse that tile.
//
// Each thread holds D/32 numbers of the query and of the output row
// (lane 0 holds dims 0, 32, 64, 96; lane 1 holds dims 1, 33, ... etc).
//
// Online softmax: we never store all N scores. For each row we keep
//   running_max = largest score seen so far
//   running_sum = sum of exp(score - running_max)
//   out         = sum of exp(score - running_max) * v
// When a new tile has a bigger max, the old sum and output are multiplied
// by exp(old_max - new_max) so everything stays relative to the same max.
// At the end, the answer is out / running_sum.
//
// Split-KV: blockIdx.z picks which part of the keys this block handles
// (keys_per_split keys each). With one split we write the final output.
// With more splits we write the unnormalized output and (max, sum) to scratch,
// and merge_splits_kernel combines them afterwards.
//
// ROWS and KEYS are template parameters so the compiler knows the loop sizes.
template <int ROWS, int KEYS>
__global__ void flash_attention_kernel(const float* Q, const float* K, const float* V, float* O,
                                       float* partial_out, float* partial_stats,
                                       int M, int N, int D, bool causal, int keys_per_split) {
    // Shared memory: [ K tile | V tile | one row of scores per warp ]
    extern __shared__ float shared_mem[];
    float* k_tile = shared_mem;
    float* v_tile = k_tile + KEYS * D;
    float* all_scores = v_tile + KEYS * D;

    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    int row = blockIdx.x * ROWS + warp;  // query row of this warp
    int head = blockIdx.y;               // flattened batch*heads index
    int split = blockIdx.z;              // which part of the keys
    float* scores = all_scores + warp * KEYS;
    int dims_per_lane = D / 32;          // 1, 2 or 4
    float scale = rsqrtf((float)D);      // 1 / sqrt(D)

    // Load this warp's query row into registers.
    float q[4] = {0, 0, 0, 0};
    float out[4] = {0, 0, 0, 0};
    if (row < M) {
        for (int c = 0; c < dims_per_lane; c++)
            q[c] = Q[((size_t)head * M + row) * D + c * 32 + lane];
    }
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    // Range of keys for this block: [first_key, end_key).
    int first_key = split * keys_per_split;
    int end_key = min(N, first_key + keys_per_split);
    if (causal) {
        // With a causal mask, query i can only see keys j <= N - M + i.
        // The last row of the block sees the most keys, so the block can stop there.
        int last_row = min((int)blockIdx.x * ROWS + ROWS - 1, M - 1);
        end_key = min(end_key, N - M + last_row + 1);
    }
    // If end_key <= first_key, this block has nothing to do (the loop is skipped)
    // and it reports an empty result: max = -inf, sum = 0.

    for (int tile_start = first_key; tile_start < end_key; tile_start += KEYS) {
        // Step 1: all threads in the block load one tile of K and V.
        for (int i = threadIdx.x; i < KEYS * D; i += blockDim.x) {
            int key = tile_start + i / D;
            int dim = i % D;
            if (key < N) {
                k_tile[i] = K[((size_t)head * N + key) * D + dim];
                v_tile[i] = V[((size_t)head * N + key) * D + dim];
            } else {
                k_tile[i] = 0.0f;
                v_tile[i] = 0.0f;
            }
        }
        __syncthreads();

        // Warps whose row is past the end still have to reach __syncthreads below,
        // so they skip the work instead of returning early.
        if (row < M) {
            // Step 2: score for each key in the tile = (q . k) / sqrt(D).
            for (int j = 0; j < KEYS; j++) {
                float partial = 0.0f;
                for (int c = 0; c < dims_per_lane; c++)
                    partial += q[c] * k_tile[j * D + c * 32 + lane];
                float score = warp_sum(partial) * scale;

                int key = tile_start + j;
                bool allowed = key < N && (!causal || key <= N - M + row);
                if (lane == 0)
                    scores[j] = allowed ? score : -INFINITY;
            }
            __syncwarp();

            // Step 3: maximum score in this tile.
            float tile_max = -INFINITY;
            for (int j = lane; j < KEYS; j += 32)
                tile_max = fmaxf(tile_max, scores[j]);
            tile_max = warp_max(tile_max);

            // If every key in the tile is masked out there is nothing to add.
            if (tile_max != -INFINITY) {
                float new_max = fmaxf(running_max, tile_max);
                float correction = expf(running_max - new_max);  // rescales old values

                // Step 4: turn scores into exp(score - new_max).
                float tile_sum = 0.0f;
                for (int j = lane; j < KEYS; j += 32) {
                    float p = expf(scores[j] - new_max);
                    scores[j] = p;
                    tile_sum += p;
                }
                tile_sum = warp_sum(tile_sum);
                __syncwarp();

                // Step 5: out = correction * out + sum_j p_j * v_j
                for (int c = 0; c < dims_per_lane; c++) {
                    float weighted_v = 0.0f;
                    for (int j = 0; j < KEYS; j++)
                        weighted_v += scores[j] * v_tile[j * D + c * 32 + lane];
                    out[c] = correction * out[c] + weighted_v;
                }
                running_sum = correction * running_sum + tile_sum;
                running_max = new_max;
            }
        }
        // Wait for every warp before the next tile overwrites shared memory.
        __syncthreads();
    }

    if (row >= M)
        return;
    size_t row_index = (size_t)head * M + row;  // row number across all heads

    if (gridDim.z == 1) {
        // No split-KV: we saw every key, so write the final answer.
        for (int c = 0; c < dims_per_lane; c++)
            O[row_index * D + c * 32 + lane] = out[c] / running_sum;
    } else {
        // Split-KV: save the partial result for the merge kernel.
        size_t total_rows = (size_t)gridDim.y * M;
        size_t slot = split * total_rows + row_index;
        for (int c = 0; c < dims_per_lane; c++)
            partial_out[slot * D + c * 32 + lane] = out[c];
        if (lane == 0) {
            partial_stats[2 * slot] = running_max;
            partial_stats[2 * slot + 1] = running_sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Split-KV merge kernel
// ---------------------------------------------------------------------------
//
// Each split gave us (max_s, sum_s, out_s) for every row. Same idea as the
// online softmax above:
//   max = max over s of max_s
//   sum = sum over s of sum_s * exp(max_s - max)
//   O   = sum over s of out_s * exp(max_s - max) / sum
// One warp per row. Split 0 always contains key 0 (which every row can see),
// so max is never -inf. Empty splits have max_s = -inf, so their weight is 0.
__global__ void merge_splits_kernel(const float* partial_out, const float* partial_stats, float* O,
                                    size_t total_rows, int D, int splits) {
    int lane = threadIdx.x % 32;
    int warps_per_block = blockDim.x / 32;
    size_t row_index = (size_t)blockIdx.x * warps_per_block + threadIdx.x / 32;
    if (row_index >= total_rows)
        return;

    float global_max = -INFINITY;
    for (int s = 0; s < splits; s++) {
        size_t slot = s * total_rows + row_index;
        global_max = fmaxf(global_max, partial_stats[2 * slot]);
    }

    float total_sum = 0.0f;
    float out[4] = {0, 0, 0, 0};
    for (int s = 0; s < splits; s++) {
        size_t slot = s * total_rows + row_index;
        float weight = expf(partial_stats[2 * slot] - global_max);
        total_sum += weight * partial_stats[2 * slot + 1];
        for (int c = 0; c < D / 32; c++)
            out[c] += weight * partial_out[slot * D + c * 32 + lane];
    }

    for (int c = 0; c < D / 32; c++)
        O[row_index * D + c * 32 + lane] = out[c] / total_sum;
}

// ---------------------------------------------------------------------------
// Naive attention from Milestone 2 (3 kernels, full score matrix)
// ---------------------------------------------------------------------------
// Extended from reference/m2_original.cu to support batches, heads,
// rectangular shapes and the causal mask.

// S = Q K^T * scale. One thread per (query i, key j).
__global__ void score_kernel(const float* Q, const float* K, float* S,
                             int M, int N, int D, bool causal, float scale) {
    int j = blockIdx.x * 16 + threadIdx.x;
    int i = blockIdx.y * 16 + threadIdx.y;
    int head = blockIdx.z;
    if (i >= M || j >= N)
        return;

    float sum = 0.0f;
    for (int x = 0; x < D; x++)
        sum += Q[((size_t)head * M + i) * D + x] * K[((size_t)head * N + j) * D + x];

    bool masked = causal && j > N - M + i;
    S[((size_t)head * M + i) * N + j] = masked ? -INFINITY : sum * scale;
}

// Softmax of each row of S, in place. One thread per row.
__global__ void softmax_kernel(float* S, int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int head = blockIdx.y;
    if (i >= M)
        return;

    float* row = S + ((size_t)head * M + i) * N;
    float max_value = row[0];  // key 0 is never masked
    for (int j = 1; j < N; j++)
        if (row[j] > max_value)
            max_value = row[j];

    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        row[j] = expf(row[j] - max_value);
        sum += row[j];
    }
    for (int j = 0; j < N; j++)
        row[j] /= sum;
}

// O = S V. One thread per (query i, dimension x).
__global__ void output_kernel(const float* S, const float* V, float* O, int M, int N, int D) {
    int x = blockIdx.x * 16 + threadIdx.x;
    int i = blockIdx.y * 16 + threadIdx.y;
    int head = blockIdx.z;
    if (i >= M || x >= D)
        return;

    float sum = 0.0f;
    for (int j = 0; j < N; j++)
        sum += S[((size_t)head * M + i) * N + j] * V[((size_t)head * N + j) * D + x];
    O[((size_t)head * M + i) * D + x] = sum;
}

// ---------------------------------------------------------------------------
// Launching kernels
// ---------------------------------------------------------------------------

size_t shared_memory_bytes(int rows, int keys, int d) {
    // K tile + V tile + scores (one row of `keys` floats per warp)
    return (2 * keys * d + rows * keys) * sizeof(float);
}

// By default a kernel can use at most 48 KB of shared memory. Bigger tiles
// (e.g. 64 keys with D = 128) need more, so we ask for it once at startup.
template <int ROWS, int KEYS>
void allow_shared_memory(int d) {
    size_t bytes = shared_memory_bytes(ROWS, KEYS, d);
    if (bytes > 48 * 1024)
        CUDA_CHECK(cudaFuncSetAttribute(flash_attention_kernel<ROWS, KEYS>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes));
}

void setup_shared_memory(int d) {
    allow_shared_memory<4, 16>(d);
    allow_shared_memory<4, 32>(d);
    allow_shared_memory<4, 64>(d);
    allow_shared_memory<8, 16>(d);
    allow_shared_memory<8, 32>(d);
    allow_shared_memory<8, 64>(d);
}

// Number of scratch floats split-KV needs: per split and row, D outputs + max + sum.
size_t split_scratch_floats(Config c, Shape s) {
    if (c.splits == 1)
        return 0;
    return (size_t)c.splits * s.b * s.h * s.m * (s.d + 2);
}

template <int ROWS, int KEYS>
void launch_flash_kernel(Shape s, const DeviceBuffers& buf, int splits) {
    // Split the keys into `splits` parts. Each part is a whole number of tiles.
    int tiles = (s.n + KEYS - 1) / KEYS;
    int tiles_per_split = (tiles + splits - 1) / splits;
    int keys_per_split = tiles_per_split * KEYS;

    dim3 grid((s.m + ROWS - 1) / ROWS, s.b * s.h, splits);
    dim3 block(ROWS * 32);
    size_t shared = shared_memory_bytes(ROWS, KEYS, s.d);

    // Scratch layout for split-KV: all partial outputs first, then all (max, sum) pairs.
    size_t total_rows = (size_t)s.b * s.h * s.m;
    float* partial_out = buf.scratch;
    float* partial_stats = buf.scratch ? buf.scratch + splits * total_rows * s.d : nullptr;

    flash_attention_kernel<ROWS, KEYS><<<grid, block, shared>>>(
        buf.q, buf.k, buf.v, buf.out, partial_out, partial_stats,
        s.m, s.n, s.d, s.causal, keys_per_split);

    if (splits > 1) {
        int threads = 256;  // 8 warps = 8 rows per block
        size_t blocks = (total_rows + 7) / 8;
        merge_splits_kernel<<<blocks, threads>>>(partial_out, partial_stats, buf.out,
                                                 total_rows, s.d, splits);
    }
}

void run_flash_attention(Config c, Shape s, const DeviceBuffers& buf) {
    if (c.rows == 4 && c.keys == 16) launch_flash_kernel<4, 16>(s, buf, c.splits);
    else if (c.rows == 4 && c.keys == 32) launch_flash_kernel<4, 32>(s, buf, c.splits);
    else if (c.rows == 4 && c.keys == 64) launch_flash_kernel<4, 64>(s, buf, c.splits);
    else if (c.rows == 8 && c.keys == 16) launch_flash_kernel<8, 16>(s, buf, c.splits);
    else if (c.rows == 8 && c.keys == 32) launch_flash_kernel<8, 32>(s, buf, c.splits);
    else if (c.rows == 8 && c.keys == 64) launch_flash_kernel<8, 64>(s, buf, c.splits);
    else throw std::runtime_error("Invalid tile configuration");
}

void run_naive_attention(Shape s, const DeviceBuffers& buf) {
    int heads = s.b * s.h;
    float scale = 1.0f / std::sqrt((float)s.d);

    dim3 score_grid((s.n + 15) / 16, (s.m + 15) / 16, heads);
    score_kernel<<<score_grid, dim3(16, 16)>>>(buf.q, buf.k, buf.scratch, s.m, s.n, s.d, s.causal, scale);

    dim3 softmax_grid((s.m + 255) / 256, heads);
    softmax_kernel<<<softmax_grid, 256>>>(buf.scratch, s.m, s.n);

    dim3 output_grid((s.d + 15) / 16, (s.m + 15) / 16, heads);
    output_kernel<<<output_grid, dim3(16, 16)>>>(buf.scratch, buf.v, buf.out, s.m, s.n, s.d);
}

// ---------------------------------------------------------------------------
// Configurations and the selector table
// ---------------------------------------------------------------------------

bool is_valid_config(Config c) {
    bool rows_ok = (c.rows == 4 || c.rows == 8);
    bool keys_ok = (c.keys == 16 || c.keys == 32 || c.keys == 64);
    bool splits_ok = (c.splits == 1 || c.splits == 2 || c.splits == 4 ||
                      c.splits == 8 || c.splits == 16 || c.splits == 32);
    return rows_ok && keys_ok && splits_ok;
}

// Reads "8x16" (one split) or "8x16s4" (four splits).
Config parse_config(const std::string& text) {
    Config c;
    char x_char = 0;
    char s_char = 0;
    std::istringstream in(text);
    bool ok = bool(in >> c.rows >> x_char >> c.keys) && x_char == 'x';
    if (ok && in >> s_char) {
        // Something follows the tile size: it must be "s<number>" and nothing else.
        ok = (s_char == 's') && bool(in >> c.splits) && !(in >> s_char);
    }
    if (!ok || !is_valid_config(c))
        throw std::runtime_error("Expected configuration such as 4x32 or 8x16s4");
    return c;
}

// Round up to the next power of two (e.g. 400 -> 512). Used to group shapes.
int bucket(int x) {
    int p = 1;
    while (p < x)
        p *= 2;
    return p;
}

// One row of the selector table (made by scripts/experiment.py tune).
struct TableEntry {
    int b, h, m_bucket, n_bucket, d, causal;
    Config config;
};

std::vector<TableEntry> load_selector_table(const std::string& path) {
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("Cannot open selector table: " + path);

    std::string line;
    std::getline(file, line);
    if (!line.empty() && line.back() == '\r')
        line.pop_back();
    // Tables made before split-KV have no "splits" column; those mean splits = 1.
    bool has_splits = (line == "b,h,m_bucket,n_bucket,d,causal,rows,keys,splits");
    if (!has_splits && line != "b,h,m_bucket,n_bucket,d,causal,rows,keys")
        throw std::runtime_error("Wrong selector CSV header");

    std::vector<TableEntry> table;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r')
            line.pop_back();
        if (line.empty())
            continue;
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream in(line);
        TableEntry e;
        bool ok = bool(in >> e.b >> e.h >> e.m_bucket >> e.n_bucket >> e.d >> e.causal
                          >> e.config.rows >> e.config.keys);
        if (ok && has_splits)
            ok = bool(in >> e.config.splits);
        if (!ok || !is_valid_config(e.config))
            throw std::runtime_error("Invalid selector table row");
        table.push_back(e);
    }
    return table;
}

// Find the table entry for this shape's bucket. If there is none, use the fallback.
Config select_config(Shape s, const std::vector<TableEntry>& table, Config fallback) {
    int m_bucket = bucket(s.m);
    int n_bucket = bucket(s.n);
    for (const TableEntry& e : table) {
        if (e.b == s.b && e.h == s.h && e.m_bucket == m_bucket && e.n_bucket == n_bucket &&
            e.d == s.d && e.causal == s.causal)
            return e.config;
    }
    return fallback;
}

// ---------------------------------------------------------------------------
// CPU reference (double precision) for checking correctness
// ---------------------------------------------------------------------------

bool check_against_cpu(Shape s, const std::vector<float>& q, const std::vector<float>& k,
                       const std::vector<float>& v, const std::vector<float>& gpu_out) {
    double max_error = 0.0;
    int failures = 0;
    std::vector<double> p(s.n);

    for (int head = 0; head < s.b * s.h; head++) {
        for (int i = 0; i < s.m; i++) {
            // Number of keys this query may look at.
            int visible = s.causal ? s.n - s.m + i + 1 : s.n;

            // Scores and their maximum.
            double max_score = -INFINITY;
            for (int j = 0; j < visible; j++) {
                double dot = 0.0;
                for (int x = 0; x < s.d; x++)
                    dot += (double)q[((size_t)head * s.m + i) * s.d + x] *
                           k[((size_t)head * s.n + j) * s.d + x];
                p[j] = dot / std::sqrt((double)s.d);
                max_score = std::max(max_score, p[j]);
            }

            // Softmax.
            double sum = 0.0;
            for (int j = 0; j < visible; j++) {
                p[j] = std::exp(p[j] - max_score);
                sum += p[j];
            }

            // Output row, compared with the GPU result.
            for (int x = 0; x < s.d; x++) {
                double expected = 0.0;
                for (int j = 0; j < visible; j++)
                    expected += p[j] / sum * v[((size_t)head * s.n + j) * s.d + x];

                double got = gpu_out[((size_t)head * s.m + i) * s.d + x];
                double error = std::abs(got - expected);
                max_error = std::max(max_error, error);
                if (!std::isfinite(got) || error > 2e-4 + 2e-4 * std::abs(expected))
                    failures++;
            }
        }
    }
    std::cerr << "Correctness: max_abs_error=" << max_error << ", failures=" << failures << "\n";
    return failures == 0;
}

// ---------------------------------------------------------------------------
// Command line options
// ---------------------------------------------------------------------------

struct Options {
    Shape shape;
    Config config;          // used by "fixed", and as the fallback for "dynamic"
    std::string mode = "fixed";  // naive, fixed or dynamic
    std::string table_path;
    int warmup = 10;
    int iters = 50;
    int seed = 42;
    float input_scale = 1.0f;  // multiplies Q and K (bigger = sharper softmax)
    bool check = false;
};

Options parse_options(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; i++) {
        std::string flag = argv[i];
        if (flag == "--check") {
            opt.check = true;
            continue;
        }
        if (i + 1 == argc)
            throw std::runtime_error("Missing value for " + flag);
        std::string value = argv[++i];

        if (flag == "--b") opt.shape.b = std::stoi(value);
        else if (flag == "--h") opt.shape.h = std::stoi(value);
        else if (flag == "--m") opt.shape.m = std::stoi(value);
        else if (flag == "--n") opt.shape.n = std::stoi(value);
        else if (flag == "--d") opt.shape.d = std::stoi(value);
        else if (flag == "--causal") opt.shape.causal = std::stoi(value);
        else if (flag == "--mode") opt.mode = value;
        else if (flag == "--config") opt.config = parse_config(value);
        else if (flag == "--table") opt.table_path = value;
        else if (flag == "--warmup") opt.warmup = std::stoi(value);
        else if (flag == "--iters") opt.iters = std::stoi(value);
        else if (flag == "--seed") opt.seed = std::stoi(value);
        else if (flag == "--input-scale") opt.input_scale = std::stof(value);
        else throw std::runtime_error("Unknown option " + flag);
    }

    const Shape& s = opt.shape;
    bool sizes_ok = s.b >= 1 && s.b <= 64 && s.h >= 1 && s.h <= 128 &&
                    s.m >= 1 && s.m <= 32768 && s.n >= 1 && s.n <= 32768;
    bool d_ok = (s.d == 32 || s.d == 64 || s.d == 128);
    bool causal_ok = (s.causal == 0 || s.causal == 1) && !(s.causal && s.m > s.n);
    bool timing_ok = opt.warmup >= 0 && opt.iters >= 1;
    bool scale_ok = std::isfinite(opt.input_scale) && opt.input_scale >= 0;
    if (!sizes_ok || !d_ok || !causal_ok || !timing_ok || !scale_ok)
        throw std::runtime_error("Invalid shape/options; causal requires M<=N, D must be 32/64/128");
    if (opt.mode != "naive" && opt.mode != "fixed" && opt.mode != "dynamic")
        throw std::runtime_error("Unknown mode");
    return opt;
}

// Value at percentile p (0..1) of a list of numbers.
double percentile(std::vector<double> values, double p) {
    std::sort(values.begin(), values.end());
    size_t index = (size_t)std::ceil(p * values.size()) - 1;
    return values[index];
}

double microseconds_between(std::chrono::steady_clock::time_point a,
                            std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}

// ---------------------------------------------------------------------------
// Main: set up data, time the chosen mode, optionally check, print one CSV row
// ---------------------------------------------------------------------------

int main(int argc, char** argv) try {
    Options opt = parse_options(argc, argv);
    Shape s = opt.shape;

    std::vector<TableEntry> table;
    if (opt.mode == "dynamic")
        table = load_selector_table(opt.table_path);

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cerr << "GPU: " << prop.name << ", compute capability " << prop.major << "." << prop.minor << "\n";
    setup_shared_memory(s.d);

    // How many floats each buffer needs.
    size_t q_count = (size_t)s.b * s.h * s.m * s.d;
    size_t kv_count = (size_t)s.b * s.h * s.n * s.d;
    size_t scratch_count = 0;
    if (opt.mode == "naive") {
        scratch_count = (size_t)s.b * s.h * s.m * s.n;  // full score matrix
    } else {
        scratch_count = split_scratch_floats(opt.config, s);
        // Dynamic mode may pick any config in the table, so make room for the largest.
        for (const TableEntry& e : table)
            scratch_count = std::max(scratch_count, split_scratch_floats(e.config, s));
    }

    size_t free_bytes, total_bytes;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    size_t needed_bytes = (2 * q_count + 2 * kv_count + scratch_count) * sizeof(float);
    if (needed_bytes > free_bytes * 0.8)
        throw std::runtime_error("Requested buffers exceed 80% of available GPU memory");

    // Random inputs in [-1, 1]. Q and K are multiplied by input_scale.
    std::vector<float> q(q_count), k(kv_count), v(kv_count), out(q_count);
    std::mt19937 rng(opt.seed);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    for (float& x : q) x = uniform(rng) * opt.input_scale;
    for (float& x : k) x = uniform(rng) * opt.input_scale;
    for (float& x : v) x = uniform(rng);

    DeviceBuffers buf;
    CUDA_CHECK(cudaMalloc(&buf.q, q_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf.k, kv_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf.v, kv_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&buf.out, q_count * sizeof(float)));
    if (scratch_count > 0)
        CUDA_CHECK(cudaMalloc(&buf.scratch, scratch_count * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(buf.q, q.data(), q_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.k, k.data(), kv_count * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.v, v.data(), kv_count * sizeof(float), cudaMemcpyHostToDevice));

    // Picks the config for this call (only "dynamic" actually looks anything up).
    auto choose_config = [&]() {
        if (opt.mode == "dynamic")
            return select_config(s, table, opt.config);
        return opt.config;
    };
    auto run_attention = [&](Config c) {
        if (opt.mode == "naive")
            run_naive_attention(s, buf);
        else
            run_flash_attention(c, s, buf);
        CUDA_CHECK(cudaGetLastError());
    };

    // Warm-up runs (not timed) so the GPU is "awake" before we measure.
    for (int i = 0; i < opt.warmup; i++)
        run_attention(choose_config());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed runs.
    //   gpu      = CUDA events around the kernel launch(es)
    //   selector = CPU time spent choosing the config
    //   host     = CPU wall time for select + launch + wait
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<double> gpu_us, selector_us, host_us;
    Config selected = opt.config;

    for (int i = 0; i < opt.iters; i++) {
        auto t0 = std::chrono::steady_clock::now();
        selected = choose_config();
        auto t1 = std::chrono::steady_clock::now();

        CUDA_CHECK(cudaEventRecord(start));
        run_attention(selected);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        auto t2 = std::chrono::steady_clock::now();

        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        gpu_us.push_back(ms * 1000.0);
        selector_us.push_back(microseconds_between(t0, t1));
        host_us.push_back(microseconds_between(t0, t2));
    }

    if (opt.check) {
        CUDA_CHECK(cudaMemcpy(out.data(), buf.out, q_count * sizeof(float), cudaMemcpyDeviceToHost));
        if (!check_against_cpu(s, q, k, v, out))
            return 2;
    }

    // One CSV row on stdout (scripts/experiment.py reads this).
    std::cout << "mode,b,h,m,n,d,causal,rows,keys,splits,gpu_median_us,gpu_p95_us,"
                 "selector_median_us,host_median_us,host_p95_us,scratch_bytes\n";
    std::cout << opt.mode << "," << s.b << "," << s.h << "," << s.m << "," << s.n << "," << s.d << ","
              << s.causal << "," << selected.rows << "," << selected.keys << "," << selected.splits
              << std::fixed << std::setprecision(6)
              << "," << percentile(gpu_us, 0.5) << "," << percentile(gpu_us, 0.95)
              << "," << percentile(selector_us, 0.5)
              << "," << percentile(host_us, 0.5) << "," << percentile(host_us, 0.95)
              << "," << scratch_count * sizeof(float) << "\n";

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(buf.q));
    CUDA_CHECK(cudaFree(buf.k));
    CUDA_CHECK(cudaFree(buf.v));
    CUDA_CHECK(cudaFree(buf.out));
    if (buf.scratch)
        CUDA_CHECK(cudaFree(buf.scratch));
    return 0;
} catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
}
