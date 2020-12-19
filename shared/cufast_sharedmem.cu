#include "cpu_fast.hpp"

int *h_circle;   // circle in host
int *h_mask;     // mask in host

int *h_iscorner; // iscorner in host
unsigned *h_score;    // score in host

unsigned char *d_img;                         // image in device
int *d_iscorner;                              // iscorner in device
unsigned *d_score;                            // score in device
__constant__ int d_circle[CIRCLEPOINTS];      // circle in device
__constant__ int d_mask[MASKSIZE * MASKSIZE]; // mask in device
static int size_img;                          // imgsize
static int size_img_char;                     // imgsize * 1
static int size_img_int;                      // imgsize * 4
static int size_circle_int;                   // 16 * 4
static int size_mask_int;                     // 9  * 4
static int width_shared;                      // BLOCK_SIZE + (2 * PADDING);
static int size_sharedmem;                    // sharedwidth * sharedwdith

static double gpu_full_time = 0.0;
cudaStream_t streamMem, streamCpt; // memory stream compute stream
cudaEvent_t start, stop;

// #define CHECK_ERROR(error) (HandleError(error, __FILE__, __LINE__))

static void HandleError(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
    getchar();
    exit(EXIT_FAILURE);
  }
}

using namespace std;

__device__ int cuGet1dcoords(int x, int y, int height, int width,
                             bool ignore_broder) {
  bool inborder = (x - PADDING < 0) || (x + PADDING >= width) ||
                  (y - PADDING < 0) || (y + PADDING >= height);
  if (ignore_broder && inborder) {
    return -1;
  } else {
    return x + width * y;
  }
}
__host__ void cuCreateCircle(int *circle, int width) {
  circle[0] = -PADDING * width;
  circle[1] = circle[0] + 1;
  circle[2] = circle[1] + width + 1;
  circle[3] = circle[2] + width + 1;
  circle[4] = circle[3] + width;
  circle[5] = circle[4] + width;
  circle[6] = circle[5] + width - 1;
  circle[7] = circle[6] + width - 1;
  circle[8] = circle[7] - 1;
  circle[9] = circle[8] - 1;
  circle[10] = circle[9] - width - 1;
  circle[11] = circle[10] - width - 1;
  circle[12] = circle[11] - width;
  circle[13] = circle[12] - width;
  circle[14] = circle[13] - width + 1;
  circle[15] = circle[14] - width + 1;
}
__host__ void cuCreateMask(int *mask, int width) {
  int idx = 0;
  int start = -(int)MASKSIZE / 2;
  int end = (int)MASKSIZE / 2;
  // printf("start:%d\n", start);
  // printf("prevstart:%d\n", -(int)(MASKSIZE / 2));
  for (int i = start; i <= end; i++) {
    for (int j = start; j <= end; j++) {
      mask[idx] = i * width + j;
      idx++;
    }
  }
}
__device__ int cuGetScore(int pixel_val, int circle_val, int threshold) {
  int val = pixel_val + threshold;
  if (circle_val > val) { // darker    circle_val > pixel_val + threshold
    return circle_val - pixel_val; // pos
  } else {
    val = pixel_val - threshold;
    if (circle_val < val) { // brighter pixel_val > circle_val + threshold
      return circle_val - pixel_val; // mimus
    } else {
      return 0;
    }
  }
}
// void memoryPreparation(unsigned char *h_input, cudaStream_t stream,
//                                 int height, int width) {
void memoryPreparation(cudaStream_t stream, unsigned char *h_input,
                       unsigned *d_score, int *d_iscorner, int width,
                       int height) {
  size_img = width * height;
  // printf("size_img: %d\n", size_img);
  // getchar();
  size_img_char = size_img * sizeof(unsigned char);
  size_img_int = size_img * sizeof(int);
  size_circle_int = CIRCLEPOINTS * sizeof(int);
  size_mask_int = MASKSIZE * MASKSIZE * sizeof(int);
  width_shared = BLOCKSIZE + (2 * PADDING);
  size_sharedmem = width_shared * width_shared * sizeof(unsigned char);

  // h_corners = (corner *)malloc(size_img_corner);
  h_circle = (int *)malloc(size_circle_int);
  h_mask = (int *)malloc(size_mask_int);
  h_iscorner = (int *)malloc(size_img_int);
  // printf("--------------------------\n");
  // printf("size_img: %d\n", size_img);
  // printf("--------------------------\n");
  // getchar();
  // CHECK_ERROR(cudaMalloc((void **)&d_img, size_img_char));
  // CHECK_ERROR(cudaMemcpyAsync(d_img, h_input, size_img_char,
  // cudaMemcpyHostToDevice, stream));
  CHECK_ERROR(cudaMalloc((void **)&d_iscorner, size_img_int));
  CHECK_ERROR(cudaMalloc((void **)&d_circle, size_circle_int));
  CHECK_ERROR(cudaMalloc((void **)&d_mask, size_mask_int));
  CHECK_ERROR(cudaMalloc((void **)&d_score, size_img_int));
  cuCreateCircle(h_circle, width_shared);
  cuCreateMask(h_mask, width_shared);
  // for(size_t i= 0; i < 16; i++){
  //   printf("%d\n", h_circle[i]);
  // }
  // getchar();

  // copy to constant : circle & mask
  CHECK_ERROR(cudaMemcpyToSymbol(d_circle, h_circle,
                                 size_circle_int)); // constant memory
  CHECK_ERROR(
      cudaMemcpyToSymbol(d_mask, h_mask, size_mask_int)); // constant memory
  // copy to global: img
  CHECK_ERROR(cudaMemcpyAsync(d_img, h_input, size_img_char,
                              cudaMemcpyHostToDevice, stream));
  // already in device: score & iscorner
  printf("%d\n", size_img_int);
  CHECK_ERROR(cudaMemsetAsync(d_score, 0, size_img_int, stream));
  CHECK_ERROR(cudaMemsetAsync(d_iscorner, 0, size_img_int, stream));
  CHECK_ERROR(cudaStreamSynchronize(stream));
}
__device__ int cuGetMaxScore(unsigned char *sharedinput, unsigned *d_score,
                             int *d_iscorner, int threshold, int shared_idx,
                             int global_idx, int height, int width) {
  // printf("kernel in\n");
  // printf("%d\n", sharedinput[shared_idx + d_circle[0]]);
  // printf("%d %d %d %d\n", shared_idx / 38, shared_idx % 38, global_idx / 32,
  // global_idx % 32);
  unsigned char pixel = sharedinput[shared_idx];
  int score;
  char compare_sign = 0;
  char last_compare_sign = -2;
  int score_sum = 0;
  bool cornerbool = false;
  unsigned char consecutive = 1;
  long max_score = 0;
  for (size_t i = 0; i < (CIRCLEPOINTS + CONSECUTIVE); i++) {
    if (consecutive >= CONSECUTIVE) {
      cornerbool = true;
      if (score_sum > max_score) {
        max_score = score_sum;
      }
    }
    score = cuGetScore(
        pixel, sharedinput[shared_idx + d_circle[i % CIRCLEPOINTS]], threshold);
    // printf("score:%d\n", score);
    compare_sign = (score < 0) ? -1 : (score > 0) ? 1 : 0; // if equal: 0
    if (compare_sign != 0 && last_compare_sign == compare_sign) {
      consecutive++;
      score_sum += abs(score);
    } else {
      if (consecutive < CONSECUTIVE) {
        consecutive = 1;
        score_sum = abs(score);
      }
    }
    last_compare_sign = compare_sign;
  }
  if (cornerbool) {
    if (score_sum > max_score) {
      max_score = score_sum;
    };
    // corner dcp;
    // dcp.x = global_idx % width;
    // dcp.y = global_idx / height;
    // dcp.score = max_score;
    // d_corners[global_idx] = dcp;
    d_score[global_idx] = max_score;
    d_iscorner[global_idx] = 1;
    return max_score;
  } else {
    // d_score[global_idx] = 0;
    return 0;
  }
}
__global__ void kernelCornerDetect(unsigned char *d_input, unsigned *d_score,
                                   int *d_iscorner, int height, int width,
                                   int size_img, int size_img_char,
                                   int size_img_int, int size_circle_int,
                                   int width_shared, int size_sharedmem,
                                   int threshold) {
  extern __shared__ unsigned char shared_data[];
  int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
  int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
  int g_coord1d = cuGet1dcoords(idx_x, idx_y, height, width, true);
  // fiil date in shared memory
  int idx_first =
      cuGet1dcoords(threadIdx.x, threadIdx.y, BLOCKSIZE, BLOCKSIZE, false);
  int shared_x1 = (idx_first % width_shared) - PADDING;
  int shared_y1 = (idx_first / width_shared) - PADDING;
  // map to shared memory
  int extern_global_x1 = shared_x1 + blockDim.x * blockIdx.x;
  int extern_global_y1 = shared_y1 + blockDim.y * blockIdx.y;
  int shared_x2 = ((idx_first % width_shared) - PADDING);
  int shared_y2 = ((idx_first / width_shared) - PADDING) + width_shared / 2;
  int extern_global_x2 = extern_global_x1;
  int extern_global_y2 = extern_global_y1 + width_shared / 2;
  int extern_global_idx1 =
      cuGet1dcoords(extern_global_x1, extern_global_y1, height, width, false);
  int extern_global_idx2 =
      cuGet1dcoords(extern_global_x2, extern_global_y2, height, width, false);
  if (idx_first < (size_sharedmem / 2) && extern_global_x1 >= 0 &&
      extern_global_x1 < width && extern_global_y1 >= 0 &&
      extern_global_y1 < height && extern_global_x2 >= 0 &&
      extern_global_x2 < width && extern_global_y2 >= 0 &&
      extern_global_y2 < height) {
    shared_data[(shared_y1 + PADDING) * width_shared + (shared_x1 + PADDING)] =
        d_input[extern_global_idx1];
    shared_data[(shared_y2 + PADDING) * width_shared + (shared_x2 + PADDING)] =
        d_input[extern_global_idx2];
  }
  __syncthreads();
  // corner check and compute max score
  int s_coord1d = cuGet1dcoords(threadIdx.x + PADDING, threadIdx.y + PADDING,
                                width_shared, width_shared, false);
  long max_score = 0;
  if (g_coord1d != -1) {
    // in broder-cut area
    max_score = cuGetMaxScore(shared_data, d_score, d_iscorner, threshold,
                              s_coord1d, g_coord1d, height, width);
  }
  __syncthreads();
  // non maximal surpression
  unsigned *score_shared = (unsigned *)shared_data;
  // extern __shared__ unsigned score_shared[];
  // score_shared[g_coord1d] = d_score[g_coord1d];
  if (idx_first < (size_sharedmem / 2) && extern_global_x1 >= 0 &&
      extern_global_x1 < width && extern_global_y1 >= 0 &&
      extern_global_y1 < height && extern_global_x2 >= 0 &&
      extern_global_x2 < width && extern_global_y2 >= 0 &&
      extern_global_y2 < height) {
    score_shared[(shared_y1 + PADDING) * width_shared + (shared_x1 + PADDING)] =
        d_score[extern_global_idx1];
    score_shared[(shared_y2 + PADDING) * width_shared + (shared_x2 + PADDING)] =
        d_score[extern_global_idx2];
  }
  __syncthreads();
  bool erase = false;
  if (g_coord1d != -1) {
    for (size_t i = 0; i < MASKSIZE * MASKSIZE; i++) {
      if (score_shared[s_coord1d + d_mask[i]] > max_score) {
        erase = true;
        break;
      }
    }
  }
  __syncthreads();
  if (erase) {
    d_score[g_coord1d] = 0;
    d_iscorner[g_coord1d] = 0;
  }
  return;
}

void cuInitConstant() {
  size_circle_int = CIRCLEPOINTS * sizeof(int);
  size_mask_int = MASKSIZE * MASKSIZE * sizeof(int);
  width_shared = BLOCKSIZE + (2 * PADDING);
  size_sharedmem = width_shared * width_shared * sizeof(unsigned char);
  h_circle = (int *)malloc(size_circle_int);
  h_mask = (int *)malloc(size_mask_int);
  // h_iscorner = (int *)malloc(size_img_int);
  cuCreateCircle(h_circle, width_shared);
  cuCreateMask(h_mask, width_shared);
  CHECK_ERROR(cudaMalloc((void **)&d_circle, size_circle_int));
  CHECK_ERROR(cudaMalloc((void **)&d_mask, size_mask_int));
  CHECK_ERROR(cudaMemcpyToSymbol(d_circle, h_circle,
                                 size_circle_int)); // constant memory
  CHECK_ERROR(
      cudaMemcpyToSymbol(d_mask, h_mask, size_mask_int)); // constant memory
}
void cuInitMemory(int width, int height) {
  /* compute size and malloc space on device */
  size_img = width * height;
  size_img_char = size_img * sizeof(unsigned char);
  size_img_int = size_img * sizeof(int);
  h_iscorner = (int *)malloc(size_img_int);
  h_score = (unsigned*)malloc(size_img_int);
  CHECK_ERROR(cudaMalloc((void **)&d_img, size_img_char));
  CHECK_ERROR(cudaMalloc((void **)&d_iscorner, size_img_int));
  CHECK_ERROR(cudaMalloc((void **)&d_score, size_img_int));
}
void cuFreeMemory() {
  CHECK_ERROR(cudaFree(d_img));
  CHECK_ERROR(cudaFree(d_score));
  CHECK_ERROR(cudaFree(d_iscorner));
  // CHECK_ERROR(cudaFree(d_mask)); !!! should not free constant memory !!!
  // CHECK_ERROR(cudaFree(d_circle));
  free(h_iscorner);
  free(h_mask);
  free(h_circle);
  free(h_score);
  // CHECK_ERROR(cudaDeviceReset());
}
void cuTransImage(unsigned char *input, cudaStream_t transstream) {
  CHECK_ERROR(cudaMemcpyAsync(d_img, input, size_img_char,
                              cudaMemcpyHostToDevice, transstream));
  cudaStreamSynchronize(transstream);
}
vector<corner> cuConerDetect(unsigned char *input, int width, int height,
                             int threshold, cudaStream_t curstream) {
  // CHECK_ERROR(cudaDeviceReset());
  cudaStream_t streamMem, streamCpt; // memory stream compute stream
  // cudaEvent_t start, stop;
  cudaStreamCreate(&streamMem); // memcpy process
  cudaStreamCreate(&streamCpt); // memcpy process
  cuInitConstant();
  cuInitMemory(width, height);
  cuTransImage(input, streamMem);
  float cu_time;
  // cudaStreamCreate(&streamMem); // memcpy process stream
  // cudaStreamCreate(&streamCpt); // compute stream

  // cuInitConstant();
  // cuInitMemory(width, height);
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  // cuInitMemory(width, height);
  // cudaDeviceSynchronize();

  // reset device memory and send new image to device
  CHECK_ERROR(cudaMemsetAsync(d_score, 0, size_img_int, streamCpt));
  // cudaStreamSynchronize(streamCpt);
  CHECK_ERROR(cudaMemsetAsync(d_iscorner, 0, size_img_int, streamCpt));
  cudaStreamSynchronize(streamCpt);
  // printf("sizeof corner: %lu\n", sizeof(corner));
  dim3 block_size(BLOCKSIZE, BLOCKSIZE);
  dim3 grid_size((int)(width - 1) / BLOCKSIZE + 1,
                 (int)(height - 1) / BLOCKSIZE + 1);
  kernelCornerDetect<<<grid_size, block_size, size_sharedmem * 4, streamCpt>>>(
      d_img, d_score, d_iscorner, height, width, size_img, size_img_char,
      size_img_int, size_circle_int, width_shared, size_sharedmem, threshold);
  // cudaStreamSynchronize(streamCpt);
  // CHECK_ERROR(cudaMemcpyAsync(h_iscorner, d_iscorner, size_img_int,
  CHECK_ERROR(
      cudaMemcpyAsync(h_iscorner, d_iscorner, size_img_int, cudaMemcpyDeviceToHost, streamCpt));
  CHECK_ERROR(
      cudaMemcpyAsync(h_score, d_score, size_img_int, cudaMemcpyDeviceToHost, streamCpt));
  cudaStreamSynchronize(streamCpt);
  // CHECK_ERROR(cudaMemcpyAsync(h_iscorner, d_iscorner, size_img_int,
  // cudaMemcpyDeviceToHost, streamCpt)); cudaStreamSynchronize(streamCpt);
  vector<corner> keypoints;
  for (size_t i = 0; i < size_img; i++) {
    if (h_iscorner[i] != 0) {
      corner cp;
      cp.x = i % width;
      cp.y = i / width;
      cp.score = h_score[i];
      keypoints.push_back(cp);
    }
  }
  // printf("corners: %lu\n", keypoints.size());
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cu_time, start, stop);
  // gpu_full_time += cu_time;
  printf("the gpu passed time is : %.3f ms\n", cu_time);
  cudaStreamDestroy(streamMem);
  cudaStreamDestroy(streamCpt);
  // cudaEventDestroy(start);
  // cudaEventDestroy(stop);
  cuFreeMemory();
  return keypoints;
}

void printCudaInfo() {
  // for fun, just print out some stats on the machine
  int deviceCount = 0;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);

  printf("---------------------------------------------------------\n");
  printf("Found %d CUDA devices\n", deviceCount);

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, i);
    printf("Device %d: %s\n", i, deviceProps.name);
    printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
    printf("   Global mem: %.0f MB\n",
           static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
    printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
  }
  printf("---------------------------------------------------------\n");
}
