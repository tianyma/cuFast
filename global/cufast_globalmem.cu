#include "cpu_fast.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define BLOCKSIZE 32
#define PADDING 3
#define CIRCLEPOINTS 16

static double full_gpu_time = 0.0;
static int p_count = 0;
static int d_imgsize;
static int d_scoresize;
static int d_cornerssize;
static int d_circlesize;

corner *h_corners;
int *h_circle;

int* d_circle;
unsigned char *d_img;
int *d_score;
corner *d_corners;
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
__host__ __device__ void cuCreateCircle(int *circle, int width) {
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

__device__ bool cuLargercompare(unsigned char pixel_val,
                                unsigned char circle_val, int threshold,
                                char sign) {
  if (sign == 1) { // center pixel is darker
    return circle_val > (threshold + pixel_val);
  } else { // center pixel is lighter
    return pixel_val > (threshold + circle_val);
  }
}
__device__ bool cuSimple_check(unsigned char *input, int pixel_id,
                               int *circle, int threshold) {
  /* check top, right, left, bottom pixels */
  int sum = 0;
  for (size_t i = 0; i < CIRCLEPOINTS; i += 4) {
    sum += cuLargercompare(input[pixel_id], input[pixel_id + circle[i]], threshold, 0);
  }
  if (sum < 3) {
    sum = 0;
    for (size_t i = 0; i < CIRCLEPOINTS; i += 4) {
      sum += cuLargercompare(input[pixel_id], input[pixel_id + circle[i]], threshold, 1);
    }
    if (sum < 3) {
      return 0;
    }
  }
  return 1;
}
__device__ bool cuFull_check(unsigned char *input, int pixel_id,
                              int* circle, int threshold) {
  /* check 12 left pixels */
  int sum = 0;
  for (size_t i = 1; i < CIRCLEPOINTS; i++) {
    sum += cuLargercompare(input[pixel_id], input[pixel_id + circle[i]], threshold, 0);
  }
  if (sum < 9) {
    sum = 0;
    for (size_t i = 1; i < CIRCLEPOINTS; i++) {
      sum +=
          cuLargercompare(input[pixel_id], input[pixel_id + circle[i]], threshold, 1);
    }
    if (sum < 9) {
      return 0;
    }
  }
  // cout<<sum<<endl;
  return 1;
}
__device__ int cuGetScoreVal(int pixel_val, int circle_val, int threshold) {
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
__device__ int cuGetscore(unsigned char *input, int pixel_id, int *d_circle, int threshold) {
  unsigned char pixel = input[pixel_id];
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
    score = cuGetScoreVal(
        pixel, input[pixel_id + d_circle[i % CIRCLEPOINTS]], threshold);
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
    return max_score;
  } else {
    return 0;
  }
}
__host__ bool cuAdjcent(corner i, corner j) {
  int xdist = i.x - j.x;
  int ydist = i.y - j.y;
  return ((xdist * xdist + ydist * ydist) <= 512);
}
__host__ vector<corner> cuNonmaxSuppression(vector<corner> cornerpoints) {
  vector<corner> keypoints;
  for (size_t i = 0; i < cornerpoints.size(); i++) {
    for (size_t j = i + 1; j < cornerpoints.size(); j++) {
      if (cuAdjcent(cornerpoints[i], cornerpoints[j])) {
        if (cornerpoints[i].score < cornerpoints[j].score) {
          cornerpoints.erase(cornerpoints.begin() + i);
        } else {
          cornerpoints.erase(cornerpoints.begin() + j);
        }
      }
    }
  }
  for (size_t i = 0; i < cornerpoints.size(); i++) {
    keypoints.push_back(cornerpoints[i]);
  }
  return keypoints;
}
void memoryPreparation(unsigned char *h_img, int img_size, int width) {
  d_imgsize = img_size * sizeof(unsigned char);
  d_scoresize = img_size * sizeof(int);
  d_cornerssize = img_size * sizeof(corner);
  d_circlesize = CIRCLEPOINTS * sizeof(int);
  h_corners = (corner *)malloc(img_size * sizeof(corner));
  h_circle = (int *)malloc(CIRCLEPOINTS * sizeof(int));
  cudaMalloc((void **)&d_img, d_imgsize);
  cudaMalloc((void **)&d_circle, d_circlesize);
  cudaMalloc((void **)&d_score, d_scoresize);
  cudaMalloc((void **)&d_corners, d_cornerssize);
  cuCreateCircle(h_circle, width);
  cudaMemcpy(d_circle, h_circle, d_circlesize, cudaMemcpyHostToDevice); // copy circle kernel to constant memory
  // cudaDeviceSynchronize();
  cudaMemcpy(d_img, h_img, d_imgsize, cudaMemcpyHostToDevice);
  cudaMemset(d_corners, 0, d_cornerssize);
  cudaMemset(d_score, 0, d_scoresize);
  cudaDeviceSynchronize();
}
__global__ void kernelCornerDetect(unsigned char *d_img, int *d_score,
                                   int *d_circle, corner *d_corners,
                                   int d_inputsize, int d_scoresize,
                                   int d_circlesize, int d_cornerssize,
                                   int height, int width, int threshold) {
  int idx_x = threadIdx.x + blockIdx.x * blockDim.x;
  int idx_y = threadIdx.y + blockIdx.y * blockDim.y;
  int idx = cuGet1dcoords(idx_x, idx_y, height, width, true);
  if(idx != -1){
    if(cuSimple_check(d_img, idx, d_circle, threshold)){
      if(cuFull_check(d_img, idx, d_circle, threshold)){
        int c_score = cuGetscore(d_img, idx, d_circle, threshold);
        corner c_point;
        c_point.x = idx % width;
        c_point.y = idx / width;
        c_point.score = c_score;
        d_corners[idx] = c_point;
      }
    }
    return;
  }
}
vector<corner> cuConerDetect(unsigned char *input, int width, int height,
                             int threshold) {
  cudaEvent_t start, stop;
  float cu_time;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  int img_size = width * height;
  memoryPreparation(input, img_size, width);
  // printf("sizeof corner: %lu\n", sizeof(corner));
  dim3 block_size(BLOCKSIZE, BLOCKSIZE);
  dim3 grid_size((int)(width - 1) / BLOCKSIZE + 1,
                 (int)(height - 1) / BLOCKSIZE + 1);
  // cudaMemcpy(d_input, input, img_size, cudaMemcpyHostToDevice);
  kernelCornerDetect<<<grid_size, block_size>>>(
      d_img, d_score, d_circle, d_corners, d_imgsize, d_scoresize,
      d_circlesize, d_cornerssize, height, width, threshold);
  cudaDeviceSynchronize();
  cudaMemcpy(h_corners, d_corners, d_cornerssize, cudaMemcpyDeviceToHost);
  vector<corner> keypoints;
  for (size_t i = 0; i < img_size; i++) {
    if (h_corners[i].score != 0) {
      keypoints.push_back(h_corners[i]);
    }
  }
  keypoints = cuNonmaxSuppression(keypoints);
  cudaDeviceSynchronize();
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&cu_time, start, stop);
  // full_gpu_time += cu_time;
  // p_count++;
  printf("the gpu passed time is : %.3f ms\n", cu_time);
  // printf("%d\n", p_count);
  cudaFree(d_img);
  cudaFree(d_score);
  cudaFree(d_circle);
  cudaFree(d_corners);
  cudaFree(h_corners);
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
