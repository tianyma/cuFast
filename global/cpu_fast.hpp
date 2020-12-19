#ifndef CPUFAST_HPP
#define CPUFAST_HPP

#include "CycleTimer.h"
#include <iostream>
#include <stdlib.h>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 32
#define PADDING 3
#define CIRCLEPOINTS 16
#define CONSECUTIVE 9
#define MASKSIZE 3

typedef struct corner {
  unsigned score;
  unsigned x;
  unsigned y;
} corner;

extern cudaEvent_t start, stop;
extern cudaStream_t streamMem, streamCpt; // memory stream compute stream
extern unsigned char* d_img;

#define CHECK_ERROR(error) (HandleError(error, __FILE__, __LINE__))
static void HandleError(cudaError_t error, const char *file, int line); 

#endif