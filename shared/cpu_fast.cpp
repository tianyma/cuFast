#include "cpu_fast.hpp"
#include <getopt.h>
#include <omp.h>
#include <opencv2/core/utility.hpp>

using namespace std;
using namespace cv;
static double opencv_cpu_full_time = 0.0;

static void HandleError(cudaError_t error, const char *file, int line) {
  if (error != cudaSuccess) {
    printf("%s in %s at line %d\n", cudaGetErrorString(error), file, line);
    getchar();
    exit(EXIT_FAILURE);
  }
}

Mat srcimg;
Mat img; // for test

vector<corner> cuConerDetect(unsigned char *input, int width, int height,
                             int threshold, cudaStream_t curstream);
void cuInitConstant();
void cuInitMemory(int width, int height);
void cuTransImage(unsigned char *input, cudaStream_t transstream);
void cuFreeMemory();
void printCudaInfo();

void usage(const char *progname) {
  printf("Usage: %s [options]\n", progname);
  printf("Program Options:\n");
  printf("  -d  --dir <DIR>      Directory name of processing images\n");
  printf("  -u  --use_opencv     Use opencv library fast algorithm "
         "implementation\n");
  printf("  -?  --help             This message\n");
}
void create_circle(int pixel_id, int *circle, int width) {
  circle[0] = pixel_id - PADDING * width;
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

bool largercompare(unsigned char pixel_val, unsigned char circle_val,
                   int threshold, char sign) {
  if (sign == 1) { // center pixel is darker
    return circle_val > (threshold + pixel_val);
  } else { // center pixel is lighter
    return pixel_val > (threshold + circle_val);
  }
}

bool simple_check(unsigned char *input, int pixel_id, int *h_circle,
                  int threshold) {
  /* check top, right, left, bottom pixels */
  int sum = 0;
  for (size_t i = 0; i < CIRCLEPOINTS; i += 4) {
    sum += largercompare(input[pixel_id], input[h_circle[i]], threshold, 0);
  }
  if (sum < 3) {
    sum = 0;
    for (size_t i = 0; i < CIRCLEPOINTS; i += 4) {
      sum += largercompare(input[pixel_id], input[h_circle[i]], threshold, 1);
    }
    if (sum < 3) {
      return 0;
    }
  }
  return 1;
}
bool full_check(unsigned char *input, int pixel_id, int *h_circle,
                unsigned int *score, int threshold) {
  /* check 12 left pixels */
  int sum = 0;
  for (size_t i = 0; i < CIRCLEPOINTS; i++) {
    sum += largercompare(input[pixel_id], input[h_circle[i]], threshold, 0);
  }
  if (sum < 9) {
    sum = 0;
    for (size_t i = 0; i < CIRCLEPOINTS; i++) {
      sum += largercompare(input[pixel_id], input[h_circle[i]], threshold, 1);
    }
    if (sum < 9) {
      return 0;
    }
  }
  // cout<<sum<<endl;
  return 1;
}
int get_score(unsigned char *input, int pixel_id, int *h_circle) {
  int sum = 0;
  for (size_t i = 0; i < CIRCLEPOINTS; i++) {
    sum += abs(input[pixel_id] - input[h_circle[i]]);
  }
  return sum;
}
bool adjcent(corner i, corner j) {
  int xdist = i.x - j.x;
  int ydist = i.y - j.y;
  return ((xdist * xdist + ydist * ydist) <= 32);
}
vector<corner> nonmaxSuppression(vector<corner> cornerpoints) {
  vector<corner> keypoints;
  for (size_t i = 0; i < cornerpoints.size(); i++) {
    for (size_t j = i + 1; j < cornerpoints.size(); j++) {
      if (adjcent(cornerpoints[i], cornerpoints[j])) {
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
vector<corner> cornerDetect(unsigned char *input, unsigned int *score,
                            int *h_circle, int width, int height,
                            int threshold) {
  vector<corner> cornerpoints; // keypoints without nonmax_depression
  vector<corner> keypoints;    // final keypoints
  int pixel_id;
  for (int row = PADDING; row < height - PADDING; row++) {
    for (int col = PADDING; col < width - PADDING; col++) {
      pixel_id = row * width + col;
      create_circle(pixel_id, h_circle, width);
      // getchar();
      if (simple_check(input, pixel_id, h_circle, threshold)) {
        if (full_check(input, pixel_id, h_circle, score, threshold)) {
          int c_score = get_score(input, pixel_id, h_circle);
          corner c_point;
          c_point.x = pixel_id % width;
          c_point.y = pixel_id / width;
          c_point.score = c_score;
          cornerpoints.push_back(c_point);
        }
      }
    }
  }
  keypoints = nonmaxSuppression(cornerpoints);
  return keypoints;
}

void fast(Mat image, vector<KeyPoint> keypoints, int threshold,
          bool nonmax_suppression, bool use_opencv) {
  if (image.channels() > 1) {
    cvtColor(image, image, COLOR_BGR2GRAY);
  }
  if (use_opencv) { // fast corner detect of opencv version
    double cpuStartTime = CycleTimer::currentSeconds();
    Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(threshold);
    detector->detect(image, keypoints);
    double cpuEndTime = CycleTimer::currentSeconds();
    double cpuOverallDuration = cpuEndTime - cpuStartTime;
    opencv_cpu_full_time = cpuOverallDuration;
    printf("CPU Time: %.3f ms \n", 1000.f * opencv_cpu_full_time);

    for (size_t i = 0; i < keypoints.size(); i++) {
      circle(img, Point(keypoints[i].pt.x, keypoints[i].pt.y), 1,
             Scalar(0, 255, 0), 2);
    }
    imshow("img", img);
    waitKey(0);
  }
}

int main(int argc, char **argv) {
  // img = imread("000000.png");

  // parse commandline options ////////////////////////////////////////////
  int opt;
  char *process_dir;
  bool use_opencv = false;
  static struct option long_options[] = {{"dir", 1, 0, 'd'},
                                         {"help", 0, 0, '?'},
                                         {"use_opencv", 0, 0, 'u'},
                                         {0, 0, 0, 0}};

  while ((opt = getopt_long(argc, argv, "d:?u", long_options, NULL)) != EOF) {
    switch (opt) {
    case 'd':
      process_dir = optarg;
      break;
    case 'u':
      use_opencv = true;
      break;
    case '?':
    default:
      usage(argv[0]);
      return 1;
    }
  }
  // end parsing of commandline options //////////////////////////////////////
  std::vector<cv::String> fn;
  cv::glob(process_dir, fn, false);
  if (!fn.size())
    return 0;
  for (size_t i = 0; i < fn.size(); i++) {
    srcimg = imread(fn[i]);

    if (srcimg.channels() > 1) {
      cvtColor(srcimg, img, COLOR_BGR2GRAY);
    }
    vector<KeyPoint> keypoints;
    int threshold = 75;
    bool nonmax_suppression = 1;
    if (use_opencv) {
      fast(img, keypoints, threshold, nonmax_suppression, use_opencv);
    } else {
      /* cuda ORB extractor */
      vector<corner> cupoints;
      cupoints =
          cuConerDetect(img.data, img.cols, img.rows, threshold, streamCpt);
      vector<KeyPoint> cukeypoints;
      Mat desc;
      for (size_t i = 0; i < cupoints.size(); i++) {
        cukeypoints.push_back(
            KeyPoint(cupoints[i].x, cupoints[i].y, 1, -1.0, cupoints[i].score));
      }
     
      for (size_t i = 0; i < cupoints.size(); i++) {
        // printf("%d\n", cupoints[i].score);
        circle(srcimg, Point(cupoints[i].x, cupoints[i].y), 1,
               Scalar(0, 255, 255), 2);
      }
      imshow("img", srcimg);
      waitKey(0);
    }
  }
}