#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <valarray>
#include <gflags/gflags.h>

#include <fmt/format.h>
#include <hdf5.h>
#include <range.hpp>

#include "utils.hpp"

using namespace util::lang;

#define DEFAULT_DATA_PATH "../data/data.hdf5"
#define DEFAULT_MODEL_PATH "../data/model.hdf5"


#define BATCH_SIZE 10000
#define NUM_ROWS 28
#define NUM_COLS 28
#define NUM_CHANNELS 1
#define NUM_DIGITS 10

DECLARE_string(data);
DECLARE_string(model);

DEFINE_string(data, "", "path to the hdf5 data file");
DEFINE_string(model, "", "path to the hdf5 model file");

// Data and reference data dimensions
static int xdims[] = {BATCH_SIZE, NUM_ROWS, NUM_COLS, NUM_CHANNELS};
static int rdims[] = {BATCH_SIZE, NUM_DIGITS};

// Model dimensions 
static int conv1dims[] = {5, 5, 1, 32};
static int conv2dims[] = {5, 5, 32, 64};
static int fc1dims[]   = {1024, 128};
static int fc2dims[]   = {128, 10};


static void loadData(float *x, float *y) {
  // Open the data file
  const auto file_id = H5Fopen(FLAGS_data.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

  // Open the dataset x and y
  const auto x_id = H5Dopen2(file_id, "/x", H5P_DEFAULT);
  const auto y_id = H5Dopen2(file_id, "/y", H5P_DEFAULT);

  // Get the dataset x dimensions
  const auto xspace = H5Dget_space(x_id);
  const auto xndims = H5Sget_simple_extent_ndims(xspace);
  assert(xndims == 4);

  hsize_t xdims[xndims];
  H5Sget_simple_extent_dims(xspace, xdims, NULL);
  fmt::print("xdims = {} x {} x {} x {}\n", xdims[0], xdims[1], xdims[2], xdims[3]);

  // Read the dataset x and y
  check_success(H5Dread(x_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, x));
  check_success(H5Dread(y_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, y));

  // Close the dataset x and y
  check_success(H5Dclose(x_id));
  check_success(H5Dclose(y_id));

  // Close the file
  check_success(H5Fclose(file_id));
}

static void loadModel(float *conv1, float *conv2, float *fc1, float *fc2) {

  // Open the model file
  const auto file_id = H5Fopen(FLAGS_model.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

  // Open the dataset
  const auto conv1_id = H5Dopen2(file_id, "/conv1", H5P_DEFAULT);
  const auto conv2_id = H5Dopen2(file_id, "/conv2", H5P_DEFAULT);
  const auto fc1_id   = H5Dopen2(file_id, "/fc1", H5P_DEFAULT);
  const auto fc2_id   = H5Dopen2(file_id, "/fc2", H5P_DEFAULT);

  // Read the dataset
  check_success(H5Dread(conv1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, conv1));
  check_success(H5Dread(conv2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, conv2));
  check_success(H5Dread(fc1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc1));
  check_success(H5Dread(fc2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc2));

  // Close the dataset x and y
  check_success(H5Dclose(conv1_id));
  check_success(H5Dclose(conv2_id));
  check_success(H5Dclose(fc1_id));
  check_success(H5Dclose(fc2_id));

  // Close the file
  check_success(H5Fclose(file_id));
}

// Convolution layer
static void conv_forward_valid(const float *X, const int xdims[4], const float *W, const int wdims[4], float *Y, const int ydims[4]) {
  const auto filter_h    = wdims[0];
  const auto filter_w    = wdims[1];
  const auto in_channel  = wdims[2];

  for (const auto i : range(0, ydims[0])) {
    for (const auto m : range(0, ydims[3])) {
      for (const auto w : range(0, ydims[2])) {
        for (const auto h : range(0, ydims[1])) {
          for (const auto p : range(0, filter_h)) {
            for (const auto q : range(0, filter_w)) {
              for (const auto c : range(0, in_channel)) {
                Y[((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m] +=
                    X[i * xdims[1] * xdims[2] * xdims[3] + (h + p) * xdims[2] * xdims[3] + (w + q) * xdims[3] + c] *
                    W[p * wdims[1] * wdims[2] * wdims[3] + q * wdims[2] * wdims[3] + c * wdims[3] + m];
              }
            }
          }
        }
      }
    }
  }
}

// Recified linear unit 4d
static void relu4(float *X, const int xdims[4]) {
  for (const auto i : range(0, xdims[0] * xdims[1] * xdims[2] * xdims[3])) {
    X[i] = (X[i] < 0) ? 0 : X[i];
  }
}

// Recified linear unit 2d
static void relu2(float *X, const int xdims[2]) {
  for (const auto i : range(0, xdims[0] * xdims[1])) {
    X[i] = (X[i] < 0) ? 0 : X[i];
  }
}

static void average_pool(const float *X, const int xdims[4], const int pool_size, float *Y, const int ydims[4]) {

  for (const auto i : range(0, ydims[0]))
    for (const auto m : range(0, ydims[3]))
      for (const auto w : range(0, ydims[2]))
        for (const auto h : range(0, ydims[1]))
          for (const auto p : range(0, pool_size))
            for (const auto q : range(0, pool_size))
              Y[((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m] +=
                  X[i * xdims[1] * xdims[2] * xdims[3] + (pool_size * h + p) * xdims[2] * xdims[3] + (pool_size * w + q) * xdims[3] + m]
                  / (1.0f * pool_size * pool_size);
}

static void fully_forward(const float *X, const int xdims[2], float *W, const int wdims[2], float *Y, const int ydims[2]) {
  int i, j, k;
  float sum;

  for (i = 0; i < xdims[0]; ++i) {
    for (j = 0; j < wdims[1]; ++j) {
      sum = 0;
      for (k = 0; k < xdims[1]; ++k) {
        sum += X[i * xdims[1] + k] * W[k * wdims[1] + j];
      }
      Y[i * wdims[1] + j] = sum;
    }
  }
}

static void argmax(const float *X, const int xdims[2], int *Y) {
  for (const auto i : range(0, xdims[0])) {
    auto max_idx = 0;
    auto max     = X[i * xdims[1]];
    for (const auto j : range(0, xdims[1])) {
      const auto elem = X[(i * xdims[1]) + j];
      if (elem > max) {
        max_idx = j;
        max     = elem;
      }
    }
    Y[i] = max_idx;
  }
}

void forward_operation(float *x, float *conv1, float *conv2, float *fc1, float *fc2, int *out) {

  const int adims[] = {xdims[0], (xdims[1] - conv1dims[0] + 1), (xdims[2] - conv1dims[1] + 1), conv1dims[3]};
  auto a            = zeros<float>(adims);
  conv_forward_valid(x, xdims, conv1, conv1dims, a, adims);

  relu4(a, adims);

  const int pool_size = 2;
  const int bdims[]   = {adims[0], adims[1] / pool_size, adims[2] / pool_size, adims[3]};
  auto b              = zeros<float>(bdims);
  average_pool(a, adims, pool_size, b, bdims);

  const int cdims[] = {bdims[0], (bdims[1] - conv2dims[0] + 1), (bdims[2] - conv2dims[1] + 1), conv2dims[3]};
  auto c            = zeros<float>(cdims);
  conv_forward_valid(b, bdims, conv2, conv2dims, c, cdims);

  relu4(c, cdims);

  const int ddims[] = {cdims[0], cdims[1] / pool_size, cdims[2] / pool_size, cdims[3]};
  auto d            = zeros<float>(ddims);
  average_pool(c, cdims, pool_size, d, ddims);

  const int ddims2[] = {ddims[0], ddims[1] * ddims[2] * ddims[3]};
  const int edims[]  = {ddims[0], fc1dims[1]};
  auto e             = zeros<float>(edims);
  fully_forward(d, ddims2, fc1, fc1dims, e, edims);

  relu2(e, edims);

  const int fdims[] = {edims[0], fc2dims[1]};
  auto f            = zeros<float>(fdims);
  fully_forward(e, edims, fc2, fc2dims, f, fdims);

  argmax(f, fdims, out);

  delete[] a;
  delete[] b;
  delete[] c;
  delete[] d;
  delete[] e;
  delete[] f;
}

int main(int argc, char **argv) {
  std::string usage = fmt::format("\nThis program.  Sample usage: {} -data [{}] -model [{}]\n", argv[0], DEFAULT_DATA_PATH, DEFAULT_MODEL_PATH);
  
  google::SetUsageMessage(usage);
  
  google::ParseCommandLineFlags(&argc, &argv, true);

  if (FLAGS_data == "" || FLAGS_model == "") {
    std::cerr << usage << "\n";
    return -1;
  }

  // Load data into x and y
  float *x = (float *) malloc(BATCH_SIZE * NUM_ROWS * NUM_COLS * NUM_CHANNELS * sizeof(float));
  float *y = (float *) malloc(BATCH_SIZE * NUM_DIGITS * sizeof(float));
  loadData(x, y);

  // Load model
  float *conv1 = (float *) malloc(5 * 5 * 1 * 32 * sizeof(float));
  float *conv2 = (float *) malloc(5 * 5 * 32 * 64 * sizeof(float));
  float *fc1   = (float *) malloc(1024 * 128 * sizeof(float));
  float *fc2   = (float *) malloc(128 * 10 * sizeof(float));
  loadModel(conv1, conv2, fc1, fc2);

  int *out = (int *) calloc(BATCH_SIZE, sizeof(int));
  forward_operation(x, conv1, conv2, fc1, fc2, out);

  int *ref = (int *) calloc(BATCH_SIZE, sizeof(int));
  argmax(y, rdims, ref);

  int num_correct = 0;
  for (int i = 0; i < BATCH_SIZE; ++i) {
    if (out[i] == ref[i])
      num_correct++;
  }
  fmt::printf("Done. Correctness: %f\n", (float) num_correct / BATCH_SIZE);

  free(x);
  free(y);
  free(conv1);
  free(conv2);
  free(fc1);
  free(fc2);
  free(out);

  return 0;
}
