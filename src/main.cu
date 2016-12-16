#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <map>
#include <sys/time.h>
#include <valarray>

#include <hdf5.h>

#include "range.hpp"
#include "utils.hpp"

#define NUM_ROWS 28
#define NUM_COLS 28
#define NUM_CHANNELS 1
#define NUM_DIGITS 10
#define TILE_SIZE 8
#define MperBlock 16
#define FILTER_SIZE 5 

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      printf("Failed to run stmt %s\n", #stmt);                         \
      printf("ERROR - Got CUDA error ...  %s\n", cudaGetErrorString(err));      \
      return ;                                                            \
    }                                                                     \
  } while (0)

static int FLAGS_batch_size = 10000; // number of images....actual value changes at runtime
static std::string FLAGS_testdata{};
static std::string FLAGS_model{};

// Data and reference data dimensions
static int xdims[] = {FLAGS_batch_size, NUM_ROWS, NUM_COLS, NUM_CHANNELS};
static int rdims[] = {FLAGS_batch_size, NUM_DIGITS};

// Model dimensions
static int conv1dims[] = {5, 5, 1, 32}; // rows, cols, #input_feature maps, #output_feature_maps
static int conv2dims[] = {5, 5, 32, 64}; // rows, cols, #input_feature maps, #output_feature_maps
static int fc1dims[]   = {1024, 128}; // not important for convolution or subsampling layers
static int fc2dims[]   = {128, 10}; // not important for convolution or subsampling layers

__constant__ float mask1[5][5][32];

void convolveWrapper(const float *X, const int xdims[4], const float *W, const int wdims[4], float *Y,
                                  const int ydims[4], bool useConstMemory=false);
__global__ void convolve(const float *X, const int xdims[4], const float *W, const int wdims[4], float *Y,
                         const int ydims[4], int W_grid, int n);
__global__ void convolve1(const float *X, const int xdims[4], float *Y, const int ydims[4], int W_grid, int num_images);

void easyConvWrapper (const float *X, const int xdims[4], const float *W, const int wdims[4], float *Y, const int ydims[4]);
__global__ void easyConv (const float *X, const int xdims[4], const float *W, const int wdims[4], float *Y,
                          const int ydims[4], int* W_grid);

static int loadData(float *x, float *y) {
  // Open the data file
  const auto file_id =
      H5Fopen(FLAGS_testdata.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

  // Open the dataset x and y (x is input, y is output)
  const auto x_id = H5Dopen2(file_id, "/x", H5P_DEFAULT); // 'x' is the name of the 'dataset' in the testfile
  const auto y_id = H5Dopen2(file_id, "/y", H5P_DEFAULT); // 'y' is the name of the 'dataset' in the testfile

  // Get the dataset x dimensions
  const auto xspace = H5Dget_space(x_id);
  const auto xndims = H5Sget_simple_extent_ndims(xspace);
  assert(xndims == 4);

  hsize_t input_dims[xndims];
  H5Sget_simple_extent_dims(xspace, input_dims, NULL);
  if (input_dims[0] != FLAGS_batch_size) {
    std::cout << "data size does not match batch size specified!\n";
    return 1; // return error
  }
  std::cout << "input dimensions = " << input_dims[0] << " x " << input_dims[1]
            << " x " << input_dims[2] << " x " << input_dims[3] << "\n";

  // Read the dataset x and y
  check_success(
      H5Dread(x_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, x));
  check_success(
      H5Dread(y_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, y));

  // Close the dataset x and y
  check_success(H5Dclose(x_id));
  check_success(H5Dclose(y_id));

  // Close the file
  check_success(H5Fclose(file_id));

  // return success
  return 0;
}

static void loadModel(float *conv1, float *conv2, float *fc1, float *fc2) {
  // Open the model file
  const auto file_id = H5Fopen(FLAGS_model.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);

  // Open the dataset
  const auto conv1_id = H5Dopen2(file_id, "/conv1", H5P_DEFAULT); // loaded from the model.hdf5 file
  const auto conv2_id = H5Dopen2(file_id, "/conv2", H5P_DEFAULT);
  const auto fc1_id   = H5Dopen2(file_id, "/fc1", H5P_DEFAULT);
  const auto fc2_id   = H5Dopen2(file_id, "/fc2", H5P_DEFAULT);

  // Read the dataset
  check_success(H5Dread(conv1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, conv1));
  check_success(H5Dread(conv2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL,
                        H5P_DEFAULT, conv2));
  check_success(
      H5Dread(fc1_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc1));
  check_success(
      H5Dread(fc2_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, fc2));

  // Close the dataset x and y
  check_success(H5Dclose(conv1_id));
  check_success(H5Dclose(conv2_id));
  check_success(H5Dclose(fc1_id));
  check_success(H5Dclose(fc2_id));

  // Close the file
  check_success(H5Fclose(file_id));
}



// From book chapter Figure 16.4
// X in the input tensor
// W is the tensor of masks
// wdims is either conv1dims or conv2dims
// Y is the output after the convolution, ydims is the dimensions of the output tensor
static void conv_forward_valid(const float *X, const int xdims[4],
                               const float *W, const int wdims[4], float *Y,
                               const int ydims[4]) 
{
  const int filter_h   = wdims[0];
  const int filter_w   = wdims[1];
  const int C = wdims[2];
  const int M = ydims[3];
  auto getWIdx = [wdims] (int p, int q, int c, int m) {
      return p * wdims[1] * wdims[2] * wdims[3] +
             q * wdims[2] * wdims[3] + c * wdims[3] + m;};
  auto getXIdx = [xdims] (int i, int y, int x, int z) {
      return i * xdims[1] * xdims[2] * xdims[3] + y * xdims[2] * xdims[3] + x * xdims[3] + z;
  };
  auto getYIdx = [ydims] (int i, int row, int col, int num_feature_map) {
    return ((i * ydims[1] + row) * ydims[2] + col) * ydims[3] + num_feature_map;
  };
  // M output feature maps, C input feature maps
  // M*C masks
  // Y_i[:,:,m] = sum (Convolve2D X_i[:,:,c] and W[:,:,c,m])
  // Y_i[:,:,:] = sum (Convolve3D m copies of X_i[:,:,c] and W[:,:,c,:]) 
  for (const int i : range(0, ydims[0])) { // number of images
    for (const int m : range(0, M)) { // for each output feature map
      for (const int w : range(0, ydims[2])) { // for each output element
        for (const int h : range(0, ydims[1])) {   
          for (const int p : range(0, filter_h)) { // apply filter
            for (const int q : range(0, filter_w)) {
              for (const int c : range(0, C)) {  // for all input feature maps
                Y[getYIdx(i,h,w,m)] += X[getXIdx(i, h+p, w+q, c)] * W[getWIdx(p,q,c,m)];
              }
            }
          }
        }
      }
    }
  }
}

int multiplyArr(const int* arr, int n) {
  int prod = 1;
  
  for (int i = 0; i < n; i++) {
    prod *= arr[i];
  }
  return prod;
}

void easyConvWrapper(const float *X, const int xdims[4],
                     const float *W, const int wdims[4], float *Y,
                     const int ydims[4]) 
{
  const int W_out = ydims[2];
  const int H_out = ydims[1];
  const int W_grid = ceil(float(W_out)/float(TILE_SIZE)); // number of horizontal tiles per output map
  const int H_grid = ceil(float(H_out)/float(TILE_SIZE)); // number of vertical tiles per output map

  const int N = ydims[0]; //num images
  const int M = wdims[3]; //num output feature_maps

  const int Z = H_grid * W_grid; // total number of tiles
  
  const dim3 blockDim(TILE_SIZE, TILE_SIZE, 1);
  const dim3 gridDim(N, M, Z);
  
  int * deviceW_Grid;
  float* deviceX;
  float* deviceY;
  float* deviceW;
  int* deviceXDims;
  int* deviceYDims;
  int* deviceWDims;
  
  int sizeX = multiplyArr(xdims, 4)*sizeof(float);
  int sizeY = multiplyArr(ydims, 4)*sizeof(float);
  int sizeW = multiplyArr(wdims, 4)*sizeof(float);
  printf("sizeX = %d, sizeY = %d, sizeW = %d\n", sizeX, sizeY, sizeW);
  wbCheck(cudaMalloc(&deviceX, sizeX));
  wbCheck(cudaMalloc(&deviceY, sizeY));
  wbCheck(cudaMalloc(&deviceW, sizeW));
  wbCheck(cudaMalloc(&deviceXDims, 4 * sizeof(int)));
  wbCheck(cudaMalloc(&deviceYDims, 4 * sizeof(int)));
  wbCheck(cudaMalloc(&deviceWDims, 4 * sizeof(int)));
  wbCheck(cudaMalloc(&deviceW_Grid, sizeof(int)));
  
  wbCheck(cudaMemcpy(deviceX, X, sizeX, cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceY, Y, sizeY, cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceW, W, sizeW, cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceXDims, xdims, 4 * sizeof(int), cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceYDims, ydims, 4 * sizeof(int), cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceWDims, wdims, 4 * sizeof(int), cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceW_Grid, &W_grid, sizeof(int), cudaMemcpyHostToDevice));
  
  easyConv<<<gridDim, blockDim>>>(deviceX, deviceXDims, deviceW, deviceWDims, deviceY, deviceYDims, deviceW_Grid);

  wbCheck(cudaMemcpy(Y, deviceY, sizeY, cudaMemcpyDeviceToHost));
  // for(int i = 0; i < multiplyArr(ydims,4); i++) {
  //   if (Y[i] > 0)
  //   printf("Y[%d] = %f\n", i,Y[i]);
  // }
  //Free CUDA Memory
}
//Y is output, X is input, W is the convolution mask
//XYZ Dims: Dimensions -- width, height, depth
__global__ void easyConv (const float *X, const int xdims[4],
                               const float *W, const int wdims[4], float *Y,
                               const int ydims[4], int* W_grid1)
{
  const int filter_h  = wdims[0];
  const int filter_w = wdims[1];
  const int C = wdims[2]; //num input feature_maps
  int W_grid = *W_grid1; // num tiles in horizontal direction
  auto getWIdx = [wdims] (int p, int q, int c, int m) {
      return p * wdims[1] * wdims[2] * wdims[3] +
             q * wdims[2] * wdims[3] + c * wdims[3] + m;};
  auto getXIdx = [xdims] (int i, int y, int x, int z) {
      return i * xdims[1] * xdims[2] * xdims[3] + y * xdims[2] * xdims[3] + x * xdims[3] + z;
  };
  auto getYIdx = [ydims] (int i, int row, int col, int num_feature_map) {
    return ((i * ydims[1] + row) * ydims[2] + col) * ydims[3] + num_feature_map;
  };
  int n, m, h, w, c, p, q;
  n = blockIdx.x;
  m = blockIdx.y;
  h = (blockIdx.z / W_grid) * TILE_SIZE + threadIdx.y;
  w = (blockIdx.z % W_grid) * TILE_SIZE + threadIdx.x;
  float acc = 0.0f;
  if (h < ydims[1] && w < ydims[2]) {
    for (p = 0; p < filter_h; p++){ // loop over KxK  filter
      for (q = 0; q < filter_w; q++){  
        for (c = 0;  c < C; c++) { // sum over all input feature maps      
            if (h+p < xdims[1] && w+q < xdims[2]) {
              acc += (X[getXIdx(n, h + p, w + q, c)] * W[getWIdx(p, q, c, m)]);
              //printf("X[%d,%d,%d,%d] = %f, W[%d,%d,%d,%d] = %f\n", n,h+p,w+q,c, X[getXIdx(n, h + p, w + q, c)], p,q,c,m, W[getWIdx(p, q, c, m)]);
            } 
        }
      }
    }
    Y[getYIdx(n, h, w, m)] = acc;
  }
  //printf("n = %d, h = %d, w = %d, m = %d, Y[%d,%d,%d,%d] = %f\n", n,h,w,m,n,h,w,m,Y[getYIdx(n, h, w, m)]);
  
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

// From book chapter Figure 16.5
static void average_pool(const float *X, const int xdims[4],
                         const int pool_size, float *Y, const int ydims[4]) 
{
  for (const auto i : range(0, ydims[0])) {
    for (const auto m : range(0, ydims[3])) {
      for (const auto w : range(0, ydims[2])) {
        for (const auto h : range(0, ydims[1])) {
          for (const auto p : range(0, pool_size)) {
            for (const auto q : range(0, pool_size)) {
              const auto yoffset =
                  ((i * ydims[1] + h) * ydims[2] + w) * ydims[3] + m;
              const auto xoffset = i * xdims[1] * xdims[2] * xdims[3] +
                                   (pool_size * h + p) * xdims[2] * xdims[3] +
                                   (pool_size * w + q) * xdims[3] + m;
              Y[yoffset] += X[xoffset] / (1.0f * pool_size * pool_size);
            }
          }
        }
      }
    }
  }
}

static void fully_forward(const float *X, const int xdims[2], float *W,
                          const int wdims[2], float *Y, const int ydims[2]) 
{
  for (const auto i : range(0, xdims[0])) {
    for (const auto j : range(0, wdims[1])) {
      float sum = 0;
      for (const auto k : range(0, xdims[1])) {
        sum += X[i * xdims[1] + k] * W[k * wdims[1] + j];
      }
      Y[i * wdims[1] + j] = sum;
    }
  }
}

// Choose the guess with largest score
static void argmax(const float *X, const int xdims[2], int *Y) 
{
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


// Y_i[:,:,m] = sum (Convolve2D X_i[:,:,c] and W[:,:,c,m])
// blockIdx.x corresponds to m
// blockIdx.y corresponds to input_feature_map
// blockIdx.z corresponds to tiles
// threadIdx.x corresponds to m, threadIdx.y and threadIdx.z correspond col,row respectively 
// so each therad computes 2 partial convolutions
    // within a thread block c is the same.. so a thread block uses a patch of TILE_SIZE*2 of X_{i,c} and computes 
    // blockDim.x convolutions with different masks
    // a block reuses each element in a tile TILE_SIZE * TILE_SIZE * blockDim.x times
// we send all the convolutions to the kernel for one image
// as the result from an image comes back we sum the results on the CPU
void convolveWrapper(const float *X, const int xdims[4],
                               const float *W, const int wdims[4], float *Y,
                               const int ydims[4], bool useConstMemory) {
  const int num_images = xdims[0];
  const int C = wdims[2];
  const int M = wdims[3]; //num output feature_maps

  //const int batch_size = 1000; // going to do this many kernel calls at once
  
  const int W_out = ydims[2];
  const int H_out = ydims[1];
  const int W_grid = ceil(float(W_out)/float(TILE_SIZE)); // number of horizontal tiles per output map
  const int H_grid = ceil(float(H_out)/float(TILE_SIZE)); // number of vertical tiles per output map
  const int Z = H_grid * W_grid; // total number of tiles  
  
  int sizeX = multiplyArr(xdims, 4) * sizeof(float);
  int sizeY = multiplyArr(ydims, 4) * sizeof(float); // for each output_feature map element, all the c different values
                                                               // are stored in different locations, and the CPU will sum them
                                                              // we'll store them like Y[i,y,x,c,m] in device, and Y[y,x,c,m] in a temp arr on host
  int sizeW = multiplyArr(wdims, 4) * sizeof(float);  

  float* deviceX;
  float* deviceY;
  float* deviceW;
    
  int* deviceXDims;
  int* deviceYDims;
  int* deviceWDims; 

  // allocate memory 
  wbCheck(cudaMalloc(&deviceX, sizeX));
  wbCheck(cudaMalloc(&deviceY, sizeY));
  wbCheck(cudaMalloc(&deviceXDims, 4 * sizeof(int)));
  wbCheck(cudaMalloc(&deviceYDims, 4 * sizeof(int)));
  

 
  // copy memory
  wbCheck(cudaMemcpy(deviceX, X, sizeX, cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceY, Y, sizeY, cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceXDims, xdims, 4 * sizeof(int), cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceYDims, ydims, 4 * sizeof(int), cudaMemcpyHostToDevice));

  dim3 gridDim((M/MperBlock) * num_images, C, Z); // ASSUMES - that M is a multiple of 16
  dim3 blockDim(MperBlock,TILE_SIZE,TILE_SIZE);
  

  if (useConstMemory) {  
    wbCheck(cudaMemcpyToSymbol(mask1, W, sizeW));
    convolve1<<<gridDim, blockDim>>>(deviceX, deviceXDims, deviceY, deviceYDims, W_grid, num_images);
  }
  else {
    wbCheck(cudaMalloc(&deviceWDims, 4 * sizeof(int)));
    wbCheck(cudaMalloc(&deviceW, sizeW));
    wbCheck(cudaMemcpy(deviceWDims, wdims, 4 * sizeof(int), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(deviceW, W, sizeW, cudaMemcpyHostToDevice));
    convolve<<<gridDim, blockDim>>>(deviceX, deviceXDims, deviceW, deviceWDims, deviceY, deviceYDims, W_grid, num_images);
  }
 

  cudaMemcpy(Y, deviceY, sizeY, cudaMemcpyDeviceToHost);

  wbCheck(cudaFree(deviceX));
  wbCheck(cudaFree(deviceY));
  if(!useConstMemory) wbCheck(cudaFree(deviceW));
  wbCheck(cudaFree(deviceXDims));
  wbCheck(cudaFree(deviceYDims));
  wbCheck(cudaFree(deviceWDims));
   
}


//change so each thread computes two convolutions
__global__ void convolve(const float *X, const int xdims[4],
                        const float *W, const int wdims[4], float *Y,
                        const int ydims[4], int W_grid, int num_images) {
  
  
  #define tx threadIdx.x
  #define ty threadIdx.y
  #define tz threadIdx.z
  
  const int m = (blockIdx.x / num_images) * blockDim.x + tx;
  const int h = (blockIdx.z / W_grid) * TILE_SIZE + tz;
  const int w = (blockIdx.z % W_grid) * TILE_SIZE + ty;
  const int n = blockIdx.x % num_images;
  const int c = blockIdx.y; // each thread does a convolution of X[n, h:h+5, w:w+5,m] with W[:, :, c, m]

  const int input_TILE_SIZE = TILE_SIZE + FILTER_SIZE - 1; // ASSUMES :  TILE_SIZE + FILTER_SIZE - 1 < 2*TILE_SIZE 
  
  __shared__ float sharedX[input_TILE_SIZE][input_TILE_SIZE];
  __shared__ float sharedW[FILTER_SIZE][FILTER_SIZE][MperBlock];

  if (tx == 0) {
    
      int hbase = (h-tz);
      int wbase = w-ty;
      for(int i = h; i < hbase + input_TILE_SIZE; i+= TILE_SIZE) {
        for (int j = w; j < wbase + input_TILE_SIZE; j+= TILE_SIZE) {
          if (i < xdims[1] && j < xdims[2])
            sharedX[i-hbase][j-wbase] = X[n * xdims[1] * xdims[2] * xdims[3] + i * xdims[2] * xdims[3] + j * xdims[3] + c];
          else
            sharedX[i-hbase][j-wbase] = 0.0f;
        }
      }
  }
  if (tz < FILTER_SIZE && ty < FILTER_SIZE) {
    sharedW[tz][ty][tx] = W[tz * wdims[1] * wdims[2] * wdims[3] + ty * wdims[2] * wdims[3] + c * wdims[3] + m]; // ASSUMES : blockdim.z, blockdim.y >= filter_h,filter_w
  }
  
  __syncthreads();
  


  if (h < ydims[1] && w < ydims[2]) {

    float sum = 0.0f;

    for (int p = 0; p < FILTER_SIZE; p++) {
      for (int q = 0; q < FILTER_SIZE; q++) {
        sum += sharedX[tz + p][ty + q] * sharedW[p][q][tx];
      }
    }
    atomicAdd(&(Y[n * ydims[1] * ydims[2] * ydims[3] + h * ydims[2] * ydims[3] + w * ydims[3] + m]), sum);

  }

}

__global__ void convolve1(const float *X, const int xdims[4], float *Y,
                          const int ydims[4], int W_grid, int num_images) {
  
  
  #define tx threadIdx.x
  #define ty threadIdx.y
  #define tz threadIdx.z
  
  const int m = (blockIdx.x / num_images) * blockDim.x + tx;
  const int h = (blockIdx.z / W_grid) * TILE_SIZE + tz;
  const int w = (blockIdx.z % W_grid) * TILE_SIZE + ty;
  const int n = blockIdx.x % num_images;
  const int c = blockIdx.y; // each thread does a convolution of X[n, h:h+5, w:w+5,m] with W[:, :, c, m]

  const int input_TILE_SIZE = TILE_SIZE + FILTER_SIZE - 1; // ASSUMES :  TILE_SIZE + FILTER_SIZE - 1 < 2*TILE_SIZE 
  
  __shared__ float sharedX[input_TILE_SIZE][input_TILE_SIZE];

  if (tx == 0) {
    
      int hbase = (h-tz);
      int wbase = w-ty;
      for(int i = h; i < hbase + input_TILE_SIZE; i+= TILE_SIZE) {
        for (int j = w; j < wbase + input_TILE_SIZE; j+= TILE_SIZE) {
          if (i < xdims[1] && j < xdims[2])
            sharedX[i-hbase][j-wbase] = X[n * xdims[1] * xdims[2] * xdims[3] + i * xdims[2] * xdims[3] + j * xdims[3] + c];
          else
            sharedX[i-hbase][j-wbase] = 0.0f;
        }
      }
  }
  
  __syncthreads();
  

  if (h < ydims[1] && w < ydims[2]) {

    float sum = 0.0f;

    for (int p = 0; p < FILTER_SIZE; p++) {
      for (int q = 0; q < FILTER_SIZE; q++) {
        sum += sharedX[tz + p][ty + q] * mask1[p][q][m];
      }
    }
    atomicAdd(&(Y[n * ydims[1] * ydims[2] * ydims[3] + h * ydims[2] * ydims[3] + w * ydims[3] + m]), sum);

  }

}


// Forward operation for the CNN, a combination of conv layer + average pooling
// + relu
void forward_operation(float *x, float *conv1, float *conv2, float *fc1,
                       float *fc2, int *out) {
  // conv layer
  const int adims[] = {xdims[0], (xdims[1] - conv1dims[0] + 1),
                       (xdims[2] - conv1dims[1] + 1), conv1dims[3]};
  auto a = zeros<float>(adims);
  convolveWrapper(x, xdims, conv1, conv1dims, a, adims, true);
  //easyConvWrapper(x, xdims, conv1, conv1dims, a, adims);
  //conv_forward_valid(x, xdims, conv1, conv1dims, a, adims);
  
  /// relu layer
  relu4(a, adims);

  // average pooling
  const int pool_size = 2;
  const int bdims[]   = {adims[0], adims[1] / pool_size, adims[2] / pool_size,
                       adims[3]};
  auto b = zeros<float>(bdims);
  average_pool(a, adims, pool_size, b, bdims);

  // conv layer
  const int cdims[] = {bdims[0], (bdims[1] - conv2dims[0] + 1),
                       (bdims[2] - conv2dims[1] + 1), conv2dims[3]};
  auto c = zeros<float>(cdims);
  convolveWrapper(b, bdims, conv2, conv2dims, c, cdims, false);
  //easyConvWrapper(b, bdims, conv2, conv2dims, c, cdims);
  //conv_forward_valid(b, bdims, conv2, conv2dims, c, cdims);
  // relu
  relu4(c, cdims);

  // average pooling
  const int ddims[] = {cdims[0], cdims[1] / pool_size, cdims[2] / pool_size,
                       cdims[3]};
  auto d = zeros<float>(ddims);
  average_pool(c, cdims, pool_size, d, ddims);

  // reshape
  const int ddims2[] = {ddims[0], ddims[1] * ddims[2] * ddims[3]};

  // matrix multiplication
  const int edims[] = {ddims[0], fc1dims[1]};
  auto e            = zeros<float>(edims);
  fully_forward(d, ddims2, fc1, fc1dims, e, edims);

  // relu
  relu2(e, edims);

  // matrix multiplication
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

  if (argc != 3 && argc != 4) {
    std::cerr << "\n"
              << "This program performs the forward opertion step for "
                 "Convolutional Neural Network(CNN).  "
                 "Sample usage: \n"
              << argv[0]
              << " [../data/test10.hdf5] [../data/model.hdf5] [10]\n";
    return -1;
  }
  FLAGS_testdata = std::string(argv[1]);
  FLAGS_model    = std::string(argv[2]);
  if (argc == 3) {
    const std::map<std::string, int> default_batch_sizes{
        {"../data/test2.hdf5", 2},
        {"../data/test10.hdf5", 10},
        {"../data/test100.hdf5", 100},
        {"../data/testfull.hdf5", 10000}};
    const auto batch_size_in_map = default_batch_sizes.find(FLAGS_testdata);
    if (batch_size_in_map == default_batch_sizes.end()) {
      std::cerr << "\nERROR:: Unrecognized file " << FLAGS_testdata << " batch_size must be specified.\n";
      return -1;
    }
    FLAGS_batch_size = batch_size_in_map->second;
  } else if (argc == 4) {
    FLAGS_batch_size = atoi(argv[3]);
  }
  xdims[0] = FLAGS_batch_size; // number of images
  rdims[0] = FLAGS_batch_size; // number of images

  // Load data into x and y
  float *x = allocate<float>(xdims);
  float *y = allocate<float>(rdims);
  loadData(x, y);

  // Load model
  float *conv1 = allocate<float>(conv1dims);
  float *conv2 = allocate<float>(conv2dims);
  float *fc1   = allocate<float>(fc1dims);
  float *fc2   = allocate<float>(fc2dims);
  loadModel(conv1, conv2, fc1, fc2);

  // Perform foward opertion
  int *out = zeros<int>(FLAGS_batch_size);

  // get start time
  const auto start = now();

  forward_operation(x, conv1, conv2, fc1, fc2, out);

  // get end time
  const auto end = now();

  // get elapsed time in milliseconds
  const auto elapsed =
      std::chrono::duration<double, std::milli>(end - start).count();

  // Get reference
  int *ref = zeros<int>(FLAGS_batch_size);
  argmax(y, rdims, ref);

  // Calculate correctness
  int num_correct = 0;
  for (const auto i : range(0, FLAGS_batch_size)) {
    if (out[i] == ref[i]) {
      num_correct++;
    }
  }
  std::cout << "Done with " << FLAGS_batch_size << " queries in "
            << "elapsed = " << elapsed << " milliseconds. Correctness: "
            << static_cast<float>(num_correct) / FLAGS_batch_size << "\n";

  delete[] x;
  delete[] y;
  delete[] conv1;
  delete[] conv2;
  delete[] fc1;
  delete[] fc2;
  delete[] out;
  delete[] ref;

  return 0;
}
