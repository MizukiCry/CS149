#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

#define DEBUG

#ifdef DEBUG
#define cudaCheckError(ans)                                                    \
  { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line,
                       bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}
#else
#define cudaCheckError(ans) ans
#endif

struct GlobalConstants {

  SceneName sceneName;

  int numCircles;
  float *position;
  float *velocity;
  float *color;
  float *radius;

  int imageWidth;
  int imageHeight;
  float *imageData;

  float invWidth;
  float invHeight;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int cuConstNoiseYPermutationTable[256];
__constant__ int cuConstNoiseXPermutationTable[256];
__constant__ float cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float cuConstColorRamp[COLOR_MAP_SIZE][3];

// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "circleBoxTest.cu_inl"
#include "lookupColor.cu_inl"
#include "noiseCuda.cu_inl"

// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {

  int imageX = blockIdx.x * blockDim.x + threadIdx.x;
  int imageY = blockIdx.y * blockDim.y + threadIdx.y;

  int width = cuConstRendererParams.imageWidth;
  int height = cuConstRendererParams.imageHeight;

  if (imageX >= width || imageY >= height)
    return;

  int offset = 4 * (imageY * width + imageX);
  float shade = .4f + .45f * static_cast<float>(height - imageY) / height;
  float4 value = make_float4(shade, shade, shade, 1.f);

  // write to global memory: As an optimization, I use a float4
  // store, that results in more efficient code than if I coded this
  // up as four seperate fp32 stores.
  *(float4 *)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {

  int imageX = blockIdx.x * blockDim.x + threadIdx.x;
  int imageY = blockIdx.y * blockDim.y + threadIdx.y;

  int width = cuConstRendererParams.imageWidth;
  int height = cuConstRendererParams.imageHeight;

  if (imageX >= width || imageY >= height)
    return;

  int offset = 4 * (imageY * width + imageX);
  float4 value = make_float4(r, g, b, a);

  // write to global memory: As an optimization, I use a float4
  // store, that results in more efficient code than if I coded this
  // up as four seperate fp32 stores.
  *(float4 *)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
//
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
  const float dt = 1.f / 60.f;
  const float pi = 3.14159;
  const float maxDist = 0.25f;

  float *velocity = cuConstRendererParams.velocity;
  float *position = cuConstRendererParams.position;
  float *radius = cuConstRendererParams.radius;

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= cuConstRendererParams.numCircles)
    return;

  if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update
    return;
  }

  // determine the fire-work center/spark indices
  int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
  int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

  int index3i = 3 * fIdx;
  int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
  int index3j = 3 * sIdx;

  float cx = position[index3i];
  float cy = position[index3i + 1];

  // update position
  position[index3j] += velocity[index3j] * dt;
  position[index3j + 1] += velocity[index3j + 1] * dt;

  // fire-work sparks
  float sx = position[index3j];
  float sy = position[index3j + 1];

  // compute vector from firework-spark
  float cxsx = sx - cx;
  float cysy = sy - cy;

  // compute distance from fire-work
  float dist = sqrt(cxsx * cxsx + cysy * cysy);
  if (dist > maxDist) { // restore to starting position
    // random starting position on fire-work's rim
    float angle = (sfIdx * 2 * pi) / NUM_SPARKS;
    float sinA = sin(angle);
    float cosA = cos(angle);
    float x = cosA * radius[fIdx];
    float y = sinA * radius[fIdx];

    position[index3j] = position[index3i] + x;
    position[index3j + 1] = position[index3i + 1] + y;
    position[index3j + 2] = 0.0f;

    // travel scaled unit length
    velocity[index3j] = cosA / 5.0;
    velocity[index3j + 1] = sinA / 5.0;
    velocity[index3j + 2] = 0.0f;
  }
}

// kernelAdvanceHypnosis
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= cuConstRendererParams.numCircles)
    return;

  float *radius = cuConstRendererParams.radius;

  float cutOff = 0.5f;
  // place circle back in center after reaching threshold radisus
  if (radius[index] > cutOff) {
    radius[index] = 0.02f;
  } else {
    radius[index] += 0.01f;
  }
}

// kernelAdvanceBouncingBalls
//
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() {
  const float dt = 1.f / 60.f;
  const float kGravity = -2.8f; // sorry Newton
  const float kDragCoeff = -0.8f;
  const float epsilon = 0.001f;

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= cuConstRendererParams.numCircles)
    return;

  float *velocity = cuConstRendererParams.velocity;
  float *position = cuConstRendererParams.position;

  int index3 = 3 * index;
  // reverse velocity if center position < 0
  float oldVelocity = velocity[index3 + 1];
  float oldPosition = position[index3 + 1];

  if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition
    return;
  }

  if (position[index3 + 1] < 0 && oldVelocity < 0.f) { // bounce ball
    velocity[index3 + 1] *= kDragCoeff;
  }

  // update velocity: v = u + at (only along y-axis)
  velocity[index3 + 1] += kGravity * dt;

  // update positions (only along y-axis)
  position[index3 + 1] += velocity[index3 + 1] * dt;

  if (fabsf(velocity[index3 + 1] - oldVelocity) < epsilon &&
      oldPosition < 0.0f &&
      fabsf(position[index3 + 1] - oldPosition) < epsilon) { // stop ball
    velocity[index3 + 1] = 0.f;
    position[index3 + 1] = 0.f;
  }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= cuConstRendererParams.numCircles)
    return;

  const float dt = 1.f / 60.f;
  const float kGravity = -1.8f; // sorry Newton
  const float kDragCoeff = 2.f;

  int index3 = 3 * index;

  float *positionPtr = &cuConstRendererParams.position[index3];
  float *velocityPtr = &cuConstRendererParams.velocity[index3];

  // loads from global memory
  float3 position = *((float3 *)positionPtr);
  float3 velocity = *((float3 *)velocityPtr);

  // hack to make farther circles move more slowly, giving the
  // illusion of parallax
  float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

  // add some noise to the motion to make the snow flutter
  float3 noiseInput;
  noiseInput.x = 10.f * position.x;
  noiseInput.y = 10.f * position.y;
  noiseInput.z = 255.f * position.z;
  float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
  noiseForce.x *= 7.5f;
  noiseForce.y *= 5.f;

  // drag
  float2 dragForce;
  dragForce.x = -1.f * kDragCoeff * velocity.x;
  dragForce.y = -1.f * kDragCoeff * velocity.y;

  // update positions
  position.x += velocity.x * dt;
  position.y += velocity.y * dt;

  // update velocities
  velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
  velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

  float radius = cuConstRendererParams.radius[index];

  // if the snowflake has moved off the left, right or bottom of
  // the screen, place it back at the top and give it a
  // pseudorandom x position and velocity.
  if ((position.y + radius < 0.f) || (position.x + radius) < -0.f ||
      (position.x - radius) > 1.f) {
    noiseInput.x = 255.f * position.x;
    noiseInput.y = 255.f * position.y;
    noiseInput.z = 255.f * position.z;
    noiseForce = cudaVec2CellNoise(noiseInput, index);

    position.x = .5f + .5f * noiseForce.x;
    position.y = 1.35f + radius;

    // restart from 0 vertical velocity.  Choose a
    // pseudo-random horizontal velocity.
    velocity.x = 2.f * noiseForce.y;
    velocity.y = 0.f;
  }

  // store updated positions and velocities to global memory
  *((float3 *)positionPtr) = position;
  *((float3 *)velocityPtr) = velocity;
}

// shadePixel -- (CUDA device code)
//
// given a pixel and a circle, determines the contribution to the
// pixel from the circle.  Update of the image is done in this
// function.  Called by kernelRenderCircles()
__device__ __inline__ void shadePixel(int circleIndex, float2 pixelCenter,
                                      float3 p, float4 *imagePtr) {

  float diffX = p.x - pixelCenter.x;
  float diffY = p.y - pixelCenter.y;
  float pixelDist = diffX * diffX + diffY * diffY;

  float rad = cuConstRendererParams.radius[circleIndex];
  ;
  float maxDist = rad * rad;

  // circle does not contribute to the image
  if (pixelDist > maxDist)
    return;

  float3 rgb;
  float alpha;

  // there is a non-zero contribution.  Now compute the shading value

  // suggestion: This conditional is in the inner loop.  Although it
  // will evaluate the same for all threads, there is overhead in
  // setting up the lane masks etc to implement the conditional.  It
  // would be wise to perform this logic outside of the loop next in
  // kernelRenderCircles.  (If feeling good about yourself, you
  // could use some specialized template magic).
  if (cuConstRendererParams.sceneName == SNOWFLAKES ||
      cuConstRendererParams.sceneName == SNOWFLAKES_SINGLE_FRAME) {

    const float kCircleMaxAlpha = .5f;
    const float falloffScale = 4.f;

    float normPixelDist = sqrt(pixelDist) / rad;
    rgb = lookupColor(normPixelDist);

    float maxAlpha = .6f + .4f * (1.f - p.z);
    maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f),
                                       0.f); // kCircleMaxAlpha * clamped value
    alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);

  } else {
    // simple: each circle has an assigned color
    int index3 = 3 * circleIndex;
    rgb = *(float3 *)&(cuConstRendererParams.color[index3]);
    alpha = .5f;
  }

  float oneMinusAlpha = 1.f - alpha;

  // BEGIN SHOULD-BE-ATOMIC REGION
  // global memory read

  float4 existingColor = *imagePtr;
  float4 newColor;
  newColor.x = alpha * rgb.x + oneMinusAlpha * existingColor.x;
  newColor.y = alpha * rgb.y + oneMinusAlpha * existingColor.y;
  newColor.z = alpha * rgb.z + oneMinusAlpha * existingColor.z;
  newColor.w = alpha + existingColor.w;

  // global memory write
  *imagePtr = newColor;

  // END SHOULD-BE-ATOMIC REGION
}

// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a circle.  Since there is no protection to
// ensure order of update or mutual exclusion on the output image, the
// resulting image will be incorrect.
__global__ void kernelRenderCircles() {

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= cuConstRendererParams.numCircles)
    return;

  int index3 = 3 * index;

  // read position and radius
  float3 p = *(float3 *)(&cuConstRendererParams.position[index3]);
  float rad = cuConstRendererParams.radius[index];

  // compute the bounding box of the circle. The bound is in integer
  // screen coordinates, so it's clamped to the edges of the screen.
  short imageWidth = cuConstRendererParams.imageWidth;
  short imageHeight = cuConstRendererParams.imageHeight;
  short minX = static_cast<short>(imageWidth * (p.x - rad));
  short maxX = static_cast<short>(imageWidth * (p.x + rad)) + 1;
  short minY = static_cast<short>(imageHeight * (p.y - rad));
  short maxY = static_cast<short>(imageHeight * (p.y + rad)) + 1;

  // a bunch of clamps.  Is there a CUDA built-in for this?
  short screenMinX = (minX > 0) ? ((minX < imageWidth) ? minX : imageWidth) : 0;
  short screenMaxX = (maxX > 0) ? ((maxX < imageWidth) ? maxX : imageWidth) : 0;
  short screenMinY =
      (minY > 0) ? ((minY < imageHeight) ? minY : imageHeight) : 0;
  short screenMaxY =
      (maxY > 0) ? ((maxY < imageHeight) ? maxY : imageHeight) : 0;

  float invWidth = 1.f / imageWidth;
  float invHeight = 1.f / imageHeight;

  // for all pixels in the bonding box
  for (int pixelY = screenMinY; pixelY < screenMaxY; pixelY++) {
    float4 *imgPtr =
        (float4 *)(&cuConstRendererParams
                        .imageData[4 * (pixelY * imageWidth + screenMinX)]);
    for (int pixelX = screenMinX; pixelX < screenMaxX; pixelX++) {
      float2 pixelCenterNorm =
          make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                      invHeight * (static_cast<float>(pixelY) + 0.5f));
      shadePixel(index, pixelCenterNorm, p, imgPtr);
      imgPtr++;
    }
  }
}

namespace Solution1 {
__global__ void kernelRenderPixelsWithCircles(int MinX, int MaxX, int MinY,
                                              int MaxY, int numCircles,
                                              int *Circles) {
  int pixelX = MinX + blockIdx.x * blockDim.x + threadIdx.x;
  int pixelY = MinY + blockIdx.y * blockDim.y + threadIdx.y;

  if (pixelX >= MaxX || pixelY >= MaxY)
    return;

  float2 pixelCenterNorm =
      make_float2((pixelX + 0.5) / cuConstRendererParams.imageWidth,
                  (pixelY + 0.5) / cuConstRendererParams.imageHeight);

  float4 *imgPtr = reinterpret_cast<float4 *>(
      &cuConstRendererParams
           .imageData[4 *
                      (pixelY * cuConstRendererParams.imageWidth + pixelX)]);

  for (int i = 0; i < numCircles; i++) {
    int index3 = 3 * Circles[i];
    shadePixel(
        Circles[i], pixelCenterNorm,
        *reinterpret_cast<float3 *>(&cuConstRendererParams.position[index3]),
        imgPtr);
  }
}

void renderPixelsWithCircles(int MinX, int MaxX, int MinY, int MaxY,
                             int numCircles, int *Circles) {
  // printf("renderPixelsWithCircles\n");
  // printf("MinX: %d, MaxX: %d, MinY: %d, MaxY: %d\n", MinX, MaxX, MinY, MaxY);
  // printf("numCircles: %d\n", numCircles);
  // for (int i = 0; i < numCircles; i++) {
  //   printf("Circles[%d]: %d\n", i, Circles[i]);
  // }
  if (numCircles == 0) {
    return;
  }

  static constexpr int THREADS_PER_BLOCK = 256;
  int blockDimX = std::min(THREADS_PER_BLOCK, MaxX - MinX);
  int blockDimY = std::min(THREADS_PER_BLOCK / blockDimX, MaxY - MinY);
  dim3 blockDim(blockDimX, blockDimY);

  int gridDimX = (MaxX - MinX + blockDim.x - 1) / blockDim.x;
  int gridDimY = (MaxY - MinY + blockDim.y - 1) / blockDim.y;
  dim3 gridDim(gridDimX, gridDimY);

  // printf("blockDim: %d, %d\n", blockDim.x, blockDim.y);
  // printf("gridDim: %d, %d\n", gridDim.x, gridDim.y);

  kernelRenderPixelsWithCircles<<<gridDim, blockDim>>>(MinX, MaxX, MinY, MaxY,
                                                       numCircles, Circles);
}

void renderPixelsWithAllCircles(int numCircles, int width, int height) {
  int *Circles;
  cudaMallocManaged(&Circles, numCircles * sizeof(int));
  for (int i = 0; i < numCircles; i++) {
    Circles[i] = i;
  }

  renderPixelsWithCircles(0, width, 0, height, numCircles, Circles);
  cudaDeviceSynchronize();

  cudaFree(Circles);
}

constexpr int BLOCK_SIZE = 16;

__global__ void kernelGetCirclesInBlock(int *numBlockCircles, int *blockCircles,
                                        int width, int height, int numBlocksX) {
  int threadX = blockIdx.x * blockDim.x + threadIdx.x;
  int threadY = blockIdx.y * blockDim.y + threadIdx.y;
  int blockMinX = threadX * BLOCK_SIZE;
  int blockMinY = threadY * BLOCK_SIZE;

  if (blockMinX >= width || blockMinY >= height)
    return;

  int threadID = threadY * numBlocksX + threadX;
  int blockMaxX = min(blockMinX + BLOCK_SIZE, width);
  int blockMaxY = min(blockMinY + BLOCK_SIZE, height);

  float invWidth = 1.f / width;
  float invHeight = 1.f / height;
  float boxL = blockMinX * invWidth;
  float boxR = blockMaxX * invWidth;
  float boxT = blockMinY * invHeight;
  float boxB = blockMaxY * invHeight;

  numBlockCircles[threadID] = 0;
  int *ptr = blockCircles + threadID * cuConstRendererParams.numCircles;
  for (int i = 0; i < cuConstRendererParams.numCircles; i++) {
    int index3 = 3 * i;
    if (circleInBox(cuConstRendererParams.position[index3],
                    cuConstRendererParams.position[index3 + 1],
                    cuConstRendererParams.radius[i], boxL, boxR, boxT, boxB)) {
      ptr[numBlockCircles[threadID]++] = i;
    }
  }

  // printf("Thread: %d, %d, %d numBlockCircles: %d\n", threadX, threadY,
  // threadID,
  //        numBlockCircles[threadID]);
}

void renderBlockedPixelsWithAllCircles(int numCircles, int width, int height) {
  // printf("renderBlockedPixelsWithAllCircles\n");
  // printf("numCircles: %d, width: %d, height: %d\n", numCircles, width,
  // height);

  int numBlocksX = (width + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int numBlocksY = (height + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int numBlocks = numBlocksX * numBlocksY;

  // printf("numBlocksX: %d, numBlocksY: %d\n", numBlocksX, numBlocksY);
  // printf("numBlocks: %d\n", numBlocks);

  int *numBlockCircles;
  cudaCheckError(cudaMallocManaged(&numBlockCircles, numBlocks * sizeof(int)));
  int *blockCircles;
  cudaCheckError(
      cudaMallocManaged(&blockCircles, numBlocks * numCircles * sizeof(int)));

  dim3 gridDim((numBlocksX + BLOCK_SIZE - 1) / BLOCK_SIZE,
               (numBlocksY + BLOCK_SIZE - 1) / BLOCK_SIZE);
  dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

  // printf("gridDim: %d, %d\n", gridDim.x, gridDim.y);
  // printf("blockDim: %d, %d\n", blockDim.x, blockDim.y);

  kernelGetCirclesInBlock<<<gridDim, blockDim>>>(numBlockCircles, blockCircles,
                                                 width, height, numBlocksX);
  cudaDeviceSynchronize();
  // printf("kernelGetCirclesInBlock finished\n");

  for (int i = 0; i < numBlocksX; i++) {
    for (int j = 0; j < numBlocksY; j++) {
      int blockIndex = j * numBlocksX + i;
      int blockMinX = i * BLOCK_SIZE;
      int blockMinY = j * BLOCK_SIZE;
      int blockMaxX = std::min(blockMinX + BLOCK_SIZE, width);
      int blockMaxY = std::min(blockMinY + BLOCK_SIZE, height);
      // printf("blockIndex: %d, blockMinX: %d, blockMaxX: %d, blockMinY: %d, "
      //        "blockMaxY: %d\n",
      //        blockIndex, blockMinX, blockMaxX, blockMinY, blockMaxY);
      // printf("numBlockCircles: %d\n", numBlockCircles[blockIndex]);
      renderPixelsWithCircles(blockMinX, blockMaxX, blockMinY, blockMaxY,
                              numBlockCircles[blockIndex],
                              blockCircles + blockIndex * numCircles);
      cudaDeviceSynchronize();
    }
  }
  cudaDeviceSynchronize();

  cudaFree(blockCircles);
  cudaFree(numBlockCircles);
}
} // namespace Solution1

#include <thrust/scan.h>
#include <thrust/sort.h>

namespace Solution2 {

constexpr int THREADS_PER_BLOCK = 256;
constexpr int BLOCK_SIZE = 16;

__inline__ __device__ int clamp(int x, int low, int high) {
  return x > low ? (x < high ? x : high) : low;
}

template <typename T> T readKernel(const T *ptr) {
  T value;
  cudaMemcpy(&value, ptr, sizeof(T), cudaMemcpyDeviceToHost);
  return value;
}

__inline__ __device__ bool pixelInCircle(float2 pixelCenter, float3 p,
                                         float rad) {
  float dx = p.x - pixelCenter.x;
  float dy = p.y - pixelCenter.y;
  return dx * dx + dy * dy <= rad * rad;
}

// for each circle, calculate the bounding box of the circle
__global__ void kernelGetCirclesSize(int *circlesSizePrefixSum, short *minX,
                                     short *maxX, short *minY, short *maxY) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= cuConstRendererParams.numCircles)
    return;

  int index3 = 3 * index;
  float3 p =
      *reinterpret_cast<float3 *>(&cuConstRendererParams.position[index3]);
  float rad = cuConstRendererParams.radius[index];

  short imageWidth = cuConstRendererParams.imageWidth;
  short imageHeight = cuConstRendererParams.imageHeight;

  minX[index] = clamp(imageWidth * (p.x - rad), 0, imageWidth);
  maxX[index] = clamp(imageWidth * (p.x + rad) + 1, 0, imageWidth);
  minY[index] = clamp(imageHeight * (p.y - rad), 0, imageHeight);
  maxY[index] = clamp(imageHeight * (p.y + rad) + 1, 0, imageHeight);

  circlesSizePrefixSum[index] =
      (maxX[index] - minX[index]) * (maxY[index] - minY[index]);
}

// for each circle, calculate the pixels covered by the circle
__global__ void kernelGetCirclesPixels(int *pixelsId, int *circlesId,
                                       int *circlesSizePrefixSum, short minX,
                                       short maxX, short minY, short maxY,
                                       int index) {
  int pixelX = minX + blockIdx.x * blockDim.x + threadIdx.x;
  int pixelY = minY + blockIdx.y * blockDim.y + threadIdx.y;

  if (pixelX >= maxX || pixelY >= maxY)
    return;

  int globalOffset = index == 0 ? 0 : circlesSizePrefixSum[index - 1];

  int offset = (pixelY - minY) * (maxX - minX) + (pixelX - minX);
  int pixelId = pixelY * cuConstRendererParams.imageWidth + pixelX;

  if (pixelInCircle(
          make_float2((pixelX + 0.5) / cuConstRendererParams.imageWidth,
                      (pixelY + 0.5) / cuConstRendererParams.imageHeight),
          *reinterpret_cast<float3 *>(
              &cuConstRendererParams.position[3 * index]),
          cuConstRendererParams.radius[index])) {
    pixelsId[globalOffset + offset] = pixelId;
    circlesId[globalOffset + offset] = index;
  } else {
    pixelsId[globalOffset + offset] = -1;
    circlesId[globalOffset + offset] = -1;
  }
}

// __device__ void kernelBlendPixel() {}

__global__ void kernelRenderPixels(int *pixelsId, int *circlesId,
                                   int totalCirclesSize) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= totalCirclesSize)
    return;

  if (pixelsId[index] == -1 ||
      (index > 0 && pixelsId[index] == pixelsId[index - 1]))
    return;

  while (true) {
    int pixelX = pixelsId[index] % cuConstRendererParams.imageWidth;
    int pixelY = pixelsId[index] / cuConstRendererParams.imageWidth;

    shadePixel(circlesId[index],
               make_float2((pixelX + 0.5) / cuConstRendererParams.imageWidth,
                           (pixelY + 0.5) / cuConstRendererParams.imageHeight),
               *reinterpret_cast<float3 *>(
                   &cuConstRendererParams.position[3 * circlesId[index]]),
               reinterpret_cast<float4 *>(
                   &cuConstRendererParams.imageData[4 * pixelsId[index]]));

    if (index + 1 < totalCirclesSize &&
        pixelsId[index] == pixelsId[index + 1]) {
      ++index;
    } else {
      break;
    }
  }
}

void renderCircles(int numCircles) {
  int *circlesSizePrefixSum;
  short *minX, *maxX, *minY, *maxY;
  cudaCheckError(cudaMalloc(&circlesSizePrefixSum, numCircles * sizeof(int)));
  cudaCheckError(cudaMalloc(&minX, numCircles * sizeof(short)));
  cudaCheckError(cudaMalloc(&maxX, numCircles * sizeof(short)));
  cudaCheckError(cudaMalloc(&minY, numCircles * sizeof(short)));
  cudaCheckError(cudaMalloc(&maxY, numCircles * sizeof(short)));

  kernelGetCirclesSize<<<(numCircles + THREADS_PER_BLOCK - 1) /
                             THREADS_PER_BLOCK,
                         THREADS_PER_BLOCK>>>(circlesSizePrefixSum, minX, maxX,
                                              minY, maxY);
  cudaDeviceSynchronize();

  thrust::inclusive_scan(thrust::device, circlesSizePrefixSum,
                         circlesSizePrefixSum + numCircles,
                         circlesSizePrefixSum);

  // pixel covered by circles
  int *pixelsId, *circlesId;
  int totalCirclesSize = readKernel(&circlesSizePrefixSum[numCircles - 1]);
  cudaCheckError(cudaMalloc(&pixelsId, totalCirclesSize * sizeof(int)));
  cudaCheckError(cudaMalloc(&circlesId, totalCirclesSize * sizeof(int)));

  short *minXHost = new short[numCircles];
  short *maxXHost = new short[numCircles];
  short *minYHost = new short[numCircles];
  short *maxYHost = new short[numCircles];

  cudaCheckError(cudaMemcpy(minXHost, minX, numCircles * sizeof(short),
                            cudaMemcpyDeviceToHost));
  cudaCheckError(cudaMemcpy(maxXHost, maxX, numCircles * sizeof(short),
                            cudaMemcpyDeviceToHost));
  cudaCheckError(cudaMemcpy(minYHost, minY, numCircles * sizeof(short),
                            cudaMemcpyDeviceToHost));
  cudaCheckError(cudaMemcpy(maxYHost, maxY, numCircles * sizeof(short),
                            cudaMemcpyDeviceToHost));

  for (int i = 0; i < numCircles; i++) {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((maxXHost[i] - minXHost[i] + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (maxYHost[i] - minYHost[i] + BLOCK_SIZE - 1) / BLOCK_SIZE);
    kernelGetCirclesPixels<<<gridDim, blockDim>>>(
        pixelsId, circlesId, circlesSizePrefixSum, minXHost[i], maxXHost[i],
        minYHost[i], maxYHost[i], i);
  }
  cudaDeviceSynchronize();

  thrust::stable_sort_by_key(thrust::device, pixelsId,
                             pixelsId + totalCirclesSize, circlesId);

  kernelRenderPixels<<<(totalCirclesSize + THREADS_PER_BLOCK - 1) /
                           THREADS_PER_BLOCK,
                       THREADS_PER_BLOCK>>>(pixelsId, circlesId,
                                            totalCirclesSize);
  cudaDeviceSynchronize();

  delete[] minXHost;
  delete[] maxXHost;
  delete[] minYHost;
  delete[] maxYHost;

  cudaFree(circlesSizePrefixSum);
  cudaFree(minX);
  cudaFree(maxX);
  cudaFree(minY);
  cudaFree(maxY);
  cudaFree(pixelsId);
  cudaFree(circlesId);
}
} // namespace Solution2

namespace Solution3 {

constexpr int BLOCK_DIM = 16;
constexpr int BLOCK_SIZE = BLOCK_DIM * BLOCK_DIM;

#define SCAN_BLOCK_DIM BLOCK_SIZE
#include "exclusiveScan.cu_inl"

__global__ void kernelRenderCircles() {
  __shared__ uint circleIsInBox[BLOCK_SIZE];
  __shared__ uint circleIndex[BLOCK_SIZE];
  __shared__ uint scratch[2 * BLOCK_SIZE];
  __shared__ int inBoxCircles[BLOCK_SIZE];

  int boxL = blockIdx.x * BLOCK_DIM;
  int boxB = blockIdx.y * BLOCK_DIM;
  int boxR = min(boxL + BLOCK_DIM, cuConstRendererParams.imageWidth);
  int boxT = min(boxB + BLOCK_DIM, cuConstRendererParams.imageHeight);
  float boxLNorm = boxL * cuConstRendererParams.invWidth;
  float boxRNorm = boxR * cuConstRendererParams.invWidth;
  float boxTNorm = boxT * cuConstRendererParams.invHeight;
  float boxBNorm = boxB * cuConstRendererParams.invHeight;

  int index = threadIdx.y * BLOCK_DIM + threadIdx.x;
  int pixelX = boxL + threadIdx.x;
  int pixelY = boxB + threadIdx.y;
  int pixelId = pixelY * cuConstRendererParams.imageWidth + pixelX;

  for (int i = 0; i < cuConstRendererParams.numCircles; i += BLOCK_SIZE) {
    int circleId = i + index;
    if (circleId < cuConstRendererParams.numCircles) {
      float3 p = *reinterpret_cast<float3 *>(
          &cuConstRendererParams.position[3 * circleId]);
      circleIsInBox[index] =
          circleInBox(p.x, p.y, cuConstRendererParams.radius[circleId],
                      boxLNorm, boxRNorm, boxTNorm, boxBNorm);
    } else {
      circleIsInBox[index] = 0;
    }
    __syncthreads();

    sharedMemExclusiveScan(index, circleIsInBox, circleIndex, scratch,
                           BLOCK_SIZE);
    if (circleIsInBox[index]) {
      inBoxCircles[circleIndex[index]] = circleId;
    }
    __syncthreads();

    int numCirclesInBox =
        circleIndex[BLOCK_SIZE - 1] + circleIsInBox[BLOCK_SIZE - 1];
    __syncthreads();

    if (pixelX < boxR && pixelY < boxT) {
      float4 *imgPtr = reinterpret_cast<float4 *>(
          &cuConstRendererParams.imageData[4 * pixelId]);
      for (int j = 0; j < numCirclesInBox; j++) {
        circleId = inBoxCircles[j];
        shadePixel(
            circleId,
            make_float2((pixelX + 0.5) * cuConstRendererParams.invWidth,
                        (pixelY + 0.5) * cuConstRendererParams.invHeight),
            *reinterpret_cast<float3 *>(
                &cuConstRendererParams.position[3 * circleId]),
            imgPtr);
      }
    }
  }
}

void renderCircles(int width, int height) {
  kernelRenderCircles<<<dim3((width + BLOCK_DIM - 1) / BLOCK_DIM,
                             (height + BLOCK_DIM - 1) / BLOCK_DIM),
                        dim3(BLOCK_DIM, BLOCK_DIM)>>>();
  cudaCheckError(cudaDeviceSynchronize());
}
} // namespace Solution3

////////////////////////////////////////////////////////////////////////////////////////

CudaRenderer::CudaRenderer() {
  image = NULL;

  numCircles = 0;
  position = NULL;
  velocity = NULL;
  color = NULL;
  radius = NULL;

  cudaDevicePosition = NULL;
  cudaDeviceVelocity = NULL;
  cudaDeviceColor = NULL;
  cudaDeviceRadius = NULL;
  cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

  if (image) {
    delete image;
  }

  if (position) {
    delete[] position;
    delete[] velocity;
    delete[] color;
    delete[] radius;
  }

  if (cudaDevicePosition) {
    cudaFree(cudaDevicePosition);
    cudaFree(cudaDeviceVelocity);
    cudaFree(cudaDeviceColor);
    cudaFree(cudaDeviceRadius);
    cudaFree(cudaDeviceImageData);
  }
}

const Image *CudaRenderer::getImage() {

  // need to copy contents of the rendered image from device memory
  // before we expose the Image object to the caller

  printf("Copying image data from device\n");

  cudaMemcpy(image->data, cudaDeviceImageData,
             sizeof(float) * 4 * image->width * image->height,
             cudaMemcpyDeviceToHost);

  return image;
}

void CudaRenderer::loadScene(SceneName scene) {
  sceneName = scene;
  loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void CudaRenderer::setup() {

  int deviceCount = 0;
  std::string name;
  cudaError_t err = cudaGetDeviceCount(&deviceCount);

  printf("---------------------------------------------------------\n");
  printf("Initializing CUDA for CudaRenderer\n");
  printf("Found %d CUDA devices\n", deviceCount);

  for (int i = 0; i < deviceCount; i++) {
    cudaDeviceProp deviceProps;
    cudaGetDeviceProperties(&deviceProps, i);
    name = deviceProps.name;

    printf("Device %d: %s\n", i, deviceProps.name);
    printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
    printf("   Global mem: %.0f MB\n",
           static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
    printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
  }
  printf("---------------------------------------------------------\n");

  // By this time the scene should be loaded.  Now copy all the key
  // data structures into device memory so they are accessible to
  // CUDA kernels
  //
  // See the CUDA Programmer's Guide for descriptions of
  // cudaMalloc and cudaMemcpy

  cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
  cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
  cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
  cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
  cudaMalloc(&cudaDeviceImageData,
             sizeof(float) * 4 * image->width * image->height);

  cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles,
             cudaMemcpyHostToDevice);
  cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles,
             cudaMemcpyHostToDevice);
  cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles,
             cudaMemcpyHostToDevice);
  cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles,
             cudaMemcpyHostToDevice);

  // Initialize parameters in constant memory.  We didn't talk about
  // constant memory in class, but the use of read-only constant
  // memory here is an optimization over just sticking these values
  // in device global memory.  NVIDIA GPUs have a few special tricks
  // for optimizing access to constant memory.  Using global memory
  // here would have worked just as well.  See the Programmer's
  // Guide for more information about constant memory.

  GlobalConstants params;
  params.sceneName = sceneName;
  params.numCircles = numCircles;
  params.imageWidth = image->width;
  params.imageHeight = image->height;
  params.position = cudaDevicePosition;
  params.velocity = cudaDeviceVelocity;
  params.color = cudaDeviceColor;
  params.radius = cudaDeviceRadius;
  params.imageData = cudaDeviceImageData;
  params.invWidth = 1.f / image->width;
  params.invHeight = 1.f / image->height;

  cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

  // also need to copy over the noise lookup tables, so we can
  // implement noise on the GPU
  int *permX;
  int *permY;
  float *value1D;
  getNoiseTables(&permX, &permY, &value1D);
  cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
  cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
  cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

  // last, copy over the color table that's used by the shading
  // function for circles in the snowflake demo

  float lookupTable[COLOR_MAP_SIZE][3] = {
      {1.f, 1.f, 1.f}, {1.f, 1.f, 1.f},  {.8f, .9f, 1.f},
      {.8f, .9f, 1.f}, {.8f, 0.8f, 1.f},
  };

  cudaMemcpyToSymbol(cuConstColorRamp, lookupTable,
                     sizeof(float) * 3 * COLOR_MAP_SIZE);
}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void CudaRenderer::allocOutputImage(int width, int height) {

  if (image)
    delete image;
  image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void CudaRenderer::clearImage() {

  // 256 threads per block is a healthy number
  dim3 blockDim(16, 16, 1);
  dim3 gridDim((image->width + blockDim.x - 1) / blockDim.x,
               (image->height + blockDim.y - 1) / blockDim.y);

  if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
    kernelClearImageSnowflake<<<gridDim, blockDim>>>();
  } else {
    kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
  }
  cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void CudaRenderer::advanceAnimation() {
  // 256 threads per block is a healthy number
  dim3 blockDim(256, 1);
  dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

  // only the snowflake scene has animation
  if (sceneName == SNOWFLAKES) {
    kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
  } else if (sceneName == BOUNCING_BALLS) {
    kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
  } else if (sceneName == HYPNOSIS) {
    kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
  } else if (sceneName == FIREWORKS) {
    kernelAdvanceFireWorks<<<gridDim, blockDim>>>();
  }
  cudaDeviceSynchronize();
}

void CudaRenderer::render() {
  // 256 threads per block is a healthy number
  // dim3 blockDim(256, 1);
  // dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);
  // kernelRenderCircles<<<gridDim, blockDim>>>();
  // cudaDeviceSynchronize();

  Solution3::renderCircles(image->width, image->height);
}
