#include <algorithm>
#include <immintrin.h>
#include <math.h>
#include <pthread.h>
#include <stdio.h>

#include "CycleTimer.h"
#include "sqrt_ispc.h"

using namespace ispc;

extern void sqrtSerial(int N, float startGuess, float *values, float *output);

void sqrtAvx2(int N, float initialGuess, float values[], float output[]) {
  static const __m256 kThreshold = _mm256_set1_ps(0.00001f);
  for (int i = 0; i < N; i += 8) {
    if (i + 8 > N) {
      sqrtSerial(N - i, initialGuess, values + i, output + i);
      break;
    }
    __m256 x = _mm256_loadu_ps(values + i);
    __m256 guess = _mm256_set1_ps(initialGuess);
    while (true) {
      __m256 error = _mm256_sub_ps(
          _mm256_mul_ps(_mm256_mul_ps(guess, guess), x), _mm256_set1_ps(1.f));

      // error = fabs(error)
      error = _mm256_andnot_ps(_mm256_set1_ps(-0.f), error);

      __m256 cmp = _mm256_cmp_ps(error, kThreshold, _CMP_GT_OQ);
      if (_mm256_testz_ps(cmp, cmp)) {
        break;
      }

      // guess = (3.f * guess - x * guess * guess * guess) * 0.5f;
      // guess = (3.f - x * guess * guess) * guess * 0.5f;
      __m256 newGuess = _mm256_mul_ps(
          _mm256_mul_ps(
              _mm256_sub_ps(_mm256_set1_ps(3.f),
                            _mm256_mul_ps(x, _mm256_mul_ps(guess, guess))),
              guess),
          _mm256_set1_ps(0.5f));

      guess = _mm256_blendv_ps(guess, newGuess, cmp);
    }
    _mm256_storeu_ps(output + i, _mm256_mul_ps(x, guess));
  }
}

static void verifyResult(int N, float *result, float *gold) {
  for (int i = 0; i < N; i++) {
    if (fabs(result[i] - gold[i]) > 1e-4) {
      printf("Error: [%d] Got %f expected %f\n", i, result[i], gold[i]);
    }
  }
}

int main() {

  const unsigned int N = 20 * 1000 * 1000;
  const float initialGuess = 1.0f;

  float *values = new float[N];
  float *output = new float[N];
  float *gold = new float[N];

  for (unsigned int i = 0; i < N; i++) {
    // starter code populates array with random input values
    values[i] = .001f + 2.998f * static_cast<float>(rand()) / RAND_MAX;

    // best cast
    // values[i] = 2.5f;

    // worst case
    // values[i] = (i & 7) ? 1.0f : 2.999f;
  }

  // generate a gold version to check results
  for (unsigned int i = 0; i < N; i++)
    gold[i] = sqrt(values[i]);

  //
  // And run the serial implementation 3 times, again reporting the
  // minimum time.
  //
  double minSerial = 1e30;
  for (int i = 0; i < 3; ++i) {
    double startTime = CycleTimer::currentSeconds();
    sqrtSerial(N, initialGuess, values, output);
    double endTime = CycleTimer::currentSeconds();
    minSerial = std::min(minSerial, endTime - startTime);
  }

  printf("[sqrt serial]:\t\t[%.3f] ms\n", minSerial * 1000);

  verifyResult(N, output, gold);

  //
  // Compute the image using the ispc implementation; report the minimum
  // time of three runs.
  //
  double minISPC = 1e30;
  for (int i = 0; i < 3; ++i) {
    double startTime = CycleTimer::currentSeconds();
    sqrt_ispc(N, initialGuess, values, output);
    double endTime = CycleTimer::currentSeconds();
    minISPC = std::min(minISPC, endTime - startTime);
  }

  printf("[sqrt ispc]:\t\t[%.3f] ms\n", minISPC * 1000);

  verifyResult(N, output, gold);

  // Clear out the buffer
  for (unsigned int i = 0; i < N; ++i)
    output[i] = 0;

  //
  // Tasking version of the ISPC code
  //
  double minTaskISPC = 1e30;
  for (int i = 0; i < 3; ++i) {
    double startTime = CycleTimer::currentSeconds();
    sqrt_ispc_withtasks(N, initialGuess, values, output);
    double endTime = CycleTimer::currentSeconds();
    minTaskISPC = std::min(minTaskISPC, endTime - startTime);
  }

  printf("[sqrt task ispc]:\t[%.3f] ms\n", minTaskISPC * 1000);

  verifyResult(N, output, gold);

  // Clear out the buffer
  for (unsigned int i = 0; i < N; ++i)
    output[i] = 0;

  //
  // Avx2 version of the ISPC code
  //
  double minAvx2 = 1e30;
  for (int i = 0; i < 3; ++i) {
    double startTime = CycleTimer::currentSeconds();
    sqrtAvx2(N, initialGuess, values, output);
    double endTime = CycleTimer::currentSeconds();
    minAvx2 = std::min(minAvx2, endTime - startTime);
  }

  printf("[sqrt avx2]:\t\t[%.3f] ms\n", minAvx2 * 1000);

  verifyResult(N, output, gold);

  printf("\t\t\t\t(%.2fx speedup from ISPC)\n", minSerial / minISPC);
  printf("\t\t\t\t(%.2fx speedup from task ISPC)\n", minSerial / minTaskISPC);
  printf("\t\t\t\t(%.2fx speedup from avx2)\n", minSerial / minAvx2);

  delete[] values;
  delete[] output;
  delete[] gold;

  return 0;
}
