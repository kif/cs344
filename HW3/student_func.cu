/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Definition Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/
#include <stdio.h>
#define REDUCE(a, b) (make_float2(fmaxf(a.x,b.x),fminf(a.y,b.y)))
#define READ_AND_MAP(i) (make_float2(data[i],data[i]))
#define WORKGROUP_SIZE 1024
/**
 * \brief max_min_global_stage1: Look for the maximum an the minimum of an array. stage1
 *
 * optimal workgroup size: 2^n greater than sqrt(SIZE), limited to 512
 * optimal total item size:  (workgroup size)^2
 * if SIZE >total item size: adjust seq_count.
 *
 * @param data:       Float pointer to global memory storing the vector of data.
 * @param out:        Float2 pointer to global memory storing the temporary results (workgroup size)
 * @param seq_count:  how many blocksize each thread should read
 * @param SIZE:     size of the
 *
**/

__global__
void max_min_stage1(
    const float *data,
    float2 *out,
    unsigned int SIZE){

    __shared__ float2 ldata[WORKGROUP_SIZE];
    unsigned int group_size =  min((unsigned int) blockDim.x, (unsigned int) WORKGROUP_SIZE);
    unsigned int lid = threadIdx.x;
    float2 acc;
    unsigned int big_block = group_size * gridDim.x;
    unsigned int i =  lid + group_size * blockIdx.x;

    if (lid<SIZE)
      acc = READ_AND_MAP(lid);
    else
      acc = READ_AND_MAP(0);
    while (i<SIZE){
      acc = REDUCE(acc, READ_AND_MAP(i));
      i += big_block; //get_global_size(0);
    }
    ldata[lid] = acc;
    __syncthreads();

    if ((lid<group_size) && (lid < 512) && ((lid + 512)<group_size)){
      ldata[lid] = REDUCE(ldata[lid], ldata[lid + 512]);
    }
    __syncthreads();
    if ((lid<group_size) && (lid < 256) && ((lid + 256)<group_size)){
      ldata[lid] = REDUCE(ldata[lid], ldata[lid + 256]);
    }
    __syncthreads();
    if ((lid<group_size) && (lid < 128) && ((lid + 128)<group_size)){
      ldata[lid] = REDUCE(ldata[lid], ldata[lid + 128]);
    }
    __syncthreads();
    if ((lid<group_size) && (lid < 64 ) && ((lid + 64 )<group_size)){
      ldata[lid] = REDUCE(ldata[lid], ldata[lid + 64 ]);
    }
    __syncthreads();
    if ((lid<group_size) && (lid < 32 ) && ((lid + 32 )<group_size)){
      ldata[lid] = REDUCE(ldata[lid], ldata[lid + 32 ]);
    }
    __syncthreads();
    if ((lid<group_size) && (lid < 16 ) && ((lid + 16 )<group_size)){
      ldata[lid] = REDUCE(ldata[lid], ldata[lid + 16 ]);
    }
    __syncthreads();
    if ((lid<group_size) && (lid < 8  ) && ((lid + 8  )<group_size)){
      ldata[lid] = REDUCE(ldata[lid], ldata[lid + 8  ]);
    }
    __syncthreads();

    if ((lid<group_size) && (lid < 4  ) && ((lid + 4  )<group_size)){
      ldata[lid] = REDUCE(ldata[lid], ldata[lid + 4  ]);
    }
    __syncthreads();
    if ((lid<group_size) && (lid < 2  ) && ((lid + 2  )<group_size)){
      ldata[lid] = REDUCE(ldata[lid], ldata[lid + 2  ]);
    }
    __syncthreads();
    if ((lid<group_size) && (lid < 1  ) && ((lid + 1  )<group_size)){
      ldata[lid] = REDUCE(ldata[lid], ldata[lid + 1  ]);
    }
    __syncthreads();

    out[blockIdx.x] = ldata[0];
}


/**
 * \brief global_max_min: Look for the maximum an the minimum of an array.
 *
 *
 *
 * @param data2:      Float2 pointer to global memory storing the vector of pre-reduced data (workgroup size).
 * @param maximum:    Float pointer to global memory storing the maximum value
 * @param minumum:    Float pointer to global memory storing the minimum value
 *
**/

__global__
void max_min_stage2(
    const float2 *data2,
    float *maximum,
    float *minimum){

  __shared__ float2 ldata[WORKGROUP_SIZE];
  unsigned int lid = threadIdx.x;
  unsigned int group_size =  min((unsigned int) blockDim.x, (unsigned int) WORKGROUP_SIZE);
  float2 acc; //= make_float2(-1.0f, -1.0f);
  if (lid<=group_size){
    ldata[lid] = data2[lid];
  };//else{
   // ldata[lid] = acc;
  //}
  __syncthreads();

  if ((lid<group_size) && (lid < 512) && ((lid + 512)<group_size)){
    ldata[lid] = REDUCE(ldata[lid], ldata[lid + 512]);
  }
  __syncthreads();
  if ((lid<group_size) && (lid < 256) && ((lid + 256)<group_size)){
    ldata[lid] = REDUCE(ldata[lid], ldata[lid + 256]);
  }
  __syncthreads();
  if ((lid<group_size) && (lid < 128) && ((lid + 128)<group_size)){
    ldata[lid] = REDUCE(ldata[lid], ldata[lid + 128]);
  }
  __syncthreads();
  if ((lid<group_size) && (lid < 64 ) && ((lid + 64 )<group_size)){
    ldata[lid] = REDUCE(ldata[lid], ldata[lid + 64 ]);
  }
  __syncthreads();
  if ((lid<group_size) && (lid < 32 ) && ((lid + 32 )<group_size)){
    ldata[lid] = REDUCE(ldata[lid], ldata[lid + 32 ]);
  }
  __syncthreads();
  if ((lid<group_size) && (lid < 16 ) && ((lid + 16 )<group_size)){
    ldata[lid] = REDUCE(ldata[lid], ldata[lid + 16 ]);
  }
  __syncthreads();
  if ((lid<group_size) && (lid < 8  ) && ((lid + 8  )<group_size)){
    ldata[lid] = REDUCE(ldata[lid], ldata[lid + 8  ]);
  }
  __syncthreads();
  if ((lid<group_size) && (lid < 4  ) && ((lid + 4  )<group_size)){
    ldata[lid] = REDUCE(ldata[lid], ldata[lid + 4  ]);
  }
  __syncthreads();
  if ((lid<group_size) && (lid < 2  ) && ((lid + 2  )<group_size)){
    ldata[lid] = REDUCE(ldata[lid], ldata[lid + 2  ]);
  }
  __syncthreads();

  if (lid == 0  ){
    if ( 1 < group_size){
      acc = REDUCE(ldata[0], ldata[1]);
    }else{
      acc = ldata[0];
    }
    maximum[0] = acc.x;
    minimum[0] = acc.y;
  }
}

/**
 * \brief histogram: calculate the histogram of an image
 *        bin = (lum[i] - lumMin) / lumRange * numBins
 *
 *
 * @param data:       Float pointer to global memory storing the log of the luminance.
 * @param maximum:    Float pointer to global memory storing the maximum value
 * @param minumum:    Float pointer to global memory storing the minimum value
 *
**/

__global__
void histogram(const float* const lum,
                    float* lumMin,
                    float* lumMax,
                    unsigned int* bins,
                    int numBins,
                    unsigned int SIZE)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx>=SIZE)
    return;
  int bin = (lum[idx] - lumMin[0]) / (lumMax[0] - lumMin[0]) * numBins;
  atomicAdd(&bins[bin],1);
}
/**
 * \brief blelloch1: Exclusive scan phase 1: reduction like perent mode,
 *
 *
 *
 * @param data:       Integer pointer to global memory storing the data: modified in place.
 * @param d:          Integer: scale at which we are working
 * @param SIZE:       integer representing the size of the array
 *
**/

__global__
void blelloch1( unsigned int* data,
                int d,
                int SIZE)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx>=SIZE)
    return;
//  __shared__ float s_data[WORKGROUP_SIZE];
  int scale = pow(2,d+1);
  if ((idx>scale/2) && ((SIZE-idx) mod scale ==0 )){
    data[idx] := data[idx] + data[idx - scale/2];
  }
}
__global__
void blelloch2( unsigned int* data,
                int d,
                int SIZE)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx>=SIZE)
    return;
//  __shared__ float s_data[WORKGROUP_SIZE];
  int scale = pow(2,d+1);
  if ((idx>scale/2) && ((SIZE-idx) mod scale ==0 )){
    data[idx] := data[idx] + data[idx - scale/2];
  }
}

#include "utils.h"

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */
  //Malloc stuff:
  float2 *d_data2;
  float *d_min, *d_max;
  int *d_bins;
  checkCudaErrors(cudaMalloc(&d_min, (size_t)  sizeof(float)));
  checkCudaErrors(cudaMalloc(&d_max, (size_t)  sizeof(float)));
//  checkCudaErrors(cudaMalloc(&d_bins, (size_t)  sizeof(int)*numBins));
  //1. get maximum and minimum of logLuminance channel.
  int image_size = numRows*numCols;
  float wg_float = fminf((float) WORKGROUP_SIZE, sqrtf((float)image_size));
  int red_size = pow(2, (int)ceil(logf(wg_float)/logf(2.0f)));
  int memory = sizeof(float) * 2 * red_size; //temporary storage for reduction

  checkCudaErrors(cudaMalloc(&d_data2, (size_t)memory));


  max_min_stage1<<<red_size, red_size>>>(d_logLuminance, d_data2, image_size);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  max_min_stage2<<<1, red_size>>>(d_data2, d_max, d_min );
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  float mmin[1];
  float mmax[1];
  cudaMemcpy(mmin, d_min, sizeof(float) * 1, cudaMemcpyDeviceToHost);
  cudaMemcpy(mmax, d_max, sizeof(float) * 1, cudaMemcpyDeviceToHost);
  printf( "CUDA Min: %f Max: %f\n",mmin[0],mmax[0]);
//  float logLumRange = mmax[0] - mmin[0];

//  3) generate a histogram of all the values in the logLuminance channel using
//         the formula: bin = (lum[i] - lumMin) / lumRange * numBins
  histogram<<<((image_size+31)/32),32>>>(d_logLuminance, d_min, d_max, d_cdf, numBins, image_size);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
//  4) Perform an exclusive scan (prefix sum) on the histogram to get
//     the cumulative distribution of luminance values (this should go in the
//     incoming d_cdf pointer which already has been allocated for you)       */
  int dmax = (int) ceil(log(1.0*numBins)/log(2.0))
  for (int d= 0, d<dmax-1, d++){
    blelloch1<<<((numBins+31)/32),32>>>(d_cdf, d, numBins);
  }

  cudaMemcpy(d_cdf+sizeof(unsigned int)*(numBins-1), d_cdf+sizeof(unsigned int)*(numBins/2-1), sizeof(unsigned int), cudaMemcpyDeviceToDevice);
  unsigned int h_zero[1];
  h_zero[0] = 0;
  cudaMemcpy(d_cdf+sizeof(unsigned int)*(numBins/2-1),h_zero, sizeof(unsigned int), cudaMemcpyHostToDevice);

  for (int d=dmax-2,d>=0, d--){
  }
     // Free memory


  checkCudaErrors(cudaFree(d_data2));
  checkCudaErrors(cudaFree(d_max));
  checkCudaErrors(cudaFree(d_min));
//  checkCudaErrors(cudaFree(d_bins));

}
