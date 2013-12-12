/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/

#include <stdio.h>
#include "utils.h"

__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  // Optimized for WG=8, treat 2048 vals per WG
  // each thread processes 256 val
  // unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int start = blockIdx.x*2048;
  __shared__ unsigned char lhist[10000];
  for (unsigned int i=0;i<1024;i++){
    unsigned int proc_bin = i + 1024*threadIdx.x;
    lhist[proc_bin] = 0;
  }
  for (unsigned char i=0;i<256;i++){
    unsigned int pos = start+i*8+threadIdx.x;
    if (pos<numVals){
      unsigned int val = vals[pos];
      unsigned int proc_bin = 1024*threadIdx.x + val;
      lhist[proc_bin]+=1;
    }
  }
  __syncthreads();
  //parallel reduce
  for (unsigned int i=0; i<1024; i+=8){
    unsigned int sum = (unsigned int)(lhist[i+threadIdx.x]) +
        lhist[i+threadIdx.x+1024] +
        lhist[i+threadIdx.x+2048] +
        lhist[i+threadIdx.x+3072] +
        lhist[i+threadIdx.x+4096] +
        lhist[i+threadIdx.x+5120] +
        lhist[i+threadIdx.x+6144] +
        lhist[i+threadIdx.x+7168];
        atomicAdd(&histo[i+threadIdx.x],sum);
  }

}


__global__
void nogood1Histo(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  //Optimized for WG=32, treat 2048 vals per WG
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible
//  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int start = blockIdx.x*2048;
  __shared__ unsigned int lvals[2048];
  __shared__ unsigned int lhist[1024];
  for (int i=0;i<64;i++){
    unsigned int j = i*32 + threadIdx.x;
    if (j+start<numVals)
      lvals[j] = vals[start+j];
    else
      lvals[j] = 0;

  }
  for (unsigned int i=0;i<32;i++){
    unsigned int proc_bin = i*32 + threadIdx.x;
    lhist[proc_bin] = 0;
  }
  __syncthreads();

  for (unsigned int i=0;i<32;i++){
    unsigned int proc_bin = i*32 + threadIdx.x;
    for (unsigned int j=0;j<2048;j++){
      if (lvals[j] == proc_bin)
        lhist[proc_bin]+=1;
    }
  }
  __syncthreads();
  for (unsigned int i=0;i<32;i++){
    unsigned int proc_bin = i*32 + threadIdx.x;
    atomicAdd(&histo[proc_bin],lhist[proc_bin]);
  }

}

__global__
void atomicHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx>=numVals)
    return;
  unsigned int value = vals[idx];
  atomicAdd(&histo[value],1);
}


void pseudoLocalAtomicHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               int numVals)
{	// WG size = 32 treat, each thread treat 32 vals
	  __shared__ unsigned int hist[1024];
	  for (unsigned int i=0;i<1024;i+=blockDim.x){
		  lhist[i] = 0;
	  }
	  __syncthreads();
//	  for (unsigned char i=0;i<1024;i=blockDim.x){
	  // do this until update is not overwritten:
	  for (unsigned char i=0;i<32;i+=1){
		  unsigned int idx = blockIdx.x * blockDim.x * (32+i) + threadIdx.x;
		  unsigned int bin = vals[idx];
		  do {
			  unsigned int myVal = lhist[bin] & 0x7FFFFFF; // read the current bin val
			  myVal = ((threadIdx.x & 0x1F) << 27) | (myVal + 1); // tag my updated val
			  lhist[bin] = myVal; // attempt to write the bin
		  } while (lhist[bin] != myVal); // while update overwritten
	  }
	  __syncthreads();
	  //Central atomic add
	  for (unsigned int i=0; i<1024; i+=blockDim.x){
		  unsigned int pos = i+threadIdx.x;
		  unsigned int val = lhist[pos];
		  if (val)
			  atomicAdd(&histo[pos], val);
	  }
}



void computeHistogram(const unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  //TODO Launch the yourHisto kernel
  printf("OK we are working on an input array of %d element and histogramming into %d bins\n",numElems,numBins);
  int BLOCK_SIZE=32;
  int PROC_PER_BLOCK=32*32;
  dim3 num_blocks((numElems+PROC_PER_BLOCK-1)/PROC_PER_BLOCK);
  dim3 block_size(BLOCK_SIZE);
  pseudoLocalAtomicHisto<<<num_blocks,block_size>>>(d_vals, d_histo, numElems);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
