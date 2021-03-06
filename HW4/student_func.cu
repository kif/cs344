//Udacity HW 4
//Radix Sorting
#define WORKGROUP_SIZE 256
#include "utils.h"
#include <thrust/host_vector.h>
#include <stdio.h>
/* Red Eye Removal
   ===============

   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */
void print_cuda_array(unsigned int* d_data, size_t size){
  unsigned int *h_data;
  h_data = (unsigned int *)malloc(sizeof(unsigned int)*size);
  cudaMemcpy(h_data, d_data, sizeof(unsigned int) * size, cudaMemcpyDeviceToHost);
  for (int i=0; i<size; i++){
      printf("%d ",h_data[i]);
  }
  printf("\n");
  free(h_data);
}

void sum_array(unsigned int* d_data, size_t size){
  unsigned int *h_data, sum=0;
  h_data = (unsigned int *)malloc(sizeof(unsigned int)*size);
  cudaMemcpy(h_data, d_data, sizeof(unsigned int) * size, cudaMemcpyDeviceToHost);
  for (int i=0; i<size; i++){
      sum+=h_data[i];
  }
  printf("Summed items: %d\n", sum);
  free(h_data);
}
void save_array(unsigned int* d_data, size_t size, char* prefix, int i){
  FILE *filePtr;
  char *name;
  name = (char *)malloc(80);
  sprintf(name,"%s-%02d.txt",prefix,i);
  filePtr = fopen(name,"w");
  unsigned int *h_data, sum=0;
  h_data = (unsigned int *)malloc(sizeof(unsigned int)*size);
  cudaMemcpy(h_data, d_data, sizeof(unsigned int) * size, cudaMemcpyDeviceToHost);
   for (int i = 0; i < size; i++) {
      fprintf(filePtr, "%d ", h_data[i]);
   }
   free(h_data);
   free(name);
}
__global__
void blelloch1( unsigned int* data,
				unsigned int step,
				unsigned int SIZE)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ((idx+step<=SIZE) && (idx % step) == 0 ){
    data[idx+step-1] += data[idx + step/2 - 1];
  }
}

__global__
void blelloch2( unsigned int* data,
				unsigned int step,
				unsigned int SIZE)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx == 0)
    data[SIZE-1] = data[SIZE/2-1];
}

__global__
void blelloch3( unsigned int* data,
				unsigned int step,
				unsigned int SIZE)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx == 0)
    data[SIZE/2-1] = 0;
}

__global__
void blelloch4( unsigned int* data,
				unsigned int step,
				unsigned int SIZE)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if ((idx+step<=SIZE) && ((idx % step) == 0 )){
    unsigned int  temp = data[idx + step/2 - 1];
    data[idx + step/2 -1] = data[idx + step - 1];
    data[idx + step - 1] += temp;
  }
}

unsigned int cumsum(unsigned int *d_array,
           			unsigned int dmax){

	unsigned int size = (1<<dmax),
		  	  sum=0;
//		  	  *h_array;
//  h_array = (unsigned int *)malloc(sizeof(unsigned int)*size);
  for (unsigned int d=0; d<(dmax); d++){
//    printf( "CUDA blelloch1 d= %d/%d (%d)\n",d,dmax,(1<<(d+1)));
    blelloch1<<<size/WORKGROUP_SIZE, WORKGROUP_SIZE>>>(d_array, (1<<(d+1)), size);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  }
//  print_cuda_array(d_array,size);
  //One should retrieve the sum of all data here

  cudaMemcpy(&sum, &d_array[size-1], 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
//  printf( "CUDA sum is %d\n", sum);
//  for (int i=0; i<size; i++) printf("%d ",h_array[i]); printf("\n");

  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  blelloch2<<<1,1>>>(d_array, 1, size);
  blelloch3<<<1,1>>>(d_array, 1, size);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  for (int d=dmax-2;d>=0; d--){
//    printf( "CUDA blelloch2 d= %d/%d (%d)\n",d,dmax,1<<(d+1));
    blelloch4<<<size/WORKGROUP_SIZE,WORKGROUP_SIZE>>>(d_array, 1<<(d+1), size);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

  }
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
  return sum;
}

__global__
void calc_predicate(unsigned int* data,
                    unsigned int* index0,
                    unsigned int* index1,
                    unsigned int bitPos,
                    unsigned int numElems){
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx<numElems){
    if (data[idx]&(1<<bitPos)){
      index1[idx] = 1;
      index0[idx] = 0;
    }
    else{
      index1[idx] = 0;
      index0[idx] = 1;
    }

  }
  else{
	  index0[idx] = 0;
	  index1[idx] = 0; //Fill the array with 0 until next power of two
  }
}

__global__
void reorder(unsigned int* inputVals,
             unsigned int* inputPos,
             unsigned int* outputVals,
             unsigned int* oututPos,
             unsigned int* index0,
             unsigned int* index1,
             unsigned int bitPos,
             unsigned int numElems,
             unsigned int offset){
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x, new_idx;
  if (idx<numElems){
   unsigned int value = inputVals[idx],
		   	    pos = inputPos[idx];
//		   	    offset = index0[numElems-1]; //todo check

    if ((value &(1<<bitPos))==0){
      new_idx = index0[idx];
    }
    else{
      new_idx = offset + index1[idx];
    }
    outputVals[new_idx] = value;
    oututPos[new_idx] = pos;
  }
}


void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
  unsigned int dmax = ((int)ceil(log((double)numElems)/log(2.0))),
		  	   padded_size = 1<<dmax,
		  	   sum0 = 0,
		  	   sum1 = 0;

//  printf("Size of the system: %d padded to %d\n",numElems, padded_size);
  //Malloc stuff
  unsigned int *d_index0,*d_index1;
  checkCudaErrors(cudaMalloc(&d_index0, (size_t)  padded_size*sizeof(unsigned int)));
  checkCudaErrors(cudaMalloc(&d_index1, (size_t)  padded_size*sizeof(unsigned int)));
  // do some work
  for (int i=0;i<8*sizeof(unsigned int);i++){
//    print_cuda_array(d_inputVals,40);
//    save_array(d_inputVals,numElems,"data",i);
    calc_predicate<<<padded_size/WORKGROUP_SIZE,WORKGROUP_SIZE>>>(d_inputVals,d_index0,d_index1,i,numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
//    sum_array(d_index0,padded_size);
//    sum_array(d_index1,padded_size);
//    print_cuda_array(d_index0,40);
//    print_cuda_array(d_index1,40);
//    save_array(d_index0,numElems,"predicate0",i);
    sum0 = cumsum(d_index0,dmax);
//    save_array(d_index0,numElems,"cumsum0",i);
    if ((sum0==0)||(sum0==numElems))
    	continue;
//    printf("Found %d items at zero for bit #%d\n",sum0,i);
    sum1 = cumsum(d_index1,dmax);
//    printf("Found %d items at one  for bit #%d\n",sum1,i);
    reorder<<<padded_size/WORKGROUP_SIZE,WORKGROUP_SIZE>>>(d_inputVals, d_inputPos, d_outputVals, d_outputPos,
    		d_index0, d_index1, i, numElems, sum0);
    cudaMemcpy(d_inputVals, d_outputVals, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_inputPos, d_outputPos, numElems * sizeof(unsigned int), cudaMemcpyDeviceToDevice);
//    print_cuda_array(d_outputVals,40);

  }
//  print_cuda_array(d_outputVals,numElems);
  //Free stuff
  checkCudaErrors(cudaFree(d_index0));
  checkCudaErrors(cudaFree(d_index1));
//  save_array(d_outputVals,numElems,"data",32);

}
