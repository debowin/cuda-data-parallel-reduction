#ifdef _WIN32
#  define NOMINMAX 
#endif

#define NUM_ELEMENTS 1000000

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <time.h>
#include <sys/time.h>

// includes, kernels
#include "vector_reduction_kernel.cu"

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest( int argc, char** argv);

int ReadFile(unsigned int*, char* file_name);
unsigned int computeOnDevice(unsigned int* h_data, int array_mem_size, double hostMs);

extern "C" 
void computeGold( unsigned int* reference, unsigned int* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int 
main( int argc, char** argv) 
{
    runTest( argc, argv);
    return EXIT_SUCCESS;
}

////////////////////////////////////////////////////////////////////////////////
//! Run naive scan test
////////////////////////////////////////////////////////////////////////////////
void
runTest( int argc, char** argv) 
{
    int num_elements = NUM_ELEMENTS;
    int errorM = 0;

    srand(time(NULL));

    const unsigned int array_mem_size = sizeof( unsigned int) * num_elements;

    // allocate host memory to store the input data
    unsigned int* h_data = (unsigned int*) malloc( array_mem_size);

    // * No arguments: Randomly generate input data and compare against the 
    //   host's result.
    // * One argument: Read the input data array from the given file.
    switch(argc-1)
    {      
        case 1:  // One Argument
            errorM = ReadFile(h_data, argv[1]);
            if(errorM != num_elements)
            {
                printf("Error reading input file!\n");
                exit(1);
            }
        break;
        
        default:  // No Arguments or one argument
            // initialize the input data on the host to be integer values
            // between 0 and 1000
            for( unsigned int i = 0; i < num_elements; ++i) 
            {
                h_data[i] = floorf(1000*(rand()/(float)RAND_MAX));
            }
        break;  
    }
    // compute reference solution
    unsigned int reference = 0;
    struct timeval start_time, end_time;
    gettimeofday(&start_time,NULL);
    computeGold(&reference , h_data, num_elements);
    gettimeofday(&end_time,NULL);
    printf("Processing %d elements...\n", num_elements);
    double start_count = (double) start_time.tv_sec
        + 1.e-6 * (double) start_time.tv_usec;
    double end_count = (double) end_time.tv_sec +
        1.e-6 * (double) end_time.tv_usec;
    double host_ms = (double)( (end_count - start_count) * 1000);
    printf("CPU Processing time: %lf (ms)\n", host_ms);
    
    // **===-------- Modify the body of this function -----------===**
    unsigned int result = computeOnDevice(h_data, num_elements, host_ms);
    // **===-----------------------------------------------------------===**


    // We can use an epsilon of 0 since values are integral and in a range 
    // that can be exactly represented
    unsigned int epsilon = 0;
    unsigned int result_regtest = (abs(result - reference) <= epsilon);
    printf( "Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");
    printf( "device: %d  host: %d\n", result, reference);
    // cleanup memory
    free( h_data);
}

// Read a floating point vector into M (already allocated) from file
int ReadFile(unsigned int* V, char* file_name)
{
    unsigned int data_read = NUM_ELEMENTS;
    FILE* input = fopen(file_name, "r");
    unsigned i = 0;
    for (i = 0; i < data_read; i++) 
        fscanf(input, "%d", &(V[i]));
    return data_read;
}

// **===----------------- Modify this function ---------------------===**
// Take h_data from host, copies it to device, setup grid and thread 
// dimentions, excutes kernel function, and copy result of reduction back
// to h_data.
// Note: float* h_data is both the input and the output of this function.
unsigned int computeOnDevice(unsigned int* h_data, int num_elements, double host_ms)
{
    // Declare device array
    unsigned int *d_data, *d_final_sum;
    int blocks = ceil(num_elements/(float)BLOCK_SIZE);

    // Allocate device array memory
    cudaMalloc((void**)&d_data, sizeof(unsigned int)*num_elements);
    cudaMalloc((void**)&d_final_sum, sizeof(unsigned int));

    // Copy host array contents to device
    cudaMemcpy(d_data, h_data, sizeof(unsigned int)*num_elements, cudaMemcpyHostToDevice);

    // Launch Kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);    
    cudaEventRecord(start);
    reduction<<<blocks, BLOCK_SIZE>>>(d_data, d_final_sum, num_elements);
    cudaEventRecord(stop); 
    cudaEventSynchronize(stop);
    float device_ms = 0;
    cudaEventElapsedTime(&device_ms, start, stop);

    printf("GPU Processing time: %f (ms)\n", device_ms);
    printf("Speedup: %fX\n", host_ms/device_ms);

    // Copy device array contents back to host
    cudaMemcpy(h_data, d_final_sum, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_data);
    cudaFree(d_final_sum);
    return *h_data;
}
