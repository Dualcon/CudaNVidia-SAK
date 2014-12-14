/**
 * @file functions.c
 * @brief Functions of cuda-sak
 *
 * Functions' implementation of cuda-sak.
 *  
 * Functions: device, handleError, filename, callkernel, genMaps, occupancy, 
 * preprocessor, reduction, skeleton, timing, unified, students, about,
 * testall, buildString
 * @version 1
 */

#include "functions.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <assert.h>

#include "memory.h"

/**
 * This function will be used when the user chooses option -d "device".
 * 
 * @return A single string with code for getting device properties
 */
char * device() {

	char * str[94] = {

		"int count;",
		"cudaDeviceProp prop;",
		"",
		"cudaGetDeviceCount(&count);",
		"",
		"for (int i=0; i<count; i++) {",
		"	cudaGetDeviceProperties(&prop, i);",
		"	printf(\"Name: %s\\n\", prop.name);",
		"	printf(\"Total global mem: %ld\\n\", prop.totalGlobalMem);",
		"	printf(\"Shared mem per block: %ld\\n\", prop.sharedMemPerBlock);",
		"	printf(\"Total registers per block %d\\n\", prop.regsPerBlock);",
		"	printf(\"Warp size: %d\\n\", prop.warpSize);",
		"	printf(\"Mem pitch: %ld\\n\", prop.memPitch);",
		"	printf(\"Max threads per block: %d\\n\", prop.maxThreadsPerBlock);",
		"	printf(\"Max thread dimensions: %d, %d, %d\\n\", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);",
		"	printf(\"Max grid size: %d, %d, %d\\n\", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);",
		"	printf(\"Clockrate: %d\\n\", prop.clockRate);",
		"	printf(\"Total constant memory: %ld\\n\", prop.totalConstMem);",
		"	printf(\"Compute capability: %d, %d\\n\", prop.minor, prop.major);",
		"	printf(\"Alignment requirement: %ld\\n\", prop.textureAlignment);",
		"	printf(\"Pitch alignment: %ld\\n\", prop.texturePitchAlignment);",
		"	printf(\"Device copy overlap: \");",
		"	if (prop.deviceOverlap)",
		"		printf(\"Enabled\\n\");",
		"	else",
		"		printf(\"Disabled\\n\");",
		"	printf(\"Number of multiprocessors: %d\\n\", prop.multiProcessorCount);",
		"	printf(\"Run-time limit for kernel execution: \");",
		"	if(prop.kernelExecTimeoutEnabled)",
		"		printf(\"Enabled\\n\");",
		"	else",
		"		printf(\"Disabled\\n\");",
		"	printf(\"Integrated device: \");",
		"	if(prop.integrated)",
		"		printf(\"Integrated\\n\");",
		"	else",
		"		printf(\"Discrete\\n\");",
		"	printf(\"Can map host memory: \");",
		"	if(prop.canMapHostMemory)",
		"		printf(\"Yes\\n\");",
		"	else",
		"		printf(\"No\\n\");",
		"	printf(\"Compute Mode %d\\n\", prop.computeMode);",
		"	printf(\"Max texture 1d: %d\\n\", prop.maxTexture1D);",
		"	printf(\"Max texture 1d linear: %d\\n\", prop.maxTexture1DLinear);",
		"	printf(\"Max texture 2d: %d, %d\\n\", prop.maxTexture2D[0], prop.maxTexture2D[1]);",
		"	printf(\"Max texture 2d linear: %d, %d, %d\\n\", prop.maxTexture2DLinear[0], prop.maxTexture2DLinear[1], prop.maxTexture2DLinear[2]);",
		"	printf(\"Max texture 2d gather: %d, %d\\n\", prop.maxTexture2DGather[0], prop.maxTexture2DGather[1]);",
		"	printf(\"Max texture 3d: %d, %d, %d\\n\", prop.maxTexture3D[0], prop.maxTexture3D[1], prop.maxTexture3D[2]);",
		"	printf(\"Max texture codemap: %d\\n\", prop.maxTextureCubemap);",
		"	printf(\"Max texture 1d layered: %d, %d\\n\", prop.maxTexture1DLayered[0], prop.maxTexture1DLayered[1]);",
		"	printf(\"Max texture 2d layered: %d, %d, %d\\n\", prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[2], prop.maxTexture2DLayered[2]);",
		"	printf(\"Max texture codemap layered: %d, %d\\n\", prop.maxTextureCubemapLayered[0], prop.maxTextureCubemapLayered[1]);",
		"	printf(\"Max surface 1d: %d\\n\", prop.maxSurface1D);",
		"	printf(\"Max surface 2d: %d, %d\\n\", prop.maxSurface2D[0], prop.maxSurface2D[1]);",
		"	printf(\"Max surface 3d: %d, %d, %d\\n\", prop.maxSurface3D[0], prop.maxSurface3D[1], prop.maxSurface3D[2]);",
		"	printf(\"Max surface 1d: %d, %d\\n\", prop.maxSurface1DLayered[0], prop.maxSurface1DLayered[1]);",
		"	printf(\"Max surface 2d layered: %d, %d, %d\\n\", prop.maxSurface2DLayered[0], prop.maxSurface2DLayered[1], prop.maxSurface2DLayered[2]);",
		"	printf(\"Max surface codemap: %d\\n\", prop.maxSurfaceCubemap);",
		"	printf(\"Max surface codemap layered: %d, %d\\n\", prop.maxSurfaceCubemapLayered[0], prop.maxSurfaceCubemapLayered[1]);",
		"	printf(\"Surface alignment: %ld\\n\", prop.surfaceAlignment);",
		"	printf(\"Concurrent kernels: \");",
		"	if(prop.concurrentKernels)",
		"		printf(\"Yes\\n\");",
		"	else",
		"		printf(\"No\\n\");",
		"	printf(\"ECC enabled: \");",
		"	if(prop.ECCEnabled)",
		"		printf(\"Enabled\\n\");",
		"	else",
		"		printf(\"Disabled\\n\");",
		"	printf(\"PCI Bus Identifier: %d\\n\", prop.pciBusID);",
		"	printf(\"PCI Device Identifier: %d\\n\", prop.pciDeviceID);",
		"	printf(\"PCI Domain Identifier: %d\\n\", prop.pciDomainID);",
		"	printf(\"TCC Driver: \");",
		"	if(prop.tccDriver)",
		"		printf(\"Using TCC Driver\\n\");",
		"	else",
		"		printf(\"Not using TCC Driver\\n\");",
		"	printf(\"Async Engine Count: \");",
		"	if(prop.asyncEngineCount)",
		"		printf(\"Yes\\n\");",
		"	else",
		"		printf(\"No\\n\");",
		"	printf(\"Unified addressing: \");",
		"	if(prop.unifiedAddressing)",
		"		printf(\"Shares a unified address with the host\\n\");",
		"	else",
		"		printf(\"Does not share a unified adress with the host\\n\");",
		"	printf(\"Memory ClockRate: %d\\n\", prop.memoryClockRate);",
		"	printf(\"Memory Bus Width: %d\\n\", prop.memoryBusWidth);",
		"	printf(\"L2 cache size: %d\\n\", prop.l2CacheSize);",
		"	printf(\"Max threads per multiprocessor: %d\\n\", prop.maxThreadsPerMultiProcessor);",
		"}"

	};

	return buildString(str, 94);

}

/**
 * This function will be used when the user chooses option -e "handleerror".
 * 
 * @return A single string with code for handling errors in CUDA
 */
char * handleError() {

	char * string[47] = {

		"#ifndef __HANDLE_ERROR_H__",
		"#define __HANDLE_ERROR_H__",
		"/*-------------------------------------------------------------------",
		" * Function to process CUDA errors",
		" * @param err [IN] CUDA error to process (usually the code returned",
		" *	 by the cuda function)",
		" * @param line [IN] line of source code where function is called",
		" * @param file [IN] name of source file where function is called",
		" * @return on error, the function terminates the process with ",
		" * 		EXIT_FAILURE code.",
		" * source: \"CUDA by Example: An Introduction to General-Purpose \"",
		" * GPU Programming\", Jason Sanders, Edward Kandrot, NVIDIA, July 2010",
		" * @note: the function should be called through the ",
		" * 	macro 'HANDLE_ERROR'",
		" *------------------------------------------------------------------*/",
		"static void HandleError( cudaError_t err,",
		"                         const char *file,",
		"                         int line ) {",
		"    if (err != cudaSuccess) ",
		"    {",
		"        printf( \"[ERROR] '%s' (%d) in '%s' at line '%d'\\n\",",
		"		cudaGetErrorString(err),err,file,line);",
		"        exit( EXIT_FAILURE );",
		"    }",
		"}",
		"/* The HANDLE_ERROR macro */",
		"#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__ ))",
		"",
		"/* ",
		" * Function that check a kernel execution, exiting ",
		" * if execution errors are found. The function should not be ",
		" * called directly, it should be called through the CHECK_KERNEL_EXEC",
		" * macro.",
		" */",
		"static void CheckKernelExec(const char *file,const int line){",
		"        cudaError_t err;",
		"        err = cudaGetLastError();",
		"        if( err != cudaSuccess ){",
		"                fprintf(stderr,\"[ERROR] '%s' (%d) in '%s' at line '%d'\\n\",",
		"                                cudaGetErrorString(err),err,file,line);",
		"                exit( EXIT_FAILURE );",
		"        }",
		"}",
		"",
		"#define CHECK_KERNEL_EXEC()     CheckKernelExec(__FILE__,__LINE__)",
		"",
		"#endif"

	};

	return buildString(string, 47);
}

/**
 * This function will be used when the user chooses option -f "filename".
 * This function writes the output of the chosen function to the file.
 *  
 * @param text Text to be written to the file
 * @param fname Name of the file to be written to
 */
void filename(char *text, char *fname) {

    	FILE * f = fopen(fname, "w");

	if (f == NULL) {

		printf("Error opening file!\n");
		exit(EXIT_FAILURE);
	}
	 
	fprintf(f, "%s", text);
	fclose(f);

	FREE(text);

}


/**
 * This function will be used when the user chooses option -k "callkernel".
 * 
 * @param name Name of the kernel to be called
 * @return A single string with code for calling the kernel
 */
char * callKernel(char * name) {

	char * aux = MALLOC( strlen(name) + strlen("<<<blocks,grid>>>(TODO:parameters);") + 1 );

	strcpy(aux, name);
	strcat(aux, "<<<blocks,grid>>>(TODO:parameters);");

	char * string[6] = {

		"dim3 grid(TODO:set grid dimensions);",
		"dim3 blocks(TODO:set block dimensions);",
		aux,
		"cudaDeviceSynchronize();",
		"cudaError_t err = cudaGetLastError();",
		"HANDLE_ERROR( err );"

	};

	char * result = buildString(string, 6);

	FREE(aux);

	return result;

}

/**
 * This function will be used when the user chooses option -m "genmaps".
 * Generates the mapping for n-D geometries to 1-D kernel.
 *
 * @param dimension Dimension (1D-3D) to be mapped to 1-D
 * @return A single string with code for mapping geometries to 1-D kernel
 */
char * genMaps (int dimension) {

	char * oneD_to_oneD[9] = {

		"int NumThreadsPerBlock = blockDim.x;",
		"int GlobalID_1D = blockIdx.x * blockDim.x + threadIdx.x;",
		"int stride = gridDim.x * NumThreadsPerBlock;",
		"int idx = GlobalID_1D;",
		"",
		"while( idx < num_elms){",
		"	c[idx] = a[idx] + b[idx];",
		"	idx += stride;",
		"}"

	};

	char * twoD_to_oneD[12] = {

		"int NumThreadsPerBlock = blockDim.x * blockDim.y;",
		"int globalID_2D = blockIdx.y * gridDim.x * NumThreadsPerBlock",
		"		+ blockIdx.x * NumThreadsPerBlock",
		"		+ threadIdx.y * blockDim.x",
		"		+ threadIdx.x;",
		"int stride = gridDim.x * gridDim.y * NumThreadsPerBlock;",
		"int idx = globalID_2D;",
		"",
		"while( idx < num_elms){",
		"	c[idx] = a[idx] + b[idx];",
		"	idx += stride;",
		"}"

	};

	char * threeD_to_oneD[14] = {

		"int NumThreadsPerBlock = blockDim.x * blockDim.y * blockDim.z;",
		"int globalID_3D =  blockIdx.z * gridDim.x * gridDim.y * NumThreadsPerBlock",
		"		+ blockIdx.y * gridDim.x * NumThreadsPerBlock",
		"		+ blockIdx.x * NumThreadsPerBlock",
		"		+ threadIdx.z * blockDim.x * blockDim.y",
		"		+ threadIdx.y * blockDim.x",
		"		+ threadIdx.x;",
		"int stride = gridDim.x * gridDim.y * gridDim.z * NumThreadsPerBlock;",
		"int idx = globalID_3D;",
		"",
		"while( idx < num_elms){",
		"	c[idx] = a[idx] + b[idx];",
		"	idx += stride;",
		"}"

	};

	switch (dimension) {
		case 1:
			return buildString(oneD_to_oneD, 9);
		case 2:
			return buildString(twoD_to_oneD, 12);
		case 3:
			return buildString(threeD_to_oneD, 14);
		default:
			return "Invalid dimension!\n";
	}

}


/**
 * This function will be used when the user chooses option -o "occupancy".
 *  
 * @return A single string with code that uses the CUDA 6.5 occupancy API
 */
char * occupancy () {

	char * string[46] = {

		"#include \"stdio.h\"",
		"#if __CUDA_ARCH__ >= 650",
		"",
		"__global__ void MyKernel(int *array, int arrayCount)",
		"{ ",
		"	int idx = threadIdx.x + blockIdx.x * blockDim.x; ",
		"	if (idx < arrayCount) ",
		"	{ ",
		"		array[idx] *= array[idx]; ",
		"	} ",
		"}",
		"",
		"void main(int *array, int arrayCount) ",
		"{ ",
		"	int blockSize;   // The launch configurator returned block size ",
		"	int minGridSize; // The minimum grid size needed to achieve the ",
		"			 // maximum occupancy for a full device launch ",
		"	int gridSize;    // The actual grid size needed, based on input size ",
		"",
		"	cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, ",
		"                                      MyKernel, 0, arrayCount); ",
		"	// Round up according to array size ",
		"	gridSize = (arrayCount + blockSize - 1) / blockSize; ",
		"",
		"	MyKernel<<< gridSize, blockSize >>>(array, arrayCount); ",
		"",
		"	cudaDeviceSynchronize(); ",
		"",
		"	// calculate theoretical occupancy",
		"	int maxActiveBlocks;",
		"	cudaOccupancyMaxActiveBlocksPerMultiprocessor( &maxActiveBlocks, ",
		"                                                 MyKernel, blockSize, ",
		"                                                 0);",
		"",
		"	int device;",
		"	cudaDeviceProp props;",
		"	cudaGetDevice(&device);",
		"	cudaGetDeviceProperties(&props, device);",
		"",
		"	float occupancy = (maxActiveBlocks * blockSize / props.warpSize) / ",
		"                         (float)(props.maxThreadsPerMultiProcessor / ",
		"			  props.warpSize);",
		"",
		"	printf(\"Launched blocks of size %d. Theoretical occupancy: %f\\n\", blockSize, occupancy);",
		"}",
		"#endif"

	};

	return buildString(string, 46);

}

/**
 * This function will be used when the user chooses option -p "preprocessor".
 * 
 * @param architecture Desired architecture of the preprocessor code 
 * @return A single string with code for preprocessor
 */
char * preprocessor(int architecture) {

	char arch[128];

	snprintf(arch, 128, "%d", architecture);

	char * arch_prefix = "#if __CUDA_ARCH__ >= ";
	char * arch_middle = "\n\t/* TODO: insert code arch >= ";
	char * arch_suffix = " here */\n#endif /* if */\n";

	size_t strSize = strlen(arch_prefix) + strlen(arch_middle) + strlen(arch_suffix) + 2 * strlen(arch) + 1;
	char * arch_final = MALLOC(strSize);

	sprintf(arch_final, "%s%s%s%s%s", arch_prefix, arch, arch_middle, arch, arch_suffix);

	return arch_final;

}

/**
 * This function will be used when the user chooses option -r "reduction".
 *
 * @return A single string with code for the traditional log 2 N reduction algorithm
 */
char * reduction() {

	char * string[33] = {

		"__global__ void TODO:kernel_name(float *in, float *out, unsigned int num_elms) {",
		"",
		"	int GlobalID = blockIdx.x * blockDim.x + threadIdx.x;",
		"	int stride = gridDim.x * blockDim.x;",
		"",
		"	extern __shared__ float reduction_share[];",
		"	float thread_sum = 0.0;",
		"",
		"	int idx = GlobalID;",
		"",
		"	while( idx < num_elms ){",
		"		thread_sum += in[idx];",
		"		idx = idx + stride;",
		"	}",
		"",
		"	reduction_share[threadIdx.x] = thread_sum;",
		"	__syncthreads();",
		"",
		"	int num_threads_on = blockDim.x / 2;",
		"",
		"	while(num_threads_on > 0) {",
		"",
		"		if(threadIdx.x < num_threads_on) {",
		"			reduction_share[threadIdx.x] += reduction_share[threadIdx.x + num_threads_on];",
		"		}",
		"",
		"		num_threads_on = num_threads_on / 2;",
		"		__syncthreads();",
		"	}",
		"",
		"	if(threadIdx.x == 0)",
		"		out[blockIdx.x] = reduction_share[0];",
		"}"

	};
 
	return  buildString(string, 33);
}

/**
 * This function will be used when the user chooses option -s "skeleton".
 * 
 * @param kernel_name Name of the kernel to be created 
 * @return A single string with general code and hints for a CUDA program
 */
char * skeleton (char * kernel_name) {

	// Preparing 1st line where kernel_name appears
	char * aux1 = "__global__ void ";
	char * aux2 = "(TODO:add parameters) {";
	char * aux3 = MALLOC( strlen(kernel_name) + strlen(aux1) + strlen(aux2) + 1);
	strcpy(aux3, aux1);
	strcat(aux3, kernel_name);
	strcat(aux3, aux2);

	// Preparing 2nd line where kernel_name appears
	char * aux4 = "	";
	char * aux5 = "<<<blocks,grid>>>(TODO:parameters);";
	char * aux6 = MALLOC(strlen(kernel_name) + strlen(aux4) + strlen(aux5) + 1);
	strcpy(aux6, aux4);
	strcat(aux6, kernel_name);
	strcat(aux6, aux5);

	char * string[51] = {
		"#include <stdio.h>",
		"#include \"HandleError.h\"",
		"",
		aux3,	//__global__ void kernel_name(TODO:add parameters){
		"	//TODO: Kernel code here",
		"}",
		"",
		"int main(void) {",
		"	/*TODO: declare and initialize variables for device and host",
		"	* Example:",
		"	* int histo_h[256];	",
		"	* unsigned int *histo_d;",
		"	*/",
		"",
		"	/*TODO: prepare device memory",
		"	* Example:",
		"	* size_t histo_size_bytes = 256 * sizeof(unsigned int);",
		"	* HANDLE_ERROR( cudaMalloc((int**)&histo_d,histo_size_bytes) );",
		"	*/",
		"	",
		"	HANDLE_ERROR( cudaMalloc((TODO:add cast to variable type)&TODO:device variable name,TODO:size to allocate in the device memory);",
		"",
		"	/*TODO: initialize device variables",
		"	* Example: ",
		"	* HANDLE_ERROR(cudaMemcpy(buffer_d,buffer_h,buffer_size_bytes,cudaMemcpyHostToDevice));",
		"	*/",
		"",
		"	HANDLE_ERROR(cudaMemcpy(TODO:device variable,TODO:host variable,TODO:size allocated to device variable, cudaMemcpyHostToDevice))",
		"",
		"	//Kernel Call",
		"	dim3 grid(TODO:set grid dimensions);",
		"	dim3 blocks(TODO:set block dimensions);",
		aux6,	//kernel_name<<<blocks,grid>>>(TODO:parameters);
		"	cudaDeviceSynchronize();",
		"	cudaError_t err = cudaGetLastError();",
		"	HANDLE_ERROR( err );",
		"",
		"	/*TODO: copy variables from device to host",
		"	* Example: HANDLE_ERROR(cudaMemcpy(histo_h,histo_d,histo_size_bytes,cudaMemcpyDeviceToHost));",
		"	*/",
		"	",
		"	HANDLE_ERROR(cudaMemcpy(TODO:host variable,TODO:device variable,TODO: size allocated to device variable, cudaMemcpyDeviceToHost));",
		"",
		"	/*TODO: free device memory",
		"	* Example: HANDLE_ERROR(cudaFree(buffer_d));",
		"	*/",
		"	",
		"	HANDLE_ERROR(cudaFree(TODO:variable to free));",
		"",
		"	return 0;",
		"}"
	};

	char * final = buildString(string, 51);

	FREE(aux3);
	FREE(aux6);

	return final;

}

/**
 * This function will be used when the user chooses option -t "timing".
 * 
 * @return A single string with general code used for timing the execution of a CUDA kernel
 */
char * timing() {

	char * string[20] = {

		"/* Variables declaration */",
		"float ElapsedTime;",
		"",
		"/* Create timer */",
		"cudaEvent_t start, stop;",
		"HANDLE_ERROR( cudaEventCreate( &start ) );",
		"HANDLE_ERROR( cudaEventCreate( &stop ) );",
		"",
		"/* Start timer */",
		"HANDLE_ERROR( cudaEventRecord( start, 0 ) );",
		"",
		"/* Code goes here */",
		"",
		"/* Terminate timer */",
		"HANDLE_ERROR( cudaEventRecord( stop, 0 ) );",
		"HANDLE_ERROR( cudaEventSynchronize( stop ) );",
		"HANDLE_ERROR( cudaEventElapsedTime( &ElapsedTime, start, stop) );",
		"HANDLE_ERROR( cudaEventDestroy( start ) );",
		"HANDLE_ERROR( cudaEventDestroy( stop ) );",
		"printf(\"ElapsedTime:%3.6f\\n\", ElapsedTime);"

	};
	
	return buildString(string, 20);

}

/**
 * This function will be used when the user chooses option -u "unified".
 * 
 * @return A single string with general code that uses the CUDA 6.5 unified memory unified API
 */
char * unified() {

	char * string[44] = {

		"#include <stdio.h>",
		"#include <stdlib.h>",
		"#include <math.h> ",
		"#define SIZE 64",
		"",
		"#if __CUDA_ARCH__ >= 650",
		"__global__ void kernel(int *a_k, int *b_k, int *c_k, int size) {",
		"  	/* idx = index that the thread is processing */",
		"	int idx = blockIdx.x*blockDim.x+threadIdx.x;",
		"	int stride = gridDim.x * blockDim.x;",
		"",
		"	while(idx < size) {",
		"		c_k[idx] = a_k[idx] * b_k[idx];",
		"		idx = idx+stride;",
		"	}",
		"}",
		"",
		"int main(int argc char *argv[]) {",
		"	int * a_d; ",
		"	int * b_d;",
		"	int * c_d;",
		"",
		"	cudaMallocManaged(&a_d, SIZE * sizeof(int));",
		"	cudaMallocManaged(&b_d, SIZE * sizeof(int));",
		"	cudaMallocManaged(&c_d, SIZE * sizeof(int));",
		"",
		"	for (int i=0; i < SIZE; i++) {",
		"		a_d[i] = i * i;",
		"		b_d[i] = i;",
		"	}",
		"",
		"	kernel<<<24>>>(a_d, b_d, c_d, SIZE);",
		"",
		"	for (int i=0; i < SIZE; i++) {",
		"		printf(\"indice %d= %d\\n\", i, c_d[i]);",
		"	}",
		"",
		"	cudaFree(a_d);",
		"	cudaFree(b_d);",
		"	cudaFree(c_d);",
		"",
		"	return 0;",
		"}",
		"#endif"

	};

	return buildString(string, 44);

}

/**
 * This function will be used when the user chooses option --students "students".
 * 
 * @param fname Name of the file to be injected with code
 * @param stringToWrite Code to be injected to the file
 * @param lineToWrite Line of the original file in which the code will be injected
 */
void students(const char * fname, char * stringToWrite, int lineToWrite) {

	int actualLine = 1;
	const char * finalFilename = "temp_cuda-sak";
	int done = 0;

	// file to read
	FILE * fp;
	fp = fopen(fname, "r");
	if( fp == NULL ) {
		perror("Error while opening the file.\n");
		exit(EXIT_FAILURE);
	}

	// create file
	FILE * fp2;
	fp2 = fopen(finalFilename, "w");
	if( fp2 == NULL ) {
		perror("Error while creating the file.\n");
		exit(EXIT_FAILURE);
	}

	char * bufferPtr;
	char ** linePtr;
	size_t bufferSize = 4096 * sizeof(char);
	size_t * linePtrSize = &bufferSize;
	bufferPtr = MALLOC(bufferSize);
	linePtr = &bufferPtr;

	while(!done) {

		if (actualLine == lineToWrite) {
			fprintf(fp2, "%s", stringToWrite);	

		} else {
			ssize_t result = getline(linePtr, linePtrSize, fp);

			if (result == -1) {

				if (feof(fp)) {
					fclose(fp);
					fclose(fp2);

					if ( (actualLine -2) < lineToWrite) {
						printf("The file doesn't have line number %d.\n", lineToWrite);

					}
			
					int status = remove(fname);

					if (status != 0) {
						perror("Error while deleting the file.\n");
						exit(EXIT_FAILURE);
					}

					status = rename(finalFilename, fname);

					if (status != 0) {
						perror("Error while renaming the temp file.\n");
						exit(EXIT_FAILURE);
					}

					FREE(stringToWrite);
					FREE(bufferPtr);
					return;

				} else {
					perror("Error while opening the file.\n");
					exit(EXIT_FAILURE);

				}
			}

			fprintf(fp2, "%s", bufferPtr);
			FREE(bufferPtr);
		}

		actualLine++;
	}
	
	FREE(stringToWrite);

}

/**
 * This function will be used when the user chooses option --about "about".
 * 
 * @return A single string with the credits of the application
 */
char * about() {

	char * str[5] = {
		"cuda-sak – Credits:",
		"Daniel Vieira, David Matos, Nelson Nunes, Rafael Costa",
		"Course of High Performance Computing 2014-15 / Prof. Patricio Domingues / Master HPC ",
		"CAD 2014/2015 – Coding Project 4/7 in Informatics Engineering – Mobile Computing",
		"Polytechnic of Leiria, Portugal"
	};

	return buildString(str, 5);

}

/**
 * This function will be used when the user chooses option -a "testall".
 * 
 * @return A single string containing all the output from the other cuda-sak functions
 */
char * testAll() {
/*
	char * deviceStr = device();
	char * handleErrorStr = handleError();
	char * callKernelStr = callKernel("kernel");
	char * genMapsStr = genMaps(3);
	char * occupancyStr = occupancy();
	char * preprocessorStr = preprocessor(650);
	char * reductionStr = reduction();
	char * skeletonStr = skeleton("kernel");
	char * timingStr = timing();
	char * unifiedStr = unified();
		
	char * str[20] = {

		"Option -d -- device", 
		deviceStr,
		"option -h -- handleError", 
		handleErrorStr, 
		"Option -k -- callKernel", 
		callKernelStr,
		"Option -m -- genMaps", 
		genMapsStr, 
		"Option -o -- occupancy",
		occupancyStr, 
		"Option -p -- preprocessor", 
		preprocessorStr,
		"Option -r -- reduction", 
		reductionStr, 
		"Option -s -- skeleton",
		skeletonStr, 
		"Option -t -- timing", 
		timingStr,
		"Option -u -- unified", 
		unifiedStr
	};
	
	FREE(deviceStr);
	FREE(handleErrorStr);
	FREE(callKernelStr);
	FREE(genMapsStr);
	FREE(occupancyStr);
	FREE(preprocessorStr);
	FREE(reductionStr);
	FREE(skeletonStr);
	FREE(timingStr);
	FREE(unifiedStr);
*/
	char * deviceStr = device();
	char * str[2] = {
		"Option -d -- device", 
		deviceStr
	};
	FREE(deviceStr);
	return buildString(str, 2);

}

/**
 * This function will be used (auxiliary) to join multiple strings.
 * 
 * @param array The array of strings to join
 * @param length Length of the strings' array 
 * @return The result of joining all the strings in one (with "\n" in between)
 */
char * buildString(char ** array, int length) {

	char * string = MALLOC(strlen(array[0]) + 1);
	strcpy(string, array[0]);

	for (int i = 1; i < length; i++) {
		string = REALLOC(string, strlen(string) + strlen("\n") + strlen(array[i]) + 1);
		strcat(string, "\n");
		strcat(string, array[i]);
	}

	string = (char *) REALLOC(string, strlen(string) + strlen("\n") + 1);
	strcat(string, "\n");

	return string;

}
