#include <stdio.h>
#include <stdlib.h>
#include <igraph/igraph.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>
#include <iostream>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

//GLOBAL VARS
igraph_neimode_t OUTALL;

//KERNELS & PREP
void Naive_Test();
__global__ void Naive(int* d_matrix, int* d_result, int n_vertices);
__global__ void Naive_Hist(int* d_result, int* d_hist, int n_vertices);
void Naive_Prep(igraph_t &graph);
__global__ void TEST(int* test);
void TEST_PREP();

//CUDA ERROR
void checkCudaError(cudaError_t e, const char* in);

//FUNCTIONS
void linkage_covariance(igraph_t &graph);

//main
int main(int argc, char** argv) {
	//checks arguments
	if(argc < 3) {

		printf("\nToo few arguments. Usage: ./%s graphFile all/out\n", argv[0]);
		return -1;
	}

	//graph direction out or all
	if(!strcmp(argv[2], "out"))
		OUTALL = IGRAPH_OUT;
	else
		OUTALL = IGRAPH_ALL;
	
	struct timeval stop, start;
	gettimeofday(&start, NULL);

	//opens graph file passed as 1st argument
	FILE *inputFile;
	inputFile = fopen(argv[1], "r");
	if(inputFile == NULL)
	{
		printf("Could not load input file...\n");
		return 1;
	}
	
	igraph_t graph;

	//builds graph from file
	igraph_read_graph_ncol(&graph, inputFile, NULL, true, IGRAPH_ADD_WEIGHTS_NO, IGRAPH_DIRECTED);

	//TEST_PREP();
	//function
	//linkage_covariance(graph);
	Naive_Prep(graph);
	//Naive_Test();

	gettimeofday(&stop, NULL);
	printf("took %2f\n", (stop.tv_sec - start.tv_sec) * 1000.0f + (stop.tv_usec - start.tv_usec) / 1000.0f);
	return 0;
}

__global__ void Naive(int* d_matrix, int* d_result, int n_vertices) {

	int row = blockIdx.x;
	int col = threadIdx.x;
	int cval;

	

	if(row < n_vertices && col < n_vertices)
	for(int i = col; i < n_vertices; i += blockDim.x) {

		if(row == col) {
			d_result[row*blockDim.x + col] = 0;
			continue;
		}

		cval = 0;


		for(int j = 0; j < n_vertices; j++) {

			cval += d_matrix[row*n_vertices + j] * d_matrix[n_vertices*j + col];
			// if(row == 0 && col == 0)
			// printf("cval(%d,%d) = %d x = %d y = %d\n", row, col, cval, d_matrix[row*n_vertices + j], d_matrix[n_vertices*j + col]);
		}

		// if(row == 1 && col == 0)
		// 	printf("\ncval(0,1) = %d\n", cval);

		d_result[row*blockDim.x + col] = cval;
	}
	__syncthreads();

	if(col == 0)
		thrust::sort(thrust::seq, &d_result[row*n_vertices], &d_result[row*n_vertices] + n_vertices);
	// if(col == 0 && row == 30) {
	// 	//thrust::sort(thrust::seq, d_result, d_result + n_vertices);
	// 	//thrust::sort(thrust::seq, &d_result[row*n_vertices], &d_result[row*n_vertices] + n_vertices);
	// 	printf("\n");
	// 	for(int i = 0; i < n_vertices; i++)
	// 		printf("%d ", d_result[row*n_vertices + i]);
	// 	printf("\n");
	// }
}
__global__ void Naive_Hist(int* d_result, int* d_hist, int n_vertices) {

	int row = blockIdx.x;
	int row2 = threadIdx.x;
	bool equal;

	if(row < n_vertices && row2 < n_vertices)
	for(int i = row2; i < n_vertices; i += blockDim.x) {

		equal = false;

		for(int j = 0; j < n_vertices; j++) {

			if(d_result[row*n_vertices +j] == d_result[i*n_vertices + j])
				equal = true;
			else {
				equal = false;
				break;
			}
		}
		if(equal) {
			atomicAdd((unsigned long long int*)&d_hist[row],1);
			//if(row2 == 0 && row == 0)
				//printf("\nTEST hist(%d) = %d\n", row, d_hist[row]);
		}
	}
}

void Naive_Prep(igraph_t &graph) {

	int *matrix, n_vertices = igraph_vcount(&graph);
	long int vsize;
	
	igraph_vector_t vec;
	igraph_vector_init(&vec, 0);
	
	//initializes matrix and sets to zero
	matrix = (int *)malloc(n_vertices*n_vertices*sizeof(int));
	memset(matrix, 0, sizeof(int)*n_vertices*n_vertices);

	for(int i = 0; i < n_vertices; i++) {
		
		igraph_neighbors(&graph, &vec, i, OUTALL);
		vsize = igraph_vector_size(&vec);

		for(int j = 0; j < vsize; j++) {

			matrix[i*n_vertices + (int)VECTOR(vec)[j]] = 1;
		}
	}

	// printf("\n");
	// for(int i = 0; i < n_vertices; i++)
	// 	printf("%d ", matrix[i]);

	// printf("\n");
	// for(int i = 0; i < n_vertices; i++)
	// 	printf("%d ", matrix[i*n_vertices+1]);
	// printf("\n");
	//CUDA SHIT
	int hsize = n_vertices;
	int *d_matrix, *d_result, hist[hsize], *d_hist;
	cudaMalloc((void**)&d_matrix, sizeof(int)*n_vertices*n_vertices);
	cudaMalloc((void**)&d_result, sizeof(int)*n_vertices*n_vertices);
	cudaMalloc((void**)&d_hist, sizeof(int)*hsize);

	cudaMemcpy(d_matrix, matrix, sizeof(int)*n_vertices*n_vertices, cudaMemcpyHostToDevice);
	cudaMemset(d_result, 0, sizeof(int)*n_vertices*n_vertices);
	cudaMemset(d_hist, 0, sizeof(int)*hsize);
	memset(hist, 0, sizeof(int)*hsize);

	// dim3 threads(1024);
	// dim3 grid(ceil((float)n_vertices/threads.x));
	Naive<<<n_vertices, 1024>>>(d_matrix, d_result, n_vertices);
	Naive_Hist<<<n_vertices, 1024>>>(d_result, d_hist, n_vertices);
	
	cudaMemcpy(hist, d_hist, sizeof(int)*hsize, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	// for(int i = 0; i < hsize; i++)
	// 	printf("%d    %d\n", i, hist[i]);

	// for(int i = 1; i < hsize; i++) {
	// 	if ((hist[i] / i) > 0)
	// 		printf("%d    %d\n", i, (hist[i] / i));
	// }
}

void Naive_Test() {

	int test[25];
	for(int i = 1; i <= 25; i++){
		test[i-1] = i;
		//printf("%d\n", test[i-1]);
	}

	int *d_test, *d_result, result[25];

	cudaMalloc((void**)&d_test, sizeof(int)*25);
	cudaMalloc((void**)&d_result, sizeof(int)*25);
	cudaMemcpy(d_test, test, sizeof(int)*25, cudaMemcpyHostToDevice);
	cudaMemset(d_result, 0, sizeof(int)*25);

	Naive<<<5, 5>>>(d_test, d_result, 5);
	cudaDeviceSynchronize();

	cudaMemcpy(result, d_result, sizeof(int)*25, cudaMemcpyDeviceToHost);

	for(int i = 0; i < 5; i++) {
		printf("\n");
		for(int j = 0; j < 5; j++)
			printf("%d ", result[i*5 + j]);
	}
}

//function
void linkage_covariance(igraph_t &graph) {

	//gets number of vertices
	int n_vertices = igraph_vcount(&graph);

	//neighbor vectors and init, holds adj vertices
	igraph_vector_t neisVec1, neisVec2, compVec;
	igraph_vector_init(&neisVec1, 1);
	igraph_vector_init(&neisVec2, 1);
	igraph_vector_init(&compVec, 1);


	igraph_vector_t arrVec[n_vertices];
	
	for(int j = 0; j < n_vertices; j++)
	{
		igraph_vector_init(&arrVec[j], 0);
	}
					
	//finds similar vertices
	for(int i = 0; i < n_vertices; i++) {
		
		igraph_neighbors(&graph, &neisVec1, i, OUTALL);
		//inner loop
		for(int j = i+1; j < n_vertices; j++) {

			igraph_neighbors(&graph, &neisVec2, j, OUTALL);
			igraph_vector_intersect_sorted(&neisVec1, &neisVec2, &compVec);
			if (igraph_vector_size(&compVec) > 0)
			{
				igraph_vector_push_back(&arrVec[i], igraph_vector_size(&compVec));
				igraph_vector_push_back(&arrVec[j], igraph_vector_size(&compVec));
			}
		}
	}

	long int histo[n_vertices];
	memset(histo, 0, sizeof(long int)*n_vertices);
	int count = 0, countMax = -1;

	for(int i = 0; i < n_vertices; i++) {
		count = 0;
		igraph_vector_sort(&arrVec[i]);
		// printf("%d:\n", i);
		// for(int k = 0; k< igraph_vector_size(&arrVec[i]); k++)
		// {
		// 	printf("%ld-", (long int)VECTOR(arrVec[i])[k]);
		// }
		// printf("\n");
		for(int j = 0; j < n_vertices; j++) {
			if(igraph_vector_size(&arrVec[i]) != igraph_vector_size(&arrVec[j]))
				continue;

			igraph_vector_sort(&arrVec[j]);
			
			if(igraph_vector_all_e(&arrVec[i], &arrVec[j]))
			{				
				count++;
			}
		}

		if(countMax < count)
			countMax = count;
		// if (count == 1)
			// printf("\n%d - %d\n", i, count);
		histo[count]++;
	}

	for(int i = 1; i <= countMax; i++) {
		if ((long) (histo[i] / i) > 0)
			printf("%d    %ld\n", i, (long) (histo[i] / i));
	}
}

//CUDA ERROR
void checkCudaError(cudaError_t e, const char* in) {
	if (e != cudaSuccess) {
		printf("CUDA Error: %s, %s \n", in, cudaGetErrorString(e));
		exit(EXIT_FAILURE);
	}
}

//TEST PREP & KERNEL
void TEST_PREP() {
	
	int test[13] = {0, 0, 0, 9, 8, 2, 0, 1, 3, 6, 4, 5, 7}, *d_test, result[13];
		
	cudaMalloc((void**)&d_test, 13*sizeof(int));
	cudaMemcpy(d_test, test, 13*sizeof(int), cudaMemcpyHostToDevice);

	
	TEST<<<1,1>>>(d_test);
	checkCudaError(cudaGetLastError(), "Checking Last Error, Kernel Launch");
	
	cudaMemcpy(result, d_test, 13*sizeof(int), cudaMemcpyDeviceToHost);

	for(int i = 0; i < 13; i++)
		printf("%d\n", result[i]);
}
__global__ void TEST(int* test) {

	thrust::sort(thrust::seq, test, test + 13);
}