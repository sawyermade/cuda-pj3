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

//NAIVE KERNELS & PREP
void Naive_Prep(igraph_t &graph);
__global__ void Naive(int* d_matrix, int* d_result, int n_vertices);
__global__ void Naive_Hist(int* d_result, int* d_hist, int n_vertices);

//TEST KERNELS & PREP
void TEST_PREP(igraph_t &graph);
__global__ void TEST(int* adj, int* lcm, int* sizes, int n);

//CUDA ERROR
void checkCudaError(cudaError_t e, const char* in);

//FUNCTIONS
void linkage_covariance(igraph_t &graph);
void LCM_cpu_baseline(igraph_t &graph);
int compare(const void* a, const void* b);

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
	else if(!strcmp(argv[2], "all"))
		OUTALL = IGRAPH_ALL;
	else {
		printf("\nInvalid Graph Direction. Use out or all.\nUsage: ./%s graphFile all/out\n", argv[0]);
	}
	
	//cpu timing shit
	struct timeval stop, start;
	

	//opens graph file passed as 1st argument
	FILE *inputFile;
	inputFile = fopen(argv[1], "r");
	if(inputFile == NULL)
	{
		printf("Could not load input file...\n");
		return 1;
	}
	
	//graph var and builds graph from file
	igraph_t graph;
	igraph_read_graph_ncol(&graph, inputFile, NULL, true, IGRAPH_ADD_WEIGHTS_NO, IGRAPH_DIRECTED);

	

	//cpu naive & needs tons of host memory
	// gettimeofday(&start, NULL);
	// LCM_cpu_baseline(graph);
	// gettimeofday(&stop, NULL);
	// printf("CPU Naive Running Time: %2f\n", (stop.tv_sec - start.tv_sec) * 1000.0f + (stop.tv_usec - start.tv_usec) / 1000.0f);

	//cpu optimized
	gettimeofday(&start, NULL);
	linkage_covariance(graph);
	gettimeofday(&stop, NULL);
	printf("CPU Optimized Running Time: %2f\n", (stop.tv_sec - start.tv_sec) * 1000.0f + (stop.tv_usec - start.tv_usec) / 1000.0f);

	//gpu naive
	Naive_Prep(graph);
	//Naive_Test();
	//TEST_PREP();
	
	
	return 0;
}

//uses adjaceny matrix, slow and takes a shit load of device memory, lots of zeros
__global__ void Naive(int* d_matrix, int* d_result, int n_vertices) {

	//each block takes care of a whole row
	//columns to be compared to same row are threads
	int row = blockIdx.x;
	int col = threadIdx.x;
	int cval;

	//compares vertice blockIdx.x to all other vertices, increments by blockDim
	if(row < n_vertices && col < n_vertices)
	for(int i = col; i < n_vertices; i += blockDim.x) {

		//sets graphs horizontal to 0
		if(row == i) {
			d_result[row*n_vertices + i] = 0;
			continue;
		}

		//sets to zero
		cval = 0;

		//gets row x col
		for(int j = 0; j < n_vertices; j++)
			cval += d_matrix[row*n_vertices + j] * d_matrix[n_vertices*j + i];

		//puts cval into graph
		d_result[row*n_vertices + i] = cval;
	}
	
	//syncs threads so new row is done and sorts it using thrust on thread 0
	__syncthreads();
	if(col == 0 && row < n_vertices)
		thrust::sort(thrust::device, &d_result[row*n_vertices], &d_result[row*n_vertices] + n_vertices);
}

//builds histogram, lots of zeros
__global__ void Naive_Hist(int* d_result, int* d_hist, int n_vertices) {

	//each block compares the same row to all others row2
	int row = blockIdx.x;
	int row2 = threadIdx.x;
	bool equal;

	//shared count for whole block/same vertice
	__shared__ int count;

	//one thread sets count to zero and syncsthreads.
	if(row2 == 0)
		count = 0;
	__syncthreads();

	//checks equality to other vertices
	if(row < n_vertices && row2 < n_vertices)
	for(int i = row2; i < n_vertices; i += blockDim.x) {

		//checks equality of vertices lcm
		equal = false;
		for(int j = 0; j < n_vertices; j++) {

			if(d_result[row*n_vertices +j] == d_result[i*n_vertices + j])
				equal = true;
			else {
				equal = false;
				break;
			}
		}

		//adds to count if vertices are equal
		if(equal)
			atomicAdd(&count, 1);
	}

	//syncsthreads so count is done and increments hist[count]
	__syncthreads();
	if(row < n_vertices && row2 == 0 && count > 0)
		atomicAdd(&d_hist[count], 1);
}

void Naive_Prep(igraph_t &graph) {

	//creates adjacency matrix and gets num vertices
	int *matrix, n_vertices = igraph_vcount(&graph);
	long int vsize;
	
	//vertice adj vectors, intialized to size 0
	igraph_vector_t vec;
	igraph_vector_init(&vec, 0);
	
	//initializes matrix and sets to zero
	matrix = (int *)malloc(n_vertices*n_vertices*sizeof(int));
	memset(matrix, 0, sizeof(int)*n_vertices*n_vertices);

	//builds adj matrix
	for(int i = 0; i < n_vertices; i++) {
		
		//gets vertice i's neighbors and number of adjacencies
		igraph_neighbors(&graph, &vec, i, OUTALL);
		vsize = igraph_vector_size(&vec);

		//puts ones in the adj matrix where they belong
		for(int j = 0; j < vsize; j++) {

			matrix[i*n_vertices + (int)VECTOR(vec)[j]] = 1;
		}
	}

	//CUDA SHIT
	int hsize = 64;
	int *hist, *d_hist;
	hist = (int*)malloc(sizeof(int)*hsize);
	cudaMalloc((void**)&d_hist, sizeof(int)*hsize);

	//creates 2 adjacency matrix graphs for device
	int *d_matrix, *d_result;
	cudaMalloc((void**)&d_matrix, sizeof(int)*n_vertices*n_vertices);
	cudaMalloc((void**)&d_result, sizeof(int)*n_vertices*n_vertices);
	
	//copys adj matrix to device and sets device hist and result to zero
	cudaMemcpy(d_matrix, matrix, sizeof(int)*n_vertices*n_vertices, cudaMemcpyHostToDevice);
	cudaMemset(d_result, 0, sizeof(int)*n_vertices*n_vertices);
	cudaMemset(d_hist, 0, sizeof(int)*hsize);
	//memset(hist, 0, sizeof(int)*hsize);

	//kernel execution time crap
	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//kernels for lcm and hist
	Naive<<<n_vertices, 1024>>>(d_matrix, d_result, n_vertices);
	Naive_Hist<<<n_vertices, 1024>>>(d_result, d_hist, n_vertices);
	
	//copies hist back to host
	checkCudaError(cudaMemcpy(hist, d_hist, sizeof(int)*hsize, cudaMemcpyDeviceToHost), "D_HIST TO HOST");
	
	//kernel execution stop
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//prints gpu histogram
	printf("\nGPU HISTOGRAM\n");
	for(int i = 1; i < hsize; i++) {
		if ((hist[i] / i) > 0)
			printf("%d    %d\n", i, (hist[i] / i));
	}

	//prints kernel running time
	printf("\n******** Total Running Time of Kernel = %0.5f ms *******\n", elapsedTime);
	printf("\n******** Total Running Time of Kernel = %0.5f sec *******\n", elapsedTime/1000);

	//frees all the shit
	free(matrix);
	free(hist);
	cudaFree(d_matrix);
	cudaFree(d_result);
	cudaFree(d_hist);
}

//qsort compare function
int compare(const void* a, const void* b) {
	return ( *(int*)a - *(int*)b );
}

//naive cpu version, slow and takes a shit load of host memory
//uses adjacency matrix on cpu
void LCM_cpu_baseline(igraph_t &graph) {

	//gets num vertices and allocates, sets to zero adj matrix
	int n_vertices = igraph_vcount(&graph), vsize;
	int *matrix = (int *)malloc(n_vertices*n_vertices*sizeof(int));
	memset(matrix, 0, sizeof(int)*n_vertices*n_vertices);

	//graph vector and initializes it to zero
	igraph_vector_t vec;
	igraph_vector_init(&vec, 0);

	//builds adj matrix
	for(int i = 0; i < n_vertices; i++) {

		//grabs neighbors and size
		igraph_neighbors(&graph, &vec, i, OUTALL);
		vsize = igraph_vector_size(&vec);

		//adds ones where its adjacent
		for(int j = 0; j < vsize; j++) {

			matrix[i*n_vertices + (int)VECTOR(vec)[j]] = 1;
		}
	}

	//result adj matrix set to zero
	int *result = (int *)malloc(n_vertices*n_vertices*sizeof(int));
	memset(result, 0, sizeof(int)*n_vertices*n_vertices);
	int cval;

	//multiplies it against itself
	for(int i = 0; i < n_vertices; i++) {

		for(int j = i+1; j < n_vertices; j++) {

			cval = 0;

			for(int k = 0; k < n_vertices; k++)
				cval += matrix[i*n_vertices + k] * matrix[k*n_vertices + j];

			//enters val and transposes
			result[i*n_vertices + j] = cval;
			result[j*n_vertices + i] = cval;
		}

		//sorts the vertice/row
		qsort(&result[i*n_vertices], n_vertices + 0, sizeof(int), compare);
	}

	//multiplies it against itself, REALL SLOW CODE LOL
	// int *result = (int *)malloc(n_vertices*n_vertices*sizeof(int));
	// memset(result, 0, sizeof(int)*n_vertices*n_vertices);
	// int cval;

	// for(int i = 0; i < n_vertices; i++) {

	// 	for(int j = 0; j < n_vertices; j++) {

	// 		cval = 0;

	// 		for(int k = 0; k < n_vertices; k++) {

	// 			cval += matrix[i*n_vertices + k] * matrix[k*n_vertices + j];
	// 		}

	// 		result[i*n_vertices + j] = cval;
	// 	}

	// 	qsort(&result[i*n_vertices], n_vertices +1, sizeof(int), compare);
	// }

	//histogram
	bool equal;
	int count, countMax = -1;
	int *hist = (int*)malloc(sizeof(int) * n_vertices);
	memset(hist, 0, sizeof(int)*n_vertices);

	for(int i = 0; i < n_vertices; i++) {

		count = 0;

		for(int j = 0; j < n_vertices; j++) {

			equal = false;

			for(int k = 0; k < n_vertices; k++) {

				if(result[i*n_vertices + k] == result[j*n_vertices + k])
					equal = true;
				else {
					equal = false;
					break;
				}
			}

			if(equal)
				++count;
		}
		if(countMax < count)
				countMax = count;

		++hist[count];
	}

	//prints results
	printf("\nCPU Naive Histogram\n");
	for(int i = 1; i <= countMax; i++) {
		if ((long) (hist[i] / i) > 0)
			printf("%d    %ld\n", i, (long) (hist[i] / i));
	}

	//frees shit
	free(matrix);
	free(result);
	free(hist);
}

void linkage_covariance(igraph_t &graph) {

	//gets number of vertices
	int n_vertices = igraph_vcount(&graph);

	//neighbor vectors and init, holds adj vertices
	igraph_vector_t neisVec1, neisVec2, compVec;
	igraph_vector_init(&neisVec1, 1);
	igraph_vector_init(&neisVec2, 1);
	igraph_vector_init(&compVec, 1);

	//jagged 2d array holding lcm
	igraph_vector_t arrVec[n_vertices];
	
	//initializes all the array of vectors to 0 size
	for(int j = 0; j < n_vertices; j++)
		igraph_vector_init(&arrVec[j], 0);
					
	//finds similar vertices
	for(int i = 0; i < n_vertices; i++) {
		
		//grabs neighbors/adj vertices
		igraph_neighbors(&graph, &neisVec1, i, OUTALL);
		
		//checks similaries with neighbors
		for(int j = i+1; j < n_vertices; j++) {

			//gets neighbors of next vertice and compares similarities using set intersection
			igraph_neighbors(&graph, &neisVec2, j, OUTALL);
			igraph_vector_intersect_sorted(&neisVec1, &neisVec2, &compVec);

			//pushes back for vertex i and transposes to j
			if (igraph_vector_size(&compVec) > 0) {
				
				igraph_vector_push_back(&arrVec[i], igraph_vector_size(&compVec));
				igraph_vector_push_back(&arrVec[j], igraph_vector_size(&compVec));
			}
		}
	}

	//vars for the histogram
	long int *hist;
	hist = (long int*)malloc(sizeof(long int)*n_vertices);
	memset(hist, 0, sizeof(long int)*n_vertices);
	int count = 0, countMax = -1;

	//calculates the histogram
	for(int i = 0; i < n_vertices; i++) {
		
		//sets count to zero and sorts the vector
		count = 0;
		igraph_vector_sort(&arrVec[i]);

		//checks for equality
		for(int j = 0; j < n_vertices; j++) {
			
			//if they arent equal size, they arent equal
			if(igraph_vector_size(&arrVec[i]) != igraph_vector_size(&arrVec[j]))
				continue;

			//sorts other row we are comparing
			igraph_vector_sort(&arrVec[j]);
			
			//if vectors are equal, increments count
			if(igraph_vector_all_e(&arrVec[i], &arrVec[j]))				
				count++;
		}

		//keep track of max count
		if(countMax < count)
			countMax = count;

		//increments hist[count] where count is 
		//identical with all other vectors including itself, count should always be > 0
		hist[count]++;
	}

	//prints histogram
	printf("\nCPU Optimized Histogram\n");
	for(int i = 1; i <= countMax; i++) {
		if ((long) (hist[i] / i) > 0)
			printf("%d    %ld\n", i, (long) (hist[i] / i));
	}

	//frees memory
	free(hist);
}

//CUDA ERROR
void checkCudaError(cudaError_t e, const char* in) {
	if (e != cudaSuccess) {
		printf("CUDA Error: %s, %s \n", in, cudaGetErrorString(e));
		exit(EXIT_FAILURE);
	}
}

//TEST PREP & KERNEL
void TEST_PREP(igraph_t &graph) {
	
	//num vertices
	int n_vertices = igraph_vcount(&graph);

	//1D adj list graphs and sizes
	int *adj;
	int *adjsizes = (int*)malloc(sizeof(int)*n_vertices);
	int totalsize = 0;

	//vector for single vertices adj list
	igraph_vector_t neisVec;

	//gets each vertex's number of neighbors and total neighbors
	for(int i = 0; i < n_vertices; totalsize += adjsizes[i++]) {
		
		igraph_neighbors(&graph, &neisVec, i, OUTALL);
		adjsizes[i] = igraph_vector_size(&neisVec);
	}

	//creats jagged & flattened to 1D adj list	
	adj = (int*)malloc(sizeof(int)*totalsize);

	//creates 1d adj list
	for(int i = 0; i < n_vertices; i++) {

		//gets neighbors and number of neighbors
		igraph_neighbors(&graph, &neisVec, i, OUTALL);

		//loads in vertice i's adjancent neighbors
		for(int j = 0; j < adjsizes[i]; j++) {
			
			if(i == 0)
				adj[j] = (int)VECTOR(neisVec)[j];

			else
				adj[i*adjsizes[i-1] + j] = (int)VECTOR(neisVec)[j];
		}
	}

	//device vars
	int *d_adj, *d_lcm, *d_adjsizes;

	//mallocs and copys
	cudaMalloc((void**)&d_adj, sizeof(int)*totalsize);
	cudaMalloc((void**)&d_lcm, sizeof(int)*totalsize);
	cudaMalloc((void**)&d_adjsizes, sizeof(int)*n_vertices);

	//copys adj list to device and initializes lcm to zero
	cudaMemcpy(d_adj, adj, sizeof(int)*totalsize, cudaMemcpyHostToDevice);
	cudaMemset(d_lcm, 0, sizeof(int)*totalsize);
	cudaMemcpy(d_adjsizes, adjsizes, sizeof(int)*n_vertices, cudaMemcpyHostToDevice);
	
	//figures out threads per block
	int threads;
	if(n_vertices > 1024)
		threads = 1024;
	else
		threads = n_vertices;

	//kernel call
	TEST<<<n_vertices, threads>>>(d_adj, d_lcm, d_adjsizes, n_vertices);
}	

//kernal
__global__ void TEST(int* adj, int* lcm, int* sizes, int n) {

	int vertex = blockIdx.x;
	int vcomp = threadIdx.x;
	int cval;

	for(int i = vcomp; i < n; i += blockDim.x) {

		//for(int j = 0; j < )
	}

	//thrust::sort(thrust::seq, test, test + 13);
}
