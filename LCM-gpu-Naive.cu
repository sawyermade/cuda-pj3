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

//OPTIMIZATION 1 KERNELS & PREP
void OPT_1_PREP(igraph_t &graph);
__global__ void OPT_1(int* adj, int* lcm, int* sizes, int n);
__global__ void OPT_1_HIST(int* lcm, int* hist, int n);

//OPTIMIZATION 2 KERNELS & PREP
void OPT_2_PREP(igraph_t &graph);
__global__ void OPT_2_SIZES(int* adj, int* lcmsizes, int* sizes, int n);
__global__ void OPT_2_SIZES_SUM(int* lcmsizes, int n);
__global__ void OPT_2(int* adj, int* lcm, int* sizes, int* lcmsizes, int n);
__global__ void OPT_2_HIST(int* lcm, int* hist, int* lcmsizes, int n);

//OPTIMIZATION 3 KERNELS & PREP
void OPT_3_PREP(igraph_t &graph);
__global__ void OPT_3_SIZES(int* adj, int* lcmsizes, int* sizes, int n);
__global__ void OPT_3_SIZES_SUM(int* lcmsizes, int n);
__global__ void OPT_3(int* adj, int* lcm, int* sizes, int* lcmsizes, int n);
__global__ void OPT_3_HIST(int* lcm, int* hist, int* lcmsizes, int n);

//SATHYAS
int LCM_Kernel_Prep(igraph_t &graph, igraph_neimode_t OUTALL, int numThreads);
__global__ void LCM_Hist_Kernel(int *d_lcmMatrix, int n_vertices);
__global__ void Get_LCM_Kernel(int *d_adjList, int *d_sizeAdj, int *d_lcmMatrix, int *d_LCMSize, int n_vertices);
__global__ void Get_LCMSize_Kernel(int *d_adjList, int *d_sizeAdj, int *d_LCMSize, int n_vertices);

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
	//linkage_covariance(graph);
	gettimeofday(&stop, NULL);
	printf("CPU Optimized Running Time on %d Nodes: %2f ms\n", igraph_vcount(&graph), (stop.tv_sec - start.tv_sec) * 1000.0f + (stop.tv_usec - start.tv_usec) / 1000.0f);

	//gpu shit
	//Naive_Prep(graph);
	//OPT_1_PREP(graph);
	//OPT_2_PREP(graph);
	//OPT_3_PREP(graph);
	LCM_Kernel_Prep(graph, OUTALL, 128);
	
	
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
		//exit(EXIT_FAILURE);
	}
}

//TEST PREP & KERNEL
void OPT_1_PREP(igraph_t &graph) {

	//num vertices
	int n_vertices = igraph_vcount(&graph);

	//1D adj list graphs and sizes
	int *adj;
	int *adjsizes = (int*)malloc(sizeof(int)*(n_vertices + 1));

	//vector for single vertices adj list
	igraph_vector_t neisVec;
	igraph_vector_init(&neisVec, 0);

	//figures out threads per block
	int threads;
	if(n_vertices > 1024)
		threads = 1024;
	else
		threads = n_vertices;

	//gets each vertex's number of neighbors and total neighbors
	adjsizes[0] = 0;
	for(int i = 1; i <= n_vertices; i++) {

		igraph_neighbors(&graph, &neisVec, i-1, OUTALL);
		adjsizes[i] = igraph_vector_size(&neisVec) + adjsizes[i-1];

	}

	

	//creats jagged & flattened to 1D adj list	
	adj = (int*)malloc(sizeof(int)*adjsizes[n_vertices]);

	//creates 1d adj list
	for(int i = 0; i < n_vertices; i++) {

		//gets neighbors and number of neighbors
		igraph_neighbors(&graph, &neisVec, i, OUTALL);

		//loads in vertice i's adjancent neighbors
		//printf("\n%d: ", i);
		for(int j = 0; j < adjsizes[i+1] - adjsizes[i]; j++) {
			
			adj[adjsizes[i] + j] = (int)VECTOR(neisVec)[j];

			//printf("[%d, %d] ", adj[adjsizes[i] + j], (int)VECTOR(neisVec)[j]);
		}
	}



	//device vars
	int *d_adj, *d_lcm, *d_adjsizes, *d_hist;

	//histogram vars
	int *hist;
	hist = (int*)malloc(sizeof(int)*n_vertices);
	memset(hist, 0, sizeof(int)*n_vertices);

	//mallocs and copys
	checkCudaError(cudaMalloc((void**)&d_adj, sizeof(int)*adjsizes[n_vertices]), "Malloc d_adj");
	checkCudaError(cudaMalloc((void**)&d_adjsizes, sizeof(int)*(n_vertices+1)), "Malloc d_adjsizes");
	checkCudaError(cudaMalloc((void**)&d_lcm, sizeof(int)*n_vertices*n_vertices), "Malloc d_lcm");

	//copys adj list to device and initializes lcm to zero
	checkCudaError(cudaMemcpy(d_adj, adj, sizeof(int)*adjsizes[n_vertices], cudaMemcpyHostToDevice), "Memcpy d_adj");
	checkCudaError(cudaMemcpy(d_adjsizes, adjsizes, sizeof(int)*(n_vertices+1), cudaMemcpyHostToDevice), "Memcpy d_adjsizes");
	checkCudaError(cudaMemset(d_lcm, 0, sizeof(int)*n_vertices*n_vertices), "Memset d_lcm");

	//device histogram stuff
	checkCudaError(cudaMalloc((void**)&d_hist, sizeof(int)*n_vertices), "Malloc d_hist");
	checkCudaError(cudaMemset(d_hist, 0, sizeof(int)*n_vertices), "Memset d_hist");

	//SIZE OF SHIT
	//printf("\nSize(adj) =     %ld Bytes\nSize(adjsize) = %ld Bytes\nSize(hist) =    %ld Bytes\nSize(lcm) =     %ld Bytes", sizeof(int)*adjsizes[n_vertices], sizeof(int)*(n_vertices + 1), sizeof(int)*n_vertices, sizeof(int)*n_vertices*n_vertices);

	

	//kernel execution time crap
	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//kernel call
	OPT_1<<<n_vertices, threads>>>(d_adj, d_lcm, d_adjsizes, n_vertices);
	checkCudaError(cudaGetLastError(), "Checking Last Error, Test Kernel Launch");
	// printf("\nTEST\n");
	//cudaDeviceSynchronize();
	

	//DEBUG
	// int *lcm = (int*)malloc(sizeof(int)*n_vertices*n_vertices);
	// cudaMemcpy(lcm, d_lcm, sizeof(int)*n_vertices*n_vertices, cudaMemcpyDeviceToHost);
	// for(int i = 0; i < n_vertices; i++) {

	// 	printf("\nv%d: ", i);
	// 	for(int j = 0; j < n_vertices; j++) {

	// 		printf("%d ", lcm[i*n_vertices + j]);
	// 	}
	// 	printf("\n");
	// }
	// for(int i = 0; i < n_vertices; i++) {

	// 	int count = 0;

	// 	for(int j = 0; j < n_vertices; j++) {

	// 		bool equal = false;

	// 		for(int k = 0; k < n_vertices; k++) {

	// 			if(lcm[i*n_vertices + k] == lcm[j*n_vertices + k])
	// 				equal = true;
	// 			else {
	// 				equal = false;
	// 				break;
	// 			}
	// 		}

	// 		if(equal)
	// 			++count;
	// 	}
	// 	// if(countMax < count)
	// 	// 		countMax = count;

	// 	++hist[count];
	// }


	// histogram shit
	
	OPT_1_HIST<<<n_vertices, threads>>>(d_lcm, d_hist, n_vertices);

	//kernel execution stop
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	checkCudaError(cudaGetLastError(), "Checking Last Error, Test Hist Launch");
	checkCudaError(cudaMemcpy(hist, d_hist, sizeof(int)*n_vertices, cudaMemcpyDeviceToHost), "Memcpy d_hist to host");

	//prints gpu histogram
	printf("\nGPU TEST HISTOGRAM\n");
	for(int i = 1; i < n_vertices; i++) {
		if ((hist[i] / i) > 0)
			printf("%d    %d\n", i, (hist[i] / i));
	}

	//prints kernel running time
	printf("\n******** Total Running Time of Kernel = %0.5f ms *******\n", elapsedTime);
	printf("\n******** Total Running Time of Kernel = %0.5f sec *******\n", elapsedTime/1000);

	//frees everything
	cudaFree(d_hist);
	cudaFree(d_lcm);
	cudaFree(d_adj);
	cudaFree(d_adjsizes);
	free(hist);
	free(adj);
	free(adjsizes);
}	

//OPTIMIZATION 1
__global__ void OPT_1(int* adj, int* lcm, int* sizes, int n) {
	
	int vertex = blockIdx.x;
	int vcomp = threadIdx.x;
	int cval;

	if(vertex < n && vcomp < n)
	for(int i = vcomp; i < n; i += blockDim.x) {

		if(vertex == i) {
			lcm[vertex*n + i] = 0;
			continue;
		}

		//resets count
		cval = 0;

		//for loop that goes through vertex neighbors
		for(int j = 0; j < sizes[vertex + 1] - sizes[vertex]; j++) {

			//loop compares to other vertex i/vcomp
			for(int k = 0; k < sizes[i+1] - sizes[i]; k++) {

				if(adj[sizes[vertex] + j] == adj[sizes[i] + k]) {

					++cval;
					break;
				}
			}
		}

		//puts in lcm
		lcm[vertex*n + i] = cval;
	}

	//sorts vertex lcm once block is done
	__syncthreads();
	if(vcomp == 0 && vertex < n)
		thrust::sort(thrust::device, &lcm[vertex*n], &lcm[vertex*n] + n);
}

__global__ void OPT_1_HIST(int* lcm, int* hist, int n) {

	//
	int vertex = blockIdx.x;
	int vcomp = threadIdx.x;
	bool equal;
	
	//
	__shared__ int cval;

	//
	if(vcomp == 0)
		cval = 0;
	__syncthreads();

	//
	if(vertex < n && vcomp < n)
	for(int i = vcomp; i < n; i += blockDim.x) {

		if(vertex == i) {
			atomicAdd(&cval, 1);
			continue;
		}
		
		equal = false;

		for(int j = 0; j < n; j++) {

			if(lcm[vertex*n + j] == lcm[i*n + j])
				equal = true;
			
			else {
				equal = false;
				break;
			}
		}

		if(equal)
			atomicAdd(&cval, 1);
	}

	__syncthreads();
	if(vertex < n && vcomp == 0 && cval > 0) {
		atomicAdd(&hist[cval], 1);
		//printf("\nv%d: %d\n", vertex, cval);
	}
}

//OPTIMIZATION 2 KERNELS & PREP
void OPT_3_PREP(igraph_t &graph) {

	//num vertices
	int n_vertices = igraph_vcount(&graph);

	//1D adj list graphs and sizes
	int *adj;
	int *adjsizes = (int*)malloc(sizeof(int)*(n_vertices + 1));
	int lcmsizes;

	//vector for single vertices adj list
	igraph_vector_t neisVec;
	igraph_vector_init(&neisVec, 0);

	//adj list shit
	igraph_adjlist_t al;
    igraph_adjlist_init(&graph, &al, OUTALL);
    igraph_adjlist_simplify(&al);
    igraph_vector_int_t *adjVec;

	//figures out threads per block
	int threads_max = 128;
	int threads;
	if(n_vertices > threads_max)
		threads = threads_max;
	else
		threads = n_vertices;

	//histogram vars
	int *hist;
	hist = (int*)malloc(sizeof(int)*n_vertices);
	memset(hist, 0, sizeof(int)*n_vertices);

	//gets each vertex's number of neighbors and total neighbors
	adjsizes[0] = 0;
	for(int i = 1; i <= n_vertices; i++) {

		// igraph_neighbors(&graph, &neisVec, i-1, OUTALL);
		// adjsizes[i] = igraph_vector_size(&neisVec) + adjsizes[i-1];

		adjVec = igraph_adjlist_get(&al, i-1);
		adjsizes[i] = igraph_vector_int_size(adjVec) + adjsizes[i-1];

	}

	//creats jagged & flattened to 1D adj list	
	adj = (int*)malloc(sizeof(int)*adjsizes[n_vertices]);

	//creates 1d adj list
	for(int i = 0; i < n_vertices; i++) {

		//gets neighbors and number of neighbors
		//igraph_neighbors(&graph, &neisVec, i, OUTALL);

		//loads in vertice i's adjancent neighbors
		// for(int j = 0; j < adjsizes[i+1] - adjsizes[i]; j++)
		// 	adj[adjsizes[i] + j] = (int)VECTOR(neisVec)[j];

		adjVec = igraph_adjlist_get(&al, i);

		for(int j = 0; j < adjsizes[i+1] - adjsizes[i]; j++)
			adj[adjsizes[i] + j] = (int)VECTOR(*adjVec)[j];
	}

	//device vars
	int *d_adj, *d_lcm, *d_adjsizes, *d_lcmsizes, *d_hist, *d_lcm_max;

	//kernel execution time crap
	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//mallocs device shit
	checkCudaError(cudaMalloc((void**)&d_adj, sizeof(int)*adjsizes[n_vertices]), "Malloc d_adj");
	checkCudaError(cudaMalloc((void**)&d_adjsizes, sizeof(int)*(n_vertices+1)), "Malloc d_adjsizes");
	checkCudaError(cudaMalloc((void**)&d_lcmsizes, sizeof(int)*(n_vertices+1)), "Malloc d_lcmsizes");
	checkCudaError(cudaMalloc((void**)&d_lcm_max, sizeof(int)), "Malloc d_lcm_max");

	//copys adj list to device and initializes lcm to zero
	checkCudaError(cudaMemcpy(d_adj, adj, sizeof(int)*adjsizes[n_vertices], cudaMemcpyHostToDevice), "Memcpy d_adj");
	checkCudaError(cudaMemcpy(d_adjsizes, adjsizes, sizeof(int)*(n_vertices+1), cudaMemcpyHostToDevice), "Memcpy d_adjsizes");
	checkCudaError(cudaMemset(d_lcmsizes, 0, sizeof(int)*(n_vertices+1)), "Memset d_lcmsizes");
	//checkCudaError(cudaMemset(&d_lcm_max, 0, sizeof(int)), "Memset d_lcmsizes");

	

	//SIZE OF SHIT
	//printf("\nSize(adj) =     %ld Bytes\nSize(adjsize) = %ld Bytes\nSize(hist) =    %ld Bytes\nSize(lcm) =     %ld Bytes", sizeof(int)*adjsizes[n_vertices], sizeof(int)*(n_vertices + 1), sizeof(int)*n_vertices, sizeof(int)*n_vertices*n_vertices);

	//lcm sizes kernel
	OPT_3_SIZES<<<n_vertices, threads>>>(d_adj, d_lcmsizes, d_adjsizes, n_vertices);
	checkCudaError(cudaGetLastError(), "Checking Last Error, OPT_3_SIZES Kernel Launch");
	//cudaDeviceSynchronize();
	OPT_3_SIZES_SUM<<<1,1>>>(d_lcmsizes, n_vertices);
	checkCudaError(cudaGetLastError(), "Checking Last Error, OPT_3_SIZES_SUM Kernel Launch");
	//cudaDeviceSynchronize();

	// //kernel execution stop
	// cudaEventRecord(stop, 0);
	// cudaEventSynchronize(start);
	// cudaEventSynchronize(stop);
	// cudaEventElapsedTime(&elapsedTime, start, stop);
	// cudaEventDestroy(start);
	// cudaEventDestroy(stop);

	//creates lcm adj list shit
	checkCudaError(cudaMemcpy(&lcmsizes, &d_lcmsizes[n_vertices], sizeof(int), cudaMemcpyDeviceToHost), "Memcpy d_lcmsizes to lcmsizes");
	checkCudaError(cudaMalloc((void**)&d_lcm, sizeof(int)*lcmsizes), "Malloc d_lcm");
	checkCudaError(cudaMemset(d_lcm, 0, sizeof(int)*lcmsizes), "Memset d_lcm");

	// //kernel execution time crap 2
	// float elapsedTime3;
	// cudaEvent_t start3, stop3;
	// cudaEventCreate(&start3);
	// cudaEventCreate(&stop3);
	// cudaEventRecord(start3, 0);

	//get lcm shit
	OPT_3<<<n_vertices, threads>>>(d_adj, d_lcm, d_adjsizes, d_lcmsizes, n_vertices);
	checkCudaError(cudaGetLastError(), "Checking Last Error, OPT_3 Kernel Launch");

	//DEBUG HIST
	// int *lcm = (int*)malloc(sizeof(int)*lcmsizes);
	// int *lsizes = (int*)malloc(sizeof(int)*(n_vertices+1));
	// cudaMemcpy(lcm, d_lcm, sizeof(int)*lcmsizes, cudaMemcpyDeviceToHost);
	// cudaMemcpy(lsizes, d_lcmsizes, sizeof(int)*(n_vertices+1), cudaMemcpyDeviceToHost);
	// for(int i = 0; i < n_vertices; i++) {

	// 	int count = 0;

	// 	for(int j = 0; j < n_vertices; j++) {

	// 		if(lsizes[i+1] - lsizes[i] != lsizes[j+1] - lsizes[j])
	// 			continue;
			
	// 		bool equal = false;

	// 		for(int k = 0; k < lsizes[i+1] - lsizes[i]; k++) {

	// 			if(lcm[lsizes[i] + k] == lcm[lsizes[j] + k])
	// 				equal = true;
	// 			else {
	// 				equal = false;
	// 				break;
	// 			}
	// 		}

	// 		if(equal)
	// 			++count;
	// 	}

	// 	++hist[count];
	// }
	// free(lcm);
	// free(lsizes);

	//histogram
	cudaFree(d_adj);
	cudaFree(d_adjsizes);
	checkCudaError(cudaMalloc((void**)&d_hist, sizeof(int)*n_vertices), "Malloc d_hist");
	checkCudaError(cudaMemset(d_hist, 0, sizeof(int)*n_vertices), "Memset d_hist");
	OPT_3_HIST<<<n_vertices, threads>>>(d_lcm, d_hist, d_lcmsizes, n_vertices);
	checkCudaError(cudaGetLastError(), "Checking Last Error, OPT_3_HIST Kernel Launch");
	checkCudaError(cudaMemcpy(hist, d_hist, sizeof(int)*n_vertices, cudaMemcpyDeviceToHost), "D_HIST TO HOST");

	//kernel execution stop
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//prints gpu histogram
	printf("\nGPU OPT 2 HISTOGRAM\n");
	for(int i = 1; i < n_vertices; i++) {
		if ((hist[i] / i) > 0)
			printf("%d    %d\n", i, (hist[i] / i));
	}

	//kernels total times
	printf("\n******** Processed %d Node Graph In %0.5f ms *******\n", n_vertices, elapsedTime);
	printf("\n******** Processed %d Node Graph In %0.5f sec *******\n", n_vertices, elapsedTime/1000);

	//frees all the shit
	free(adj);
	free(hist);
	free(adjsizes);
	cudaFree(d_hist);
	cudaFree(d_lcm);
	cudaFree(d_lcmsizes);
}

__global__ void OPT_3_SIZES(int* adj, int* lcmsizes, int* sizes, int n) {

	int vertex = blockIdx.x;
	int vcomp = threadIdx.x;
	int cval;

	if(vertex < n && vcomp < n)
	for(int i = vcomp; i < n; i += blockDim.x) {

		//skips to next vertex
		if(vertex == i) {
			continue;
		}

		//resets count
		cval = 0;

		//for loop that goes through vertex neighbors
		for(int j = 0; j < sizes[vertex + 1] - sizes[vertex]; j++) {

			//loop compares to other vertex i/vcomp
			for(int k = 0; k < sizes[i+1] - sizes[i]; k++) {

				if(adj[sizes[vertex] + j] == adj[sizes[i] + k]) {

					++cval;
					break;
				}
			}

			if(cval > 0) {
				atomicAdd(&lcmsizes[vertex + 1], 1);
				break;
			}
		}
	}
}

//
__global__ void OPT_3_SIZES_SUM(int* lcmsizes, int n) {
	
	for(int i = 0; i < n; i++)
		lcmsizes[i+1] += lcmsizes[i];
}

//
__global__ void OPT_3(int* adj, int* lcm, int* sizes, int* lcmsizes, int n) {

	int vertex = blockIdx.x;
	int vcomp = threadIdx.x;
	int cval;

	__shared__ int pos;

	if(vcomp == 0)
		pos = 0;
	__syncthreads();

	if(vertex < n && vcomp < n)
	for(int i = vcomp; i < n; i += blockDim.x) {

		if(vertex == i) {
			continue;
		}

		//resets count
		cval = 0;

		//for loop that goes through vertex neighbors
		for(int j = 0; j < sizes[vertex + 1] - sizes[vertex]; j++) {

			//loop compares to other vertex i/vcomp
			for(int k = 0; k < sizes[i+1] - sizes[i]; k++) {

				if(adj[sizes[vertex] + j] == adj[sizes[i] + k]) {

					++cval;
					break;
				}
			}
		}

		//copies to array
		if(cval > 0) {
			//__syncthreads();
			lcm[lcmsizes[vertex] + atomicAdd(&pos, 1)] = cval;
		}
	}

	//sorts vertex lcm once block is done
	__syncthreads();
	if(vcomp == 0 && vertex < n)
		thrust::sort(thrust::device, &lcm[lcmsizes[vertex]], &lcm[lcmsizes[vertex+1]]);
}

//
__global__ void OPT_3_HIST(int* lcm, int* hist, int* lcmsizes, int n) {

	//
	int vertex = blockIdx.x;
	int vcomp = threadIdx.x;
	bool equal;
	
	//
	__shared__ int cval;

	//
	if(vcomp == 0)
		cval = 0;
	__syncthreads();

	//
	if(vertex < n && vcomp < n)
	for(int i = vcomp; i < n; i += blockDim.x) {

		if(vertex == i) {
			atomicAdd(&cval, 1);
			continue;
		}

		if((lcmsizes[vertex+1] - lcmsizes[vertex]) != (lcmsizes[i+1] - lcmsizes[i]))
			continue;
		
		equal = false;

		for(int j = 0; j < lcmsizes[vertex+1] - lcmsizes[vertex]; j++) {

			if(lcm[lcmsizes[vertex] + j] == lcm[lcmsizes[i] + j])
				equal = true;
			
			else {
				equal = false;
				break;
			}
		}

		if(equal)
			atomicAdd(&cval, 1);
	}

	__syncthreads();
	if(vertex < n && vcomp == 0 && cval > 0) {
		atomicAdd(&hist[cval], 1);
		//printf("\nv%d: %d\n", vertex, cval);
	}
}

//OPTIMIZATION 2 KERNELS & PREP
void OPT_2_PREP(igraph_t &graph) {

	//num vertices
	int n_vertices = igraph_vcount(&graph);

	//1D adj list graphs and sizes
	int *adj;
	int *adjsizes = (int*)malloc(sizeof(int)*(n_vertices + 1));
	int lcmsizes;

	//adj list shit
	igraph_adjlist_t al;
    igraph_adjlist_init(&graph, &al, OUTALL);
    igraph_adjlist_simplify(&al);
    igraph_vector_int_t *adjVec;

	//figures out threads per block
	int threads_max = 128;
	int threads;
	if(n_vertices > threads_max)
		threads = threads_max;
	else
		threads = n_vertices;

	//histogram vars
	int *hist;
	hist = (int*)malloc(sizeof(int)*n_vertices);
	memset(hist, 0, sizeof(int)*n_vertices);

	//gets each vertex's number of neighbors and total neighbors
	adjsizes[0] = 0;
	for(int i = 1; i <= n_vertices; i++) {

		adjVec = igraph_adjlist_get(&al, i-1);
		adjsizes[i] = igraph_vector_int_size(adjVec) + adjsizes[i-1];

	}

	//creats jagged & flattened to 1D adj list	
	adj = (int*)malloc(sizeof(int)*adjsizes[n_vertices]);

	//creates 1d adj list
	for(int i = 0; i < n_vertices; i++) {

		adjVec = igraph_adjlist_get(&al, i);

		for(int j = 0; j < adjsizes[i+1] - adjsizes[i]; j++)
			adj[adjsizes[i] + j] = (int)VECTOR(*adjVec)[j];
	}

	//device vars
	int *d_adj, *d_lcm, *d_adjsizes, *d_lcmsizes, *d_hist, *d_lcm_max;

	//kernel execution time crap
	float elapsedTime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//mallocs device shit
	checkCudaError(cudaMalloc((void**)&d_adj, sizeof(int)*adjsizes[n_vertices]), "Malloc d_adj");
	checkCudaError(cudaMalloc((void**)&d_adjsizes, sizeof(int)*(n_vertices+1)), "Malloc d_adjsizes");
	checkCudaError(cudaMalloc((void**)&d_lcmsizes, sizeof(int)*(n_vertices+1)), "Malloc d_lcmsizes");
	checkCudaError(cudaMalloc((void**)&d_lcm_max, sizeof(int)), "Malloc d_lcm_max");

	//copys adj list to device and initializes lcm to zero
	checkCudaError(cudaMemcpy(d_adj, adj, sizeof(int)*adjsizes[n_vertices], cudaMemcpyHostToDevice), "Memcpy d_adj");
	checkCudaError(cudaMemcpy(d_adjsizes, adjsizes, sizeof(int)*(n_vertices+1), cudaMemcpyHostToDevice), "Memcpy d_adjsizes");
	checkCudaError(cudaMemset(d_lcmsizes, 0, sizeof(int)*(n_vertices+1)), "Memset d_lcmsizes");
	//checkCudaError(cudaMemset(&d_lcm_max, 0, sizeof(int)), "Memset d_lcmsizes");

	

	//SIZE OF SHIT
	//printf("\nSize(adj) =     %ld Bytes\nSize(adjsize) = %ld Bytes\nSize(hist) =    %ld Bytes\nSize(lcm) =     %ld Bytes", sizeof(int)*adjsizes[n_vertices], sizeof(int)*(n_vertices + 1), sizeof(int)*n_vertices, sizeof(int)*n_vertices*n_vertices);

	//lcm sizes kernel
	OPT_3_SIZES<<<n_vertices, threads>>>(d_adj, d_lcmsizes, d_adjsizes, n_vertices);
	checkCudaError(cudaGetLastError(), "Checking Last Error, OPT_3_SIZES Kernel Launch");
	//cudaDeviceSynchronize();
	OPT_3_SIZES_SUM<<<1,1>>>(d_lcmsizes, n_vertices);
	checkCudaError(cudaGetLastError(), "Checking Last Error, OPT_3_SIZES_SUM Kernel Launch");
	//cudaDeviceSynchronize();

	// //kernel execution stop
	// cudaEventRecord(stop, 0);
	// cudaEventSynchronize(start);
	// cudaEventSynchronize(stop);
	// cudaEventElapsedTime(&elapsedTime, start, stop);
	// cudaEventDestroy(start);
	// cudaEventDestroy(stop);

	//creates lcm adj list shit
	checkCudaError(cudaMemcpy(&lcmsizes, &d_lcmsizes[n_vertices], sizeof(int), cudaMemcpyDeviceToHost), "Memcpy d_lcmsizes to lcmsizes");
	checkCudaError(cudaMalloc((void**)&d_lcm, sizeof(int)*lcmsizes), "Malloc d_lcm");
	checkCudaError(cudaMemset(d_lcm, 0, sizeof(int)*lcmsizes), "Memset d_lcm");

	// //kernel execution time crap 2
	// float elapsedTime3;
	// cudaEvent_t start3, stop3;
	// cudaEventCreate(&start3);
	// cudaEventCreate(&stop3);
	// cudaEventRecord(start3, 0);

	//get lcm shit
	int alt_threads = 128;
	OPT_3<<<ceil((float)n_vertices/alt_threads), alt_threads>>>(d_adj, d_lcm, d_adjsizes, d_lcmsizes, n_vertices);
	checkCudaError(cudaGetLastError(), "Checking Last Error, OPT_3 Kernel Launch");

	//DEBUG HIST
	// int *lcm = (int*)malloc(sizeof(int)*lcmsizes);
	// int *lsizes = (int*)malloc(sizeof(int)*(n_vertices+1));
	// cudaMemcpy(lcm, d_lcm, sizeof(int)*lcmsizes, cudaMemcpyDeviceToHost);
	// cudaMemcpy(lsizes, d_lcmsizes, sizeof(int)*(n_vertices+1), cudaMemcpyDeviceToHost);
	// for(int i = 0; i < n_vertices; i++) {

	// 	int count = 0;

	// 	for(int j = 0; j < n_vertices; j++) {

	// 		if(lsizes[i+1] - lsizes[i] != lsizes[j+1] - lsizes[j])
	// 			continue;
			
	// 		bool equal = false;

	// 		for(int k = 0; k < lsizes[i+1] - lsizes[i]; k++) {

	// 			if(lcm[lsizes[i] + k] == lcm[lsizes[j] + k])
	// 				equal = true;
	// 			else {
	// 				equal = false;
	// 				break;
	// 			}
	// 		}

	// 		if(equal)
	// 			++count;
	// 	}

	// 	++hist[count];
	// }
	// free(lcm);
	// free(lsizes);

	//histogram
	cudaFree(d_adj);
	cudaFree(d_adjsizes);
	checkCudaError(cudaMalloc((void**)&d_hist, sizeof(int)*n_vertices), "Malloc d_hist");
	checkCudaError(cudaMemset(d_hist, 0, sizeof(int)*n_vertices), "Memset d_hist");
	OPT_3_HIST<<<n_vertices, threads>>>(d_lcm, d_hist, d_lcmsizes, n_vertices);
	checkCudaError(cudaGetLastError(), "Checking Last Error, OPT_3_HIST Kernel Launch");
	checkCudaError(cudaMemcpy(hist, d_hist, sizeof(int)*n_vertices, cudaMemcpyDeviceToHost), "D_HIST TO HOST");

	//kernel execution stop
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(start);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//prints gpu histogram
	printf("\nGPU OPT 2 HISTOGRAM\n");
	for(int i = 1; i < n_vertices; i++) {
		if ((hist[i] / i) > 0)
			printf("%d    %d\n", i, (hist[i] / i));
	}

	//kernels total times
	printf("\n******** Processed %d Node Graph In %0.5f ms *******\n", n_vertices, elapsedTime);
	printf("\n******** Processed %d Node Graph In %0.5f sec *******\n", n_vertices, elapsedTime/1000);

	//frees all the shit
	free(adj);
	free(hist);
	free(adjsizes);
	cudaFree(d_hist);
	cudaFree(d_lcm);
	cudaFree(d_lcmsizes);
}

//
__global__ void OPT_2_SIZES(int* adj, int* lcmsizes, int* sizes, int n) {

	int vertex = blockIdx.x;
	int vcomp = threadIdx.x;
	int cval;

	if(vertex < n && vcomp < n)
	for(int i = vcomp; i < n; i += blockDim.x) {

		//skips to next vertex
		if(vertex == i) {
			continue;
		}

		//resets count
		cval = 0;

		//for loop that goes through vertex neighbors
		for(int j = 0; j < sizes[vertex + 1] - sizes[vertex]; j++) {

			//loop compares to other vertex i/vcomp
			for(int k = 0; k < sizes[i+1] - sizes[i]; k++) {

				if(adj[sizes[vertex] + j] == adj[sizes[i] + k]) {

					++cval;
					break;
				}
			}

			if(cval > 0) {
				atomicAdd(&lcmsizes[vertex + 1], 1);
				break;
			}
		}
	}
}

//
__global__ void OPT_2_SIZES_SUM(int* lcmsizes, int n) {

	for(int i = 0; i < n; i++)
		lcmsizes[i+1] += lcmsizes[i];
}

//
__global__ void OPT_2(int* adj, int* lcm, int* sizes, int* lcmsizes, int n) {

	int vertex = threadIdx.x + blockIdx.x*blockDim.x;
	int cval;
	int pos = 0;

	if(vertex < n)
	for(int i = 0; i < n; i++) {

		if(vertex == i) {
			continue;
		}

		//resets count
		cval = 0;

		//for loop that goes through vertex neighbors
		for(int j = 0; j < sizes[vertex + 1] - sizes[vertex]; j++) {

			//loop compares to other vertex i/vcomp
			for(int k = 0; k < sizes[i+1] - sizes[i]; k++) {

				if(adj[sizes[vertex] + j] == adj[sizes[i] + k]) {

					++cval;
					break;
				}
			}
		}

		//copies to array
		if(cval > 0) {
			atomicAdd(&lcm[lcmsizes[vertex] + pos], cval);
			++pos;
		}
	}

	//sorts vertex lcm once block is done
	if(vertex < n)
		thrust::sort(thrust::device, &lcm[lcmsizes[vertex]], &lcm[lcmsizes[vertex+1]]);
}

//
__global__ void OPT_2_HIST(int* lcm, int* hist, int* lcmsizes, int n) {

	//
	int vertex = blockIdx.x;
	int vcomp = threadIdx.x;
	bool equal;
	
	//
	__shared__ int cval;

	//
	if(vcomp == 0)
		cval = 0;
	__syncthreads();

	//
	if(vertex < n && vcomp < n)
	for(int i = vcomp; i < n; i += blockDim.x) {

		if(vertex == i) {
			atomicAdd(&cval, 1);
			continue;
		}

		if((lcmsizes[vertex+1] - lcmsizes[vertex]) != (lcmsizes[i+1] - lcmsizes[i]))
			continue;
		
		equal = false;

		for(int j = 0; j < lcmsizes[vertex+1] - lcmsizes[vertex]; j++) {

			if(lcm[lcmsizes[vertex] + j] == lcm[lcmsizes[i] + j])
				equal = true;
			
			else {
				equal = false;
				break;
			}
		}

		if(equal)
			atomicAdd(&cval, 1);
	}

	__syncthreads();
	if(vertex < n && vcomp == 0 && cval > 0) {
		atomicAdd(&hist[cval], 1);
		//printf("\nv%d: %d\n", vertex, cval);
	}
}


__global__ void Get_LCMSize_Kernel(int *d_adjList, int *d_sizeAdj, int *d_LCMSize, int n_vertices)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;     
    if(i<n_vertices)
    {
        int indexUsed = 0;
        int iStart = 0, iEnd = 0;
        int k = 0;

        if(i > 0)
        {       
            k = d_sizeAdj[i-1];
        }

        iEnd = d_sizeAdj[i];

        __syncthreads();

        for(int j = 0; j < n_vertices; j++) {
            if(i==j)
                continue;
            iStart = k;
            int jStart = 0, jEnd = 0;

            if(j > 0)
                jStart = d_sizeAdj[j-1];
            jEnd = d_sizeAdj[j];
            
            int compVec = 0;

            while (iStart < iEnd && jStart < jEnd)
            {
                    if(d_adjList[iStart] < d_adjList[jStart])
                        iStart++;
                    else if (d_adjList[jStart] < d_adjList[iStart])
                        jStart++;
                    else // if arr1[i] == arr2[j] 
                    {
                        jStart++;
                        iStart++;
                        compVec++;
                        // break;
                    }
            }

            if (compVec > 0)
            {
                indexUsed++;
            }
        }
    
        __syncthreads();
        d_LCMSize[i] = indexUsed;
        // __syncthreads();
    
    }

}

__global__ void Get_LCM_Kernel(int *d_adjList, int *d_sizeAdj, int *d_lcmMatrix, int *d_LCMSize, int n_vertices)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;     
    if(i<n_vertices)
    {
        int indexUsed = 0, indexOffset = 0;
        int iStart = 0, iEnd = 0;
        int k = 0;

        if(i > 0)
        {       
            k = d_sizeAdj[i-1];
            indexOffset = d_LCMSize[i-1];
        }

        iEnd = d_sizeAdj[i];
        
        for(int j = indexOffset; j<iEnd; j++)
        {
            d_lcmMatrix[j] = 0;
        }

        __syncthreads();

        for(int j = 0; j < n_vertices; j++) {
            if(i==j)
                continue;
            iStart = k;
            int jStart = 0, jEnd = 0;

            if(j > 0)
                jStart = d_sizeAdj[j-1];
            jEnd = d_sizeAdj[j];
            
            int compVec = 0;

            while (iStart < iEnd && jStart < jEnd)
            {
                    if(d_adjList[iStart] < d_adjList[jStart])
                        iStart++;
                    else if (d_adjList[jStart] < d_adjList[iStart])
                        jStart++;
                    else // if arr1[i] == arr2[j] 
                    {
                        jStart++;
                        iStart++;
                        compVec++;
                    }
            }

            if (compVec > 0)
            {
                atomicAdd((int*)&d_lcmMatrix[indexUsed + indexOffset], compVec);
                // d_lcmMatrix[indexUsed + indexOffset] = compVec;
                indexUsed++;
            }
            // __syncthreads();
        }
    
        // __syncthreads();
        // d_LCMSize[i] = indexUsed;
        // __syncthreads();
    
    }

}


__global__ void LCM_Hist_Kernel(int *d_lcmMatrix, int n_vertices)
{
    
}

int LCM_Kernel_Prep(igraph_t &graph, igraph_neimode_t OUTALL, int numThreads)
{
    printf("Allocating Adjacency List\n");
    int n_vertices = igraph_vcount(&graph);
    igraph_adjlist_t al;
    igraph_adjlist_init(&graph, &al, OUTALL);
    igraph_adjlist_simplify(&al);

    int **adjList2D;
    int totalSize = 0;

    int *adjList, *d_adjList;
    int *sizeAdj, *d_sizeAdj;

    int *lcmMatrix, *d_lcmMatrix;

    int *d_LCMSize, *LCMSize, *LCMSize_Calc;
    
    adjList2D = (int **) calloc(n_vertices, sizeof(int *));
    sizeAdj = (int *) malloc(n_vertices * sizeof(int));
    LCMSize = (int *) malloc(n_vertices * sizeof(int));
    LCMSize_Calc = (int *) malloc(n_vertices * sizeof(int));
    memset(LCMSize, 0, n_vertices*sizeof(int));
    memset(LCMSize_Calc, 0, n_vertices*sizeof(int));
    printf("Computing Adjacency List - %d vertices...\n", n_vertices);

    for (int i = 0; i < n_vertices; i++) {
        igraph_vector_int_t *adjVec = igraph_adjlist_get(&al, i);

        // igraph_vector_t adjVec;
        // igraph_vector_init(&adjVec, 0);
        // igraph_neighbors(&graph, &adjVec, i, OUTALL);

        adjList2D[i] = (int *) malloc(igraph_vector_int_size(adjVec) * sizeof(int));
        sizeAdj[i] = (int) igraph_vector_int_size(adjVec);
        totalSize += sizeAdj[i];
        for(int k = 0; k< igraph_vector_int_size(adjVec); k++)
        {
            adjList2D[i][k] = (int) VECTOR(*adjVec)[k];
        }
    }

    for(int i = 0; i< n_vertices; i++)
    {
        qsort(adjList2D[i], sizeAdj[i], sizeof(int), compare);
    }
    
    adjList = (int *) malloc(totalSize * sizeof(int));
    int l = -1;
    for (int q = 0; q < n_vertices; q++)
    {
        for (int t = 0; t < sizeAdj[q]; t++)
        {
            l++;
            adjList[l] = adjList2D[q][t];
        }
    }
    for(int i = 0; i< n_vertices; i++)
    {
        free(adjList2D[i]);
        if(i>0)
        {
            sizeAdj[i] += sizeAdj[i - 1];
        }
    }
    
    free(adjList2D);
    // memset(LCMSize, 0, n_vertices*sizeof(int));
    printf("%d-%d\n", totalSize, sizeAdj[n_vertices-1]);
    printf("Got Adj List...\n Allocating on gpu mem...");
    checkCudaError(cudaMalloc((void**)&d_adjList, totalSize * sizeof(int)), "Malloc Error d_adjList");
    checkCudaError(cudaMalloc((void**)&d_sizeAdj, n_vertices * sizeof(int)), "Malloc Error d_sizeAdj");
    checkCudaError(cudaMalloc((void**)&d_LCMSize, n_vertices * sizeof(int)), "Malloc Error d_sizeAdj");

    cudaMemcpy(d_adjList, adjList, totalSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sizeAdj, sizeAdj, n_vertices * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_LCMSize, LCMSize_Calc, n_vertices * sizeof(int), cudaMemcpyHostToDevice);

    dim3 DimGrid(ceil(n_vertices/numThreads), 1, 1);   
    if (n_vertices%numThreads) 
    {
        DimGrid.x++;
    }

    dim3 DimBlock(numThreads, 1, 1);
    int totLCMSize = 0;
    printf("Launching Size Kernel...\n");
    Get_LCMSize_Kernel<<<DimGrid,DimBlock>>>(d_adjList, d_sizeAdj, d_LCMSize, n_vertices);
    cudaThreadSynchronize();
    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(), "Checking Last Error, Size Kernel Launch");
    cudaMemcpy(LCMSize_Calc, d_LCMSize, n_vertices * sizeof(int), cudaMemcpyDeviceToHost);
    
    for(int i = 0; i<n_vertices; i++)
    {
        totLCMSize += LCMSize_Calc[i];
        LCMSize[i] = LCMSize_Calc[i];
    }

    for(int i = 1; i<n_vertices; i++)
    {
        if(i>0)
            LCMSize[i] += LCMSize[i - 1];
    }
    printf("%d - %d\n", totalSize, totLCMSize);
    
    lcmMatrix = (int *) malloc(totLCMSize * sizeof(int));
    memset(lcmMatrix, 0, totLCMSize*sizeof(int));
    checkCudaError(cudaMalloc((void**)&d_lcmMatrix, totLCMSize * sizeof(int)), "Malloc Error d_lcmMatrix");
    cudaMemcpy(d_lcmMatrix, lcmMatrix, totLCMSize * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_LCMSize, LCMSize, n_vertices * sizeof(int), cudaMemcpyHostToDevice);
    printf("Launching LCM Kernel...\n");
    
    // LCM_Kernel<<<DimGrid,DimBlock>>>(d_adjList, d_sizeAdj, d_lcmMatrix, d_LCMSize, n_vertices);
    Get_LCM_Kernel<<<DimGrid,DimBlock>>>(d_adjList, d_sizeAdj, d_lcmMatrix, d_LCMSize, n_vertices);
    
	cudaThreadSynchronize();
    cudaDeviceSynchronize();
    checkCudaError(cudaGetLastError(), "Checking Last Error, Kernel Launch");
    printf("Copying to CPU Memory...\n");
    checkCudaError(cudaMemcpy(lcmMatrix, d_lcmMatrix, totLCMSize * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy Error d_lcmMatrix");
    // checkCudaError(cudaMemcpy(LCMSize, d_LCMSize, n_vertices * sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy Error LCMSize");
    
    cudaFree(d_lcmMatrix);
    cudaFree(d_LCMSize);
    cudaFree(d_adjList);
    cudaFree(d_sizeAdj);
    free(sizeAdj);
    free(adjList);

    printf("Allocating Histogram...\n");
    int *histo;
    histo = (int *) malloc(n_vertices * sizeof(int));
    memset(histo, 0, sizeof(int)*n_vertices);
    int count = 0, countMax = -1;
    // int *neisVec1, *neisVec2;
    // neisVec1 = (int *) malloc(1 * sizeof(int));
    // neisVec2 = (int *) malloc(1 * sizeof(int));
   
    printf("Sorting LCM...\n");
    // int totLCMSize1 = 0;

/*
    for(int i = 1; i< n_vertices; i++)
    {
    	LCMSize[i] += LCMSize[i-1];
    }
*/
    
	for(int i = 0; i< n_vertices; i++)
	{
		int offset = 0;
		if(i > 0)
		{
			offset = LCMSize[i - 1];
		}
		// printf("%d - %d\n", offset, LCMSize_Calc[i]);
		qsort(lcmMatrix + offset, LCMSize_Calc[i], sizeof(int), compare);
		// totLCMSize1 += LCMSize[i];
	}
	// for(int i = 0; i<LCMSize_Calc[4000]; i++)
	// 	printf("%d-", lcmMatrix[LCMSize[3999] + i]);
	printf("Computing Histogram...\n");
	// return 0;
    for(int i = 0; i< n_vertices; i++)
    {
        int iStart = 0;
        if(i>0)
            iStart = LCMSize[i - 1]; //Offset
        count = 0;

        for(int j = 0; j < n_vertices; j++) {
            if(LCMSize_Calc[i] != LCMSize_Calc[j])
                continue;
            
            int jStart = 0;
            
            if(j>0)
                jStart = LCMSize[j - 1]; //Offset
            
            int eq = 1;
            for(int k = 0; k < LCMSize_Calc[i]; k++)
            {
            	if(lcmMatrix[iStart + k] != lcmMatrix[jStart + k])
            	{
            		eq = 0;
            		break;
            	}
            }
            if(eq == 1)
            {               
                count++;
            }
        }

        if(countMax < count)
            countMax = count;
        histo[count]++;
    }
    printf("Finished Histogram...\n");
    for(int i = 1; i <= countMax; i++) {
        if ((long) (histo[i] / i) > 0)
            printf("%d    %d\n", i, (int) (histo[i] / i));
    }
    free(lcmMatrix);
    free(LCMSize_Calc);
    free(LCMSize);
    free(histo);
	return 0;
}