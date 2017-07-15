#include <stdio.h>
#include <igraph/igraph.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>
#include <algorithm>
#include <stdlib.h>
#include <iterator>
#include <vector>

typedef struct {
  int *array;
  size_t used;
  size_t size;
} Array;

void initArray(Array *a, size_t initialSize) {
  a->array = (int *)malloc(initialSize * sizeof(int));
  a->used = 0;
  a->size = initialSize;
}

void insertArray(Array *a, int element) {
  // a->used is the number of used entries, because a->array[a->used++] updates a->used only *after* the array has been accessed.
  // Therefore a->used can go up to a->size 
  if (a->used == a->size) {
    a->size *= 2;
    a->array = (int *)realloc(a->array, a->size * sizeof(int));
  }
  a->array[a->used++] = element;
}

void freeArray(Array *a) {
  free(a->array);
  a->array = NULL;
  a->used = a->size = 0;
}

//GLOBAL VARS
igraph_neimode_t OUTALL;

//KERNELS
__global__ void TEST(int n, float* x, float* y);
__global__ void LCM_Kernel(igraph_t d_graph, igraph_vector_t *d_arrVec, int n_vertices, igraph_neimode_t OUTALL);


void TEST_PREP();
void LCM_Kernel_Prep(igraph_t &graph, int numThreads, igraph_neimode_t OUTALL);

//CUDA ERROR
void checkCudaError(cudaError_t e, const char* in);

//FUNCTIONS
void linkage_covariance(igraph_t &graph);
void LCM_CPU_Kernel(long int **adjList, int *sizeAdj, int n_vertices);
void LCM_CPU(igraph_t &graph, igraph_neimode_t OUTALL);

//main
int main(int argc, char** argv) {
	int numThreads = 32;
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

	// TEST_PREP();
	// LCM_Kernel_Prep(graph, numThreads, OUTALL);

	LCM_CPU(graph, OUTALL);

	//function
	// linkage_covariance(graph);

	gettimeofday(&stop, NULL);
	printf("took %2f\n", (stop.tv_sec - start.tv_sec) * 1000.0f + (stop.tv_usec - start.tv_usec) / 1000.0f);
	return 0;
}

int compare (const void * a, const void * b)
{
  return ( *(int*)a - *(int*)b );
}

int commonNeighbor(long int arr1[], long int arr2[], int m, int n)
{
  int i = 0, j = 0;
  int numCommon = 0;
  while (i < m && j < n)
  {
    if (arr1[i] < arr2[j])
      i++;
    else if (arr2[j] < arr1[i])
      j++;
    else /* if arr1[i] == arr2[j] */
    {
      // printf(" %d ", arr2[j++]);
      j++;
      i++;
      numCommon++;
    }
  }
  return numCommon;
}

int equalArray(Array a1, Array a2)
{
	if( a1.used != a2.used)
	{
		return 0;
	}
	for(int i = 0; i < a1.used; i++)
	{
		if(a1.array[i] != a2.array[i])
			return 0;
	}
	return 1;

}
void LCM_CPU(igraph_t &graph, igraph_neimode_t OUTALL)
{
	int n_vertices = igraph_vcount(&graph);
	igraph_adjlist_t al;
	igraph_adjlist_init(&graph, &al, OUTALL);
	igraph_adjlist_simplify(&al);

	long int **adjList;
	int *sizeAdj;

	adjList = (long int **) calloc(n_vertices, sizeof(long int *));
	sizeAdj = (int *) calloc(n_vertices, sizeof(int));
	for (int i = 0; i < n_vertices; i++) {
		igraph_vector_t *adjVec = igraph_adjlist_get(&al, i);

		adjList[i] = (long int *) calloc(igraph_vector_size(adjVec), sizeof(long int *));
		sizeAdj[i] = (int) igraph_vector_size(adjVec);
		for(int k = 0; k< igraph_vector_size(adjVec); k++)
		{
			adjList[i][k] = (long int) VECTOR(*adjVec)[k];
		}
	}

	for(int i = 0; i< n_vertices; i++)
	{
		qsort(adjList[i], sizeAdj[i], sizeof(long int), compare);
	}

	LCM_CPU_Kernel(adjList, sizeAdj, n_vertices);
}

void LCM_CPU_Kernel(long int **adjList, int *sizeAdj, int n_vertices)
{
	Array *lcmMatrix;
	lcmMatrix = (Array *) calloc(n_vertices, sizeof(Array));
	for(int i = 0; i < n_vertices; i++) {
		initArray(&lcmMatrix[i], sizeAdj[i]);
	}
	//finds similar vertices
	for(int i = 0; i < n_vertices; i++) {
		
		long int* neisVec1 = adjList[i];
		//inner loop
		for(int j = i+1; j < n_vertices; j++) {
			long int* neisVec2 = adjList[j];
			int compVec = commonNeighbor(neisVec1, neisVec2, sizeAdj[i], sizeAdj[j]);
			if (compVec > 0)
			{
				insertArray(&lcmMatrix[i], compVec);
				insertArray(&lcmMatrix[j], compVec);
			}
		}
	}
	printf("Finished Computing LCM\n");
	for(int i = 0; i < n_vertices; i++) {
		qsort(lcmMatrix[i].array, lcmMatrix[i].used, sizeof(int), compare);
		// printf("%d:\t", i);
		// for(int j=0;j < lcmMatrix[i].used; j++)
		// {
		// 	printf("%d-", lcmMatrix[i].array[j]);
		// }
		// printf("\n");
	}
	
	long int histo[n_vertices];
	memset(histo, 0, sizeof(long int)*n_vertices);
	int count = 0, countMax = -1;

	for(int i = 0; i < n_vertices; i++) {
		count = 0;
		for(int j = 0; j < n_vertices; j++) {
			if(lcmMatrix[i].used != lcmMatrix[j].used)
				continue;
			int eq = equalArray(lcmMatrix[i],lcmMatrix[j]);
			if(eq == 1)
			{				
				count++;
			}
		}

		if(countMax < count)
			countMax = count;
		histo[count]++;
	}

	for(int i = 1; i <= countMax; i++) {
		if ((long) (histo[i] / i) > 0)
			printf("%d    %ld\n", i, (long) (histo[i] / i));
	}

}

/*
void TEST_PREP() {
	int n = 100;
	float *x, *y, *d_x, *d_y;
	x = (float*)malloc(n*sizeof(float));
	y = (float*)malloc(n*sizeof(float));

	for(int i = 0; i < n; i++) {
		x[i] = 1.0f;
		y[i] = 2.0f;
	}

	checkCudaError(cudaMalloc((void**)&d_x, n*sizeof(float)), "Malloc Error");
	cudaMalloc((void**)&d_y, n*sizeof(float));
	cudaMemcpy(d_x, x, n*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, n*sizeof(float), cudaMemcpyHostToDevice);
	
	TEST<<<1,1>>>(n, d_x, d_y);
	checkCudaError(cudaGetLastError(), "Checking Last Error, Kernel Launch");
	
	cudaMemcpy(y, d_y, n*sizeof(float), cudaMemcpyDeviceToHost);

	for(int i = 0; i < n; i++)
		printf("%f\n", y[i]);
}

void LCM_Kernel_Prep(igraph_t &graph, int numThreads, igraph_neimode_t OUTALL)
{
	int n_vertices = igraph_vcount(&graph);
	igraph_t d_graph;
	long int histogram[n_vertices];
	// long int d_histogram[n_vertices];
	igraph_vector_t d_arrVec[n_vertices];

	memset(histogram, 0, sizeof(long int)*n_vertices);

	// cudaMalloc((void**)&d_histogram, n_vertices*sizeof(long int));
	cudaMalloc((void**)&d_graph, n_vertices*sizeof(igraph_vector_t));
	cudaMalloc((void**)&d_graph, sizeof(graph));
	
	// cudaMemcpy(d_histogram, histogram, n_vertices*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(&d_graph, &graph, sizeof(graph), cudaMemcpyHostToDevice);
	
	LCM_Kernel<<<ceil(n_vertices/numThreads), numThreads>>>(d_graph, d_arrVec, n_vertices, OUTALL);

	igraph_vector_t arrVec[n_vertices];
	cudaMemcpy(arrVec, &d_arrVec, n_vertices*sizeof(igraph_vector_t), cudaMemcpyDeviceToHost);
	// cudaMemcpy(histogram, d_histogram, n_vertices*sizeof(float), cudaMemcpyDeviceToHost);

	int count = 0, countMax = -1;
	igraph_vector_t compVec;

	for(int i = 0; i < n_vertices; i++) {
		count = 0;
		igraph_vector_sort(&arrVec[i]);
		printf("%d:\n", i);
		for(int k = 0; k< igraph_vector_size(&arrVec[i]); k++)
		{
			printf("%ld-", (long int)VECTOR(arrVec[i])[k]);
		}
		printf("\n");
		for(int j = 0; j < n_vertices; j++) {
			if(igraph_vector_size(&arrVec[i]) != igraph_vector_size(&arrVec[j]))
				continue;

			igraph_vector_sort(&arrVec[j]);
			igraph_vector_difference_sorted(&arrVec[i], &arrVec[j], &compVec);
			
			if(igraph_vector_all_e(&arrVec[i], &arrVec[j]))
			{				
				count++;
			}
		}

		if(countMax < count)
			countMax = count;
		// if (count == 1)
			// printf("\n%d - %d\n", i, count);
		histogram[count]++;
	}

	for(int i = 1; i <= countMax; i++) {
		if ((long) (histogram[i] / i) > 0)
			printf("%d    %ld\n", i, (long) (histogram[i] / i));
	}

	cudaFree(&d_graph);
	// cudaFree(d_histogram);
	cudaFree(&d_arrVec);
}


__global__ void TEST(int n, float* x, float* y) {

	for(int i = 0; i < n; i++)
		y[i] += x[i];
}


__global__ void LCM_Kernel(igraph_t d_graph, igraph_vector_t *d_arrVec, int n_vertices, igraph_neimode_t OUTALL){
	
	int i = threadIdx.x + blockDim.x * blockIdx.x;

	if(i < n_vertices){
		for(int j = 0; j < n_vertices; j++)
		{
			igraph_vector_init(&d_arrVec[j], 0);	
		}
		__syncthreads();

		igraph_vector_t neisVec1, neisVec2, compVec;
		igraph_vector_init(&neisVec1, 1);
		igraph_vector_init(&neisVec2, 1);
		igraph_vector_init(&compVec, 1);


		//finds similar vertices
		for(int i = 0; i < n_vertices; i++) {
			
			igraph_neighbors(&d_graph, &neisVec1, i, OUTALL);
			//inner loop
			for(int j = i+1; j < n_vertices; j++) {

				igraph_neighbors(&d_graph, &neisVec2, j, OUTALL);
				igraph_vector_intersect_sorted(&neisVec1, &neisVec2, &compVec);
				if (igraph_vector_size(&compVec) > 0)
				{
					igraph_vector_push_back(&d_arrVec[i], igraph_vector_size(&compVec));
					igraph_vector_push_back(&d_arrVec[j], igraph_vector_size(&compVec));
				}
			}
		}
	}
}
*/

//function
void linkage_covariance(igraph_t &graph) {

	//gets number of vertices
	int n_vertices = igraph_vcount(&graph);
	int n_edges = igraph_ecount(&graph);

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
			igraph_vector_difference_sorted(&arrVec[i], &arrVec[j], &compVec);
			
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
