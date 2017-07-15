#include <stdio.h>
#include <igraph/igraph.h>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>

//GLOBAL VARS
igraph_neimode_t OUTALL;

//KERNELS & PREP
__global__ void TEST(int n, float* x, float* y);
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
	int n_vertices = igraph_vcount(&graph);

	TEST_PREP();
	//function
	// linkage_covariance(graph);

	gettimeofday(&stop, NULL);
	printf("took %2f\n", (stop.tv_sec - start.tv_sec) * 1000.0f + (stop.tv_usec - start.tv_usec) / 1000.0f);
	return 0;
}

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
__global__ void TEST(int n, float* x, float* y) {

	for(int i = 0; i < n; i++)
		y[i] += x[i];
}
