#include <stdio.h>
#include <igraph/igraph.h>
#include <string.h>
//#include <stdlib.h>
//#include <iostream>
//cuda
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>

void linkage_covariance(igraph_t &graph);
igraph_neimode_t OUTALL;


__global__ void LCMKernel(igraph_t &graph, int n_vertices, int numThreads) 
{     	
	int i = threadIdx.x + blockDim.x * blockIdx.x;    
	printf("HI from thread %d\n", i);
}


int h_LCM(igraph_t &graph, int n_vertices, int numThreads) {
   //Define number of threads
   // int numThreads = 1024;
   dim3 DimGrid(ceil(n_vertices/numThreads), 1, 1);   
   if (n_vertices%numThreads) 
   {
		DimGrid.x++;   
   }
   dim3 DimBlock(numThreads, 1, 1);   
  
   long int histo[n_vertices];
   memset(histo, 0, sizeof(long int)*n_vertices);
   long int d_histogram[n_vertices];
   cudaMalloc((void **)&d_histogram, sizeof(long int)*n_vertices);
   cudaMemcpy(d_histogram, histo, sizeof(long int)*n_vertices, cudaMemcpyHostToDevice);

   //Call kernel function	
   LCMKernel<<<DimGrid,DimBlock>>>(graph, n_vertices, numThreads);
   
   //Copy computed value to host
   cudaMemcpy(histo, d_histogram, sizeof(long int)*n_vertices, cudaMemcpyDeviceToHost);
   return 0;
}

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

	int numThreads = 64;
	//function
	// linkage_covariance(graph);
	h_LCM(graph, n_vertices, numThreads);

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