#include <stdio.h>
#include <igraph/igraph.h>
#include <string.h>
//#include <stdlib.h>
//#include <iostream>
//cuda
#include <cuda_runtime.h>
#include <cuda.h>
#include <sys/time.h>

igraph_neimode_t OUTALL;

void linkage_covariance(igraph_t &graph);

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
	
	//timing stuff
	struct timeval stop, start;
	gettimeofday(&start, NULL);

	//graph shit
	igraph_t graph;

	//opens graph file passed as 1st argument
	FILE *inputFile;
	inputFile = fopen(argv[1], "r");
	if(inputFile == NULL)
	{
		printf("Could not load input file...\n");
		return 1;
	}

	//builds graph from file
	igraph_read_graph_ncol(&graph, inputFile, NULL, true, IGRAPH_ADD_WEIGHTS_NO, IGRAPH_DIRECTED);

	//function
	linkage_covariance(graph);
	gettimeofday(&stop, NULL);
	printf("took %2f\n", (stop.tv_sec - start.tv_sec) * 1000.0f + (stop.tv_usec - start.tv_usec) / 1000.0f);
	return 0;
}

//function
void linkage_covariance(igraph_t &graph) {

	//gets number of vertices
	int n_vertices = igraph_vcount(&graph);
	int n_edges = igraph_ecount(&graph);
	//printf("\nn_vertices = %d n_edges = %d\n", n_vertices, n_edges);

	//neighbor vectors and init, holds adj vertices
	igraph_vector_t neisVec1, neisVec2, compVec;
	igraph_vector_init(&neisVec1, 1);
	igraph_vector_init(&neisVec2, 1);
	igraph_vector_init(&compVec, 1);

	//array of vectors and initializes them
	igraph_vector_t arrVec[n_vertices];
	for(int j = 0; j < n_vertices; j++)
	{
		igraph_vector_init(&arrVec[j], 0);	
	}
					
	//finds similar vertices
	for(int i = 0; i < n_vertices; i++) {

		//gets vertex i's neighbors
		igraph_neighbors(&graph, &neisVec1, i, OUTALL);

		//inner loop
		for(int j = i+1; j < n_vertices; j++) {

			//gets j's neighbors and finds intersections of i and j
			igraph_neighbors(&graph, &neisVec2, j, OUTALL);
			igraph_vector_intersect_sorted(&neisVec1, &neisVec2, &compVec);

			//if anything is similar
			if (igraph_vector_size(&compVec) > 0) {

				igraph_vector_push_back(&arrVec[i], igraph_vector_size(&compVec));
				igraph_vector_push_back(&arrVec[j], igraph_vector_size(&compVec));
			}
		}
	}

	//histogram and count vars
	long int histo[n_vertices];
	memset(histo, 0, sizeof(long int)*n_vertices);
	int count = 0, countMax = -1;

	//builds histogram
	for(int i = 0; i < n_vertices; i++) {
		
		//sets count to 0 and sorts vector i
		count = 0;
		igraph_vector_sort(&arrVec[i]);

		for(int j = 0; j < n_vertices; j++) {
			
			//if not the same, continues
			if(igraph_vector_size(&arrVec[i]) != igraph_vector_size(&arrVec[j]))
				continue;

			//
			igraph_vector_sort(&arrVec[j]);
				
			//increases count if vector i and j are the same
			if(igraph_vector_all_e(&arrVec[i], &arrVec[j]))
				count++;
		}

		//finds max count
		if(countMax < count)
			countMax = count;
		
		//increments histogram
		histo[count]++;
	}

	//prints histogram
	for(int i = 1; i <= countMax; i++) {
		if ((long) (histo[i] / i) > 0)
			printf("%d    %ld\n", i, (long) (histo[i] / i));
	}
}