#include <stdio.h>
#include <igraph/igraph.h>
#include <string.h>
//#include <stdlib.h>
//#include <iostream>
//cuda
#include <cuda_runtime.h>
#include <cuda.h>

void linkage_covariance(igraph_t &graph);

int main(int argc, char** argv) {

	//graph shit
	igraph_t graph;

	//opens graph file passed as 1st argument
	FILE *inputFile;
	inputFile = fopen(argv[1], "r");
	if(inputFile == NULL)
		printf("\nYou Done Fucked Up\n");

	//builds graph from file
	igraph_read_graph_ncol(&graph, inputFile, NULL, true, IGRAPH_ADD_WEIGHTS_NO, IGRAPH_DIRECTED);

	//function
	linkage_covariance(graph);

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

	//array of vectors
	igraph_vector_t arrVec[n_vertices];
	
	// printf("\nTEST\n");
	// igraph_neighbors(&graph, &neisVec1, 0, IGRAPH_OUT);
	// igraph_neighbors(&graph, &neisVec2, 0, IGRAPH_OUT);
	// igraph_vector_difference_sorted(&neisVec1, &neisVec2, &compVec);
	// printf("\ndifferenc = %ld\n", (long int)igraph_vector_size(&compVec));

	//finds similar vertices
	for(int i = 0; i < n_vertices; i++) {
		
		//initializes i'th vector and sets to 0
		igraph_vector_init(&arrVec[i], n_vertices);

		//gets vertex i's neighbors
		igraph_neighbors(&graph, &neisVec1, i, IGRAPH_OUT);
		//printf("\nvertex %d adjacents = %ld\n", i+1, igraph_vector_size(&neisVec));

		//inner loop
		for(int j = 0; j < n_vertices; j++) {

			if(j < i) {
				
				VECTOR(arrVec[i])[j] = VECTOR(arrVec[j])[i];
				continue;
			}

			if(i == j) {
				
				VECTOR(arrVec[i])[j] = 0;
				continue;
			}


			//gets j's neighbors and finds intersections of i and j
			igraph_neighbors(&graph, &neisVec2, j, IGRAPH_OUT);
			igraph_vector_intersect_sorted(&neisVec1, &neisVec2, &compVec);
			//printf("\nnumber in common v%d and v%d = %ld\n", i, j, igraph_vector_size(&compVec));

			//adds to array of vectors the number of similar neighbors
			VECTOR(arrVec[i])[j] = igraph_vector_size(&compVec);
			// if(i == 0)
			// 	printf("\nv0 and v%d in common = %ld\n", j, (long int)VECTOR(arrVec[i])[j]);
		}
	}

	long int histo[n_vertices];
	memset(histo, 0, sizeof(long int)*n_vertices);
	int count, countMax = -1;

	for(int i = 0; i < n_vertices; i++) {

		count = 0;

		for(int j = i+1; j < n_vertices; j++) {

			igraph_vector_difference_sorted(&arrVec[i], &arrVec[j], &compVec);

			if(igraph_vector_size(&compVec) == 0)
				++count;
		}

		if(countMax < count)
			countMax = count;

		++histo[count];
	}

	for(int i = 0; i < countMax; i++) {

		printf("%d    %ld\n", i+1, histo[i]);
	}
}