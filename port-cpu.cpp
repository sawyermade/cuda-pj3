#include <stdio.h>
#include <igraph/igraph.h>
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
	igraph_read_graph_ncol(&graph, inputFile, NULL, false, IGRAPH_ADD_WEIGHTS_NO, IGRAPH_DIRECTED);

	//function
	linkage_covariance(graph);

	return 0;
}

//function
void linkage_covariance(igraph_t &graph) {

	//gets number of vertices
	int n_vertices = igraph_vcount(&graph);
	int n_edges = igraph_ecount(&graph);
	printf("\nn_vertices = %d n_edges = %d\n", n_vertices, n_edges);

	//neighbor vector, holds adj vertices
	igraph_vector_t neisVec;
	igraph_vector_init(&neisVec, 1);
	
	// printf("\nTEST\n");

	//finds similar vertices
	for(int i = 0; i < n_vertices; i++) {
		
		igraph_neighbors(&graph, &neisVec, i, IGRAPH_OUT);
		printf("\nvertex %d adjacents = %ld\n", i+1, igraph_vector_size(&neisVec));
	}
}