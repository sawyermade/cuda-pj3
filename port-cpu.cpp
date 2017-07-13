#include <stdio.h>
#include <igraph/igraph.h>
//#include <iostream>
//cuda
#include <cuda_runtime.h>
#include <cuda.h>

int main(int argc, char* argv[]) {

	igraph_t graph;
	igraph_bool_t directed = false;

	// ifstream inputFile;
	// inputFile.open(argv[1], fstream::in);

	FILE *inputFile;
	inputFile = fopen(argv[1], "r");
	if(inputFile == NULL)
		printf("\nYou Done Fucked Up\n");

	//igraph_read_graph_edgelist(&graph, inputFile, 0, directed);
	igraph_read_graph_ncol(&graph, inputFile, NULL, false, IGRAPH_ADD_WEIGHTS_NO, false);

	//std::cout << "\nWorks Fuck You\n" << std::endl;

	return 0;
}