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

int main(int argc, char** argv) {
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

	//array of vectors
	igraph_vector_t arrVec[n_vertices];
	
	// printf("\nTEST\n");
	// igraph_neighbors(&graph, &neisVec1, 0, IGRAPH_OUT);
	// igraph_neighbors(&graph, &neisVec2, 0, IGRAPH_OUT);
	// igraph_vector_difference_sorted(&neisVec1, &neisVec2, &compVec);
	// printf("\ndifferenc = %ld\n", (long int)igraph_vector_size(&compVec));

	for(int j = 0; j < n_vertices; j++)
	{
		igraph_vector_init(&arrVec[j], 0);	
	}
					
	//finds similar vertices
	for(int i = 0; i < n_vertices; i++) {
		
		// //initializes i'th vector and sets to 0
		// igraph_vector_init(&arrVec[i], 0);

		//gets vertex i's neighbors
		igraph_neighbors(&graph, &neisVec1, i, IGRAPH_OUT);
		// printf("\nvertex %d adjacents = %ld\n", i+1, igraph_vector_size(&neisVec1));
		// int nonZero = 0;
		//inner loop
		for(int j = i+1; j < n_vertices; j++) {

			// if(j < i) {
				
			// 	VECTOR(arrVec[i])[j] = VECTOR(arrVec[j])[i];
			// 	continue;
			// }

			// if(i == j) {
				
			// 	VECTOR(arrVec[i])[j] = 0;
			// 	continue;
			// }


			//gets j's neighbors and finds intersections of i and j
			igraph_neighbors(&graph, &neisVec2, j, IGRAPH_OUT);
			igraph_vector_intersect_sorted(&neisVec1, &neisVec2, &compVec);
			// printf("\nnumber in common v%d and v%d = %ld\n", i, j, igraph_vector_size(&compVec));
			if (igraph_vector_size(&compVec) > 0)
			{
				// printf("\nnumber in common v%d and v%d = %ld\n", i, j, igraph_vector_size(&compVec));
				//adds to array of vectors the number of similar neighbors
				// VECTOR(arrVec[i])[nonZero] = igraph_vector_size(&compVec);
				// nonZero++;

				igraph_vector_push_back(&arrVec[i], igraph_vector_size(&compVec));
				igraph_vector_push_back(&arrVec[j], igraph_vector_size(&compVec));
				// printf("%d %d %ld", i, j, igraph_vector_size(&arrVec[j]));
				// igraph_vector_push_back(&arrVec[i], igraph_vector_size(&compVec));
				// printf("%d %d %ld", i, j, igraph_vector_size(&arrVec[j]));
				// if (igraph_vector_size(&arrVec[j]) == 0)
				// {
				// 	igraph_vector_init(&arrVec[j], 0);
				// }
				
				// if(i == 0)
				// 	printf("\nv0 and v%d in common = %ld\n", j, (long int)VECTOR(arrVec[i])[j]);
			}
		}
		// printf("%d - %d", i, nonZero);
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
			// igraph_vector_init(&compVec, 1);

			// printf("\n");
			igraph_vector_sort(&arrVec[j]);
			igraph_vector_difference_sorted(&arrVec[i], &arrVec[j], &compVec);
			
			// if ((i == 17 && j == 19) || (i == 19 && j == 17))
			// {
			// 	printf("%d - %d \n", i, j);
			// 	printf("%d:\n", i);
			// 	for(int k = 0; k< igraph_vector_size(&arrVec[i]); k++)
			// 	{
			// 		printf("%ld-", (long int)VECTOR(arrVec[i])[k]);
			// 	}
			// 	printf("\n%d:\n", j);
			// 	for(int k = 0; k< igraph_vector_size(&arrVec[j]); k++)
			// 	{
			// 		printf("%ld-", (long int)VECTOR(arrVec[j])[k]);
			// 	}
			// 	printf("\n-----------------\n");
			// }
				
			// if(igraph_vector_size(&compVec) == 0)
			if(igraph_vector_all_e(&arrVec[i], &arrVec[j]))
			{				
				// for(int k = 0; k< igraph_vector_size(&arrVec[j]); k++)
				// {
				// 	printf("%ld-", (long int)VECTOR(arrVec[j])[k]);
				// }
				// printf("\n%d == %d\n", i, j);
				count++;
			}
			// if(countMax < count)
			// countMax = count;
			// if (count == 1)
			// 	printf("\n%d - %d\n", i, count);
			// histo[count]++;
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