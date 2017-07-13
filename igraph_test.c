#include <mpi.h>
#include <igraph/igraph.h>
#define true 1
#define false 0

int main(void)
{
    igraph_integer_t diameter;
    igraph_t graph;
    igraph_bool_t namesBool = true;
    FILE *ifile;
    igraph_vs_t vs;
    ifile=fopen("../terrornet_edges", "r");
    igraph_read_graph_ncol(&graph, ifile, 
               NULL,
               namesBool, 
               IGRAPH_ADD_WEIGHTS_NO,
               IGRAPH_DIRECTED); 
    // igraph_write_graph_pajek(&g, ofile);
    fclose(ifile);
     igraph_diameter(&graph, &diameter, 0, 0, 0, IGRAPH_UNDIRECTED, 1);
     printf("Diameter of a random graph with average degree 5: %d\n",
             (int) diameter);
    int n_vertices = igraph_vcount(&graph);
    igraph_adjlist_t adjlist;

    printf("Num vertices: %d\n", n_vertices);

    igraph_adjlist_init(&graph, &adjlist, IGRAPH_OUT);
    igraph_vector_t *vect_adj;

    for (int n = 0; n < n_vertices; n++)
    {
        igraph_vs_t adc;
        igraph_vs_adj(&adc, n, IGRAPH_OUT);
        printf("%d\n", igraph_vs_type(&adc));

        vect_adj = (igraph_vector_t *)igraph_adjlist_get(&adjlist, n);

        printf("\nvertex %d n adjs %ld\n", n, igraph_vector_size(vect_adj));
        for (int f = 0; f < igraph_vector_size(vect_adj); f++)
        {
            printf("node id: %d. - ", (long int)igraph_vector_e(vect_adj, f));
            long int neighbor = VECTOR(*vect_adj)[f];
            printf("nid: %d \t", neighbor);
        }
        printf("\n");
    }

    igraph_destroy(&graph);
    return 0;
}
