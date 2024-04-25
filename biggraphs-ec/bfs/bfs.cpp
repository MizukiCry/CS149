#include "bfs.h"

#include <cstddef>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../common/CycleTimer.h"
#include "../common/graph.h"

#define ROOT_NODE_ID 0
#define NOT_VISITED_MARKER -1

void vertex_set_clear(vertex_set *list) { list->count = 0; }

void vertex_set_init(vertex_set *list, int count) {
  list->max_vertices = count;
  list->vertices = (int *)malloc(sizeof(int) * list->max_vertices);
  vertex_set_clear(list);
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(Graph g, vertex_set *frontier, vertex_set *new_frontier,
                   int *distances) {
  const int new_dist = distances[frontier->vertices[0]] + 1;
#pragma omp parallel
  {
    Vertex *buffer = new Vertex[g->num_nodes];
    int buffer_size = 0;

#pragma omp for schedule(dynamic, 200)
    for (int i = 0; i < frontier->count; i++) {
      const int node = frontier->vertices[i];
      for (const Vertex *x = outgoing_begin(g, node),
                        *limit = outgoing_end(g, node);
           x < limit; x++) {
        if (distances[*x] == NOT_VISITED_MARKER &&
            __sync_bool_compare_and_swap(distances + *x, NOT_VISITED_MARKER,
                                         new_dist)) {
          buffer[buffer_size++] = *x;
        }
      }
    }

    int index = __sync_fetch_and_add(&new_frontier->count, buffer_size);
    memcpy(new_frontier->vertices + index, buffer,
           buffer_size * sizeof(Vertex));

    delete[] buffer;
  }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol) {

  vertex_set list1;
  vertex_set list2;
  vertex_set_init(&list1, graph->num_nodes);
  vertex_set_init(&list2, graph->num_nodes);

  vertex_set *frontier = &list1;
  vertex_set *new_frontier = &list2;

  // initialize all nodes to NOT_VISITED
  for (int i = 0; i < graph->num_nodes; i++)
    sol->distances[i] = NOT_VISITED_MARKER;

  // setup frontier with the root node
  frontier->vertices[frontier->count++] = ROOT_NODE_ID;
  sol->distances[ROOT_NODE_ID] = 0;

  while (frontier->count != 0) {

#ifdef VERBOSE
    double start_time = CycleTimer::currentSeconds();
#endif

    vertex_set_clear(new_frontier);

    top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
    double end_time = CycleTimer::currentSeconds();
    printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

    // swap pointers
    vertex_set *tmp = frontier;
    frontier = new_frontier;
    new_frontier = tmp;
  }
}

void bottom_up_step(Graph g, vertex_set *frontier, vertex_set *new_frontier,
                    int *distances) {
  const int new_dist = distances[frontier->vertices[0]] + 1;

#pragma omp parallel
  {
    Vertex *buffer = new Vertex[g->num_nodes];
    int buffer_size = 0;

#pragma omp for schedule(dynamic, 2000)
    for (int i = 0; i < g->num_nodes; ++i) {
      if (distances[i] != NOT_VISITED_MARKER)
        continue;
      for (const Vertex *x = incoming_begin(g, i), *limit = incoming_end(g, i);
           x < limit; ++x) {
        if (distances[*x] == new_dist - 1) {
          distances[i] = new_dist;
          buffer[buffer_size++] = i;
          break;
        }
      }
    }

    int index = __sync_fetch_and_add(&new_frontier->count, buffer_size);
    memcpy(new_frontier->vertices + index, buffer,
           buffer_size * sizeof(Vertex));

    delete[] buffer;
  }
}

void bfs_bottom_up(Graph graph, solution *sol) {
  vertex_set list1;
  vertex_set list2;
  vertex_set_init(&list1, graph->num_nodes);
  vertex_set_init(&list2, graph->num_nodes);

  vertex_set *frontier = &list1;
  vertex_set *new_frontier = &list2;

  for (int i = 0; i < graph->num_nodes; i++)
    sol->distances[i] = NOT_VISITED_MARKER;

  frontier->vertices[frontier->count++] = ROOT_NODE_ID;
  sol->distances[ROOT_NODE_ID] = 0;

  while (frontier->count != 0) {
    vertex_set_clear(new_frontier);

    bottom_up_step(graph, frontier, new_frontier, sol->distances);

    vertex_set *tmp = frontier;
    frontier = new_frontier;
    new_frontier = tmp;
  }
}

void bfs_hybrid(Graph graph, solution *sol) {
  vertex_set list1;
  vertex_set list2;
  vertex_set_init(&list1, graph->num_nodes);
  vertex_set_init(&list2, graph->num_nodes);

  vertex_set *frontier = &list1;
  vertex_set *new_frontier = &list2;

  for (int i = 0; i < graph->num_nodes; i++)
    sol->distances[i] = NOT_VISITED_MARKER;

  frontier->vertices[frontier->count++] = ROOT_NODE_ID;
  sol->distances[ROOT_NODE_ID] = 0;

  // int remaining_nodes = graph->num_nodes - 1;

  while (frontier->count != 0) {
    vertex_set_clear(new_frontier);

    if (frontier->count < graph->num_nodes * 0.05) {
      top_down_step(graph, frontier, new_frontier, sol->distances);
    } else {
      bottom_up_step(graph, frontier, new_frontier, sol->distances);
    }

    // remaining_nodes -= frontier->count;

    vertex_set *tmp = frontier;
    frontier = new_frontier;
    new_frontier = tmp;
  }
}
