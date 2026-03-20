#pragma once
#include "Graph.h"
#include <iostream>
#include <vector>


//HOST functions used
void complete_edges(Graph &g,int num_threads_per_block);
void store_dists_to_intmdt_vertex(const Graph& g, int num_threads_per_block, int intmdt_vertex_idx,int *d_to);
void store_dists_from_intmdt_vertex(const Graph& g, int num_threads_per_block, int intmdt_vertex_idx,int *d_from);
void fw_per_intmdt_vertex(Graph& g,int *h_wts,int *d_wts,int *d_to,int *d_from,bool *d_flag,int num_threads_per_block);
void Floyd_Warshall(Graph& g,int num_threads_per_block,long long max_capacity);