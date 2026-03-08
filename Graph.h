#pragma once
#include<vector>
#include<iostream>

using namespace std;

class Graph
{
    int V;
    int **tile_list; //pointer to the flattened adjacency matrix
    int tile_dim; //length of each square tile, has to be a multiple of V, have not handled edge cases
    int tile_count; //total number of tiles in adjacency matrix
    public:
        int getVertices(); //made these functions as const so that one can use them via const reference variables
        int **get_tile_list();
        int get_tile_dim();
        int get_tile_count();
        void set_tile_list(int **);
        Graph(int V,int s) : V(V),tile_dim(s),tile_count((V/tile_dim)*(V/tile_dim)) //s is for tile_dim
        {
            tile_list=(int **)malloc((tile_count)*sizeof(int *));if(tile_list==NULL){cout<<"Tile List Memory Allocation Error!\n";exit(1);}
            int i;for(i=0;i<tile_count;i++){tile_list[i]=(int *)calloc(tile_dim*tile_dim,sizeof(int));if(tile_list[i]==NULL){cout<<"Tile Memory Allocation Error\n"<<i<<endl;exit(1);}};   
        } 
        void addEdge(int u, int v, int w);
        void printGraph();  
};

Graph readGraphFromCSV(const string &fileName,int tile_dim);
/*
tile_list is an array of pointers to tiles, each element of it is a pointer to a tile stored in row major form.
*/