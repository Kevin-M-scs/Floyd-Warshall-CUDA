#include "Graph.h"
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm> // for remove and find_if function
#include <cctype>    // for isspace function
using namespace std;

int Graph::getVertices()
{
    return V;
}

int Graph::get_tile_dim() 
{
    return tile_dim;
}

int Graph::get_tile_count() 
{
    return tile_count;
}

int **Graph::get_tile_list() 
{
    return tile_list;
}

void Graph::set_tile_list(int **tile_list)
{
    this->tile_list=tile_list;
}

void Graph::addEdge(int u, int v, int w)
{ 
    int idx_u=u-1,idx_v=v-1; //1->1 weight corresponds to index (0,0) in adjacency matrix and so on, assuming csv file has nodes starting from 1
    int tile_x= idx_u/tile_dim;//finding index of tile corresponding to edge
    int tile_y= idx_v/tile_dim;//flattened position of tile in tile_list array
    int tile_flat_idx= tile_x*(V/tile_dim)+tile_y;
    int pos_x= idx_u%tile_dim;//index of edge weight within its respective tile
    int pos_y= idx_v%tile_dim;//these will be used to find the flattened index
    int wt_flat_idx= pos_x*(tile_dim)+pos_y;//flattened position of weight in its corresponding tile's array
    tile_list[tile_flat_idx][wt_flat_idx]=w;
}

void Graph::printGraph()
{ 
    int tile_row,tile_col,i,idx_u=0,idx_v=0,inner_row_count;
    for (tile_row = 0; tile_row < (V/tile_dim); ++tile_row)
    {   
        inner_row_count=0; 
        while(inner_row_count<tile_dim)
        {
            idx_v=0;
            cout<<idx_u+1<<" -> ";
            for (tile_col = 0; tile_col < (V/tile_dim); ++tile_col)
            {
                int tile_flat_idx= tile_row*(V/tile_dim)+tile_col;
                for(i=0;i<tile_dim;i++)
                {
                    int pos_x= idx_u%tile_dim;
                    int pos_y= idx_v%tile_dim;
                    int wt_flat_idx= pos_x*(tile_dim)+pos_y;
                    cout << "(" << idx_v+1 << ", " << tile_list[tile_flat_idx][wt_flat_idx] << ") ";
                    idx_v++;
                }
            }
            cout<<endl;
            idx_u++;
            inner_row_count++;
        }
    }
}

Graph readGraphFromCSV(const string& fileName,int tile_dim)
{
    ifstream file(fileName);
    if (!file.is_open())
    {
        cerr << "Error opening file!" << endl;
        exit(1);
    }
    string line;
    int max_node = 0;

    // Trim whitespace from each string
    auto trim = [](string &s)
    {
        s.erase(s.begin(), find_if(s.begin(), s.end(), [](int ch)
                                   { return !isspace(ch); }));
        s.erase(find_if(s.rbegin(), s.rend(), [](int ch)
                        { return !isspace(ch); })
                    .base(),
                s.end());
    };

    while(getline(file, line))
    {
         if(line.empty()) continue; //skip empty lines
        // Remove quotes if present
        line.erase(remove(line.begin(), line.end(), '\"'), line.end());

        stringstream ss(line);
        string u_str, v_str, w_str;
        getline(ss, u_str, ',');
        getline(ss, v_str, ',');
        getline(ss, w_str);
        
        trim(u_str);
        trim(v_str);
        trim(w_str);
        max_node = max(max_node, max(stoi(u_str), stoi(v_str)));
    }
    // cout<<max_node<<endl;
    Graph g(max_node,tile_dim); 
    file.clear();
    file.seekg(0);
    int count = 0;

    while (getline(file, line) )
    {
        if(line.empty()) continue; //skip empty lines
        // Remove quotes if present
        line.erase(remove(line.begin(), line.end(), '\"'), line.end());

        stringstream ss(line);
        string u_str, v_str, w_str;
        getline(ss, u_str, ',');
        getline(ss, v_str, ',');
        getline(ss, w_str);

        trim(u_str);
        trim(v_str);
        trim(w_str);

        int u = stoi(u_str);
        int v = stoi(v_str);
        int w = stoi(w_str);
        
        g.addEdge(u, v, w);
        count++;
    }
    // cout<<"number of nodes : "<<max_node<<endl;
    cout<<"number of edges : "<<count<<endl;
    return g;
}