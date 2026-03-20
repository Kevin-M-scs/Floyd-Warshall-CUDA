#include"Graph.h"
#include"floyd_warshallCUDA.h"
#include<stdlib.h> //for the atoi() function
#include<chrono> //for timing
//Final FW approach which includes procedures to handle cases where the entire adjacency matrix does not fit into RAM
int main(int argc,char *argv[]){
    if(argc==1){
        cout<<"Path to Graph's CSV file NOT mentioned!"<<endl;
        exit(1);
    }
    char *path = argv[1];
    int tile_dim=2500; //change according to desired setup
    int num_threads_per_block=256;
    long long max_capacity=22500000000; //this is 1.5L squared 22500000000 ie it refers to the max num of weights that can be stored in VRAM
    //Assuming such a max size so that there is sufficient buffer space in VRAM for d_to,d_from and device related data, if VRAM = 120GB
    Graph graph= readGraphFromCSV(path,tile_dim);
    complete_edges(graph,num_threads_per_block);
    cout<<"Graph Created"<<endl;
    cout<<"PATH USED: "<<path<<endl;

    // graph.printGraph();
    // cout<<"\n"<<endl;

    auto startCUDA= std::chrono::high_resolution_clock::now();
    Floyd_Warshall(graph,num_threads_per_block,max_capacity);
    auto endCUDA= std::chrono::high_resolution_clock::now();
    auto durationCUDA=std::chrono::duration_cast<std::chrono::milliseconds>(endCUDA-startCUDA);
    cout<<"CUDA Time: "<<durationCUDA.count()<<endl;
    cout<<"\n"<<endl;

    cout<<"SUCCESSFUL RUN"<<endl;


    // graph.printGraph();
    return 0;
}

