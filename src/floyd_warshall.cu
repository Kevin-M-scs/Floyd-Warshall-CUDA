#include"floyd_warshallCUDA.h"
#include<limits>

const int INF= numeric_limits<int>::max();

//kernels
__global__ void fill_up(int *tile,int tile_size,int tile_dim,int tile_row,int tile_col) //this kernel fills up INF where needed
{
    //this kernel fills up the left out weights in the adjacency matrix tile by tile
    int idx= blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=tile_size)return; //one tile is filled at a time where each thread works on one element

    if(tile_row!=tile_col)//In non diagonal tiles there wont be any self pointing edges (ie weight=0)
    {
        if(tile[idx]==0){tile[idx]=INF;}
    }
    else//ie if we have a diagonal tile
    {
        if(!(idx%(tile_dim+1)==0) && tile[idx]==0){tile[idx]=INF;}
    }
}

__global__ void store_dists_to_intmdt_vertex(int *d_adj,int *d_to,int offset,int intmdt_vertex_idx,int vertex_tile_col,int wt_col,int tile_dim,int V)
{
    int idx= blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=V)return;
    int tile_row= idx/tile_dim;
    int tile_start_offset= offset*(tile_row*(V/tile_dim)+vertex_tile_col);

    int *tile= d_adj + tile_start_offset; //each thread gets access to its respective tile on VRAM
    int wt_row= idx % tile_dim; //row index which thread has to handle within the tile
    int wt= tile[wt_row*tile_dim + wt_col];

    d_to[idx]=wt;
}

__global__ void     store_dists_from_intmdt_vertex(int *d_adj,int *d_from,int offset,int intmdt_vertex_idx,int vertex_tile_row,int wt_row,int tile_dim,int V)
{
    int idx= blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=V)return;
    int tile_col= idx/tile_dim;
    int tile_start_offset= offset*(vertex_tile_row*(V/tile_dim)+tile_col);

    int *tile= d_adj + tile_start_offset; //each thread gets access to its respective tile on VRAM
    int wt_col= idx % tile_dim; //col index which thread has to handle within the tile
    int wt= tile[wt_row*tile_dim + wt_col];

    d_from[idx]=wt;
}

__global__ void relaxation_per_intmdt_vertex(int *tile_row_start,int *d_to,int *d_from,int tile_row,int offset,int tile_dim,int V) //here tile_row is the absolute tile_row
{
    int idx= blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>=V*tile_dim)return;

    int vram_abs_idx= ( offset * tile_row ) + idx; //indexing in the 1D flat array on VRAM  //rel_tile_row
    int tile_col= idx / (tile_dim*tile_dim);
    int abs_row= ((vram_abs_idx / tile_dim)%tile_dim) + (tile_row*tile_dim); //absolute row of weight in adj matrix format //tile_row_abs and abs_vram_idx
    int abs_col= (vram_abs_idx % tile_dim) + (tile_col*tile_dim);//absolute col of weight in adj matrix format

    if((d_to[abs_row]!=INF && d_from[abs_col]!=INF) && (d_to[abs_row] + d_from[abs_col] < tile_row_start[idx]))
    {
        tile_row_start[idx] = d_to[abs_row] + d_from[abs_col];
    }
}


//host functions
void complete_edges(Graph &g,int num_threads_per_block)
{ 
    int V= g.getVertices();
    int **tile_list= g.get_tile_list();
    int tile_dim= g.get_tile_dim();
    int tile_size= tile_dim*tile_dim;//number of elements per tile
    int tile_count= g.get_tile_count();

    //allocate memory for the tile on VRAM
    int *tile;int tile_row,tile_col;
    cudaMalloc((void**)&tile,tile_size*sizeof(int));

    for(tile_row = 0; tile_row < (V/tile_dim); tile_row++)
    {
        for (tile_col = 0; tile_col < (V/tile_dim); ++tile_col)
        {
            int tile_flat_idx= tile_row*(V/tile_dim)+tile_col;
            cudaMemcpy(tile,tile_list[tile_flat_idx],tile_size*sizeof(int),cudaMemcpyHostToDevice);
            dim3 thread(num_threads_per_block,1,1);
            dim3 block((tile_size+num_threads_per_block-1)/num_threads_per_block,1,1);
            //call kernel for filling INF in selected tile
            fill_up<<<block,thread>>>(tile,tile_size,tile_dim,tile_row,tile_col);
            cudaDeviceSynchronize();//CPU waits for GPU
            cudaMemcpy(tile_list[tile_flat_idx],tile,tile_size*sizeof(int),cudaMemcpyDeviceToHost);
        }
    }
    g.set_tile_list(tile_list);//to bring the modified tile list to the graph object
    cudaFree(tile);
}

void copy_adj_mat(Graph& g,int *d_adj,int offset,int direction,int start_tile,int end_tile,int batch_no) //here direction=0 means RAM to VRAM and 1 means VRAM to RAM
{
    int tile_dim=g.get_tile_dim();int V=g.getVertices();int **tile_list=g.get_tile_list();
    int i=0; //for each batch, i is reset to 0 so that it starts overwriting the d_adj buffer from the beginning in VRAM
    
    int tile_flat_idx;
    for(tile_flat_idx=start_tile;tile_flat_idx<=end_tile;tile_flat_idx++)
    {
        if(direction==0) cudaMemcpy(d_adj + offset*i,tile_list[tile_flat_idx],offset*sizeof(int),cudaMemcpyHostToDevice);
        else if(direction==1) cudaMemcpy(tile_list[tile_flat_idx],d_adj + offset*i,offset*sizeof(int),cudaMemcpyDeviceToHost);
        i++;
        //Note: We can do VRAM pointer arithmetic inside cudaMemcpy() despite cudaMempcy() being called from the host
    }
    // if(direction==0) printf("Batch[%d] tiles successfully copied to VRAM\n",batch_no);  //Have commented else it will fill up the terminal
    // else if(direction==1)  printf("Batch[%d] tiles successfully copied back to RAM\n",batch_no);
   
}

void store_intmdt_vertex_dists(Graph& g,int num_threads_per_block,int intmdt_vertex_idx,int *d_adj,int *d_to,int *d_from,int offset)
{
    int tile_dim=g.get_tile_dim();int V=g.getVertices();
    //the below values are common to all threads hence precomputed them on HOST and passed them as args to kernel to avoid any resource wastage, though overhead avoided may be minimal
    //Since the matrix is a square matrix thus the same values can be used in both the kernels (d_to and d_from extraction)
    int vertex_tile_col_or_row= intmdt_vertex_idx/tile_dim;//the column of tiles which contains the distances from all other vertices TO this vertex
    int wt_col_or_row= intmdt_vertex_idx % tile_dim;//the col within a tile which has to be extracted
   
    dim3 thread(num_threads_per_block,1,1);
    dim3 block((V+num_threads_per_block-1)/num_threads_per_block,1,1); //we need totally V threads

    cudaStream_t s1,s2;
    cudaStreamCreate(&s1); //creating two streams, one for the row extraction kernel and one for the column extraction kernel
    cudaStreamCreate(&s2); //this allows us to launch the two kernel in parallel instead of sequentially!

    store_dists_to_intmdt_vertex<<<block,thread,0,s1>>>(d_adj,d_to,offset,intmdt_vertex_idx,vertex_tile_col_or_row,wt_col_or_row,tile_dim,V);
    store_dists_from_intmdt_vertex<<<block,thread,0,s2>>>(d_adj,d_from,offset,intmdt_vertex_idx,vertex_tile_col_or_row,wt_col_or_row,tile_dim,V);

    cudaDeviceSynchronize();
    
    //testing
    // int *h_to=(int *)malloc(V*sizeof(int));
    // cudaMemcpy(h_to,d_from,V*sizeof(int),cudaMemcpyDeviceToHost);
    // int i;
    // for(i=0;i<V;i++){
    //     cout<<h_to[i]<<" ";
    // }
    // cout<<"\n";
}
//mod stands for modified, all modified functions are those written to be used in the special case where the full adj matrix doesnt fit into VRAM and it has to be broken down into batches
void store_dists_to_intmdt_vertex_mod(Graph& g, int num_threads_per_block, int intmdt_vertex_idx,int *d_to) 
{
    /*Note that this function copies the to and from arrays from the ENTIRE adjacency matrix into VRAM and NOT only those required for the current
    batch. It was done to manage a tradeoff between having kernel launches per batch VS having a single host function replicate the same but it is
    called only once per intermediate vertex. This does indeed add significant overhead.
    */
    int tile_dim=g.get_tile_dim();int tile_row;int *tile;int row;
    int **tile_list= g.get_tile_list();
    int V=g.getVertices();
    int vertex_tile_col=intmdt_vertex_idx/tile_dim; //the column of tiles which contains the distances from all other vertices TO this vertex
    
    //collect the desired column
    int *h_to=(int *)malloc(V*sizeof(int));int pointer=0;//used to shift the position of insertion in h_to
    for(tile_row = 0 ; tile_row < (V/tile_dim); tile_row++)
    {
        int tile_flat_idx= tile_row*(V/tile_dim)+vertex_tile_col;
        tile= tile_list[tile_flat_idx];
        int col_idx_in_tile= intmdt_vertex_idx % tile_dim;
        for(row=0;row<tile_dim;row++)
        {
            int wt_flat_idx= row*tile_dim + col_idx_in_tile;
            *(h_to+pointer)=tile[wt_flat_idx];
            pointer++;
        }
    }
    //copy the collected column to VRAM
    cudaMemcpy(d_to,h_to,V*sizeof(int),cudaMemcpyHostToDevice);
    free(h_to);
}

void store_dists_from_intmdt_vertex_mod(Graph& g, int num_threads_per_block, int intmdt_vertex_idx,int *d_from) //used only if matrix doesnt fully fit in VRAM
{
    int tile_dim=g.get_tile_dim();int tile_col;int *tile;int col;
    int **tile_list= g.get_tile_list();
    int V=g.getVertices();
    int vertex_tile_row=intmdt_vertex_idx/tile_dim; //the row of tiles which contains the distances FROM this vertex to all other vertices
    
    //collect the desired row
    int *h_from=(int *)malloc(V*sizeof(int));int pointer=0;//used to shift the position of insertion in h_from
    for(tile_col = 0 ; tile_col < (V/tile_dim); tile_col++)
    {
        int tile_flat_idx= vertex_tile_row*(V/tile_dim) + tile_col;
        tile= tile_list[tile_flat_idx];
        int row_idx_in_tile= intmdt_vertex_idx % tile_dim;
        for(col=0;col<tile_dim;col++)
        {
            int wt_flat_idx= row_idx_in_tile*tile_dim + col;
            *(h_from+pointer)=tile[wt_flat_idx];
            pointer++;
        }
    }
    //copy the collected row to VRAM
    cudaMemcpy(d_from,h_from,V*sizeof(int),cudaMemcpyHostToDevice);
    free(h_from); 
}

void fw_per_inmdt_vertex(Graph& g,int num_threads_per_block,int *d_adj,int *d_to,int *d_from)
{
    int tile_row;int tile_dim=g.get_tile_dim(),V=g.getVertices();
    int offset= tile_dim * V; //the size of one full row of tiles

    dim3 thread(num_threads_per_block,1,1);
    dim3 block(((offset)+num_threads_per_block-1)/num_threads_per_block,1,1); //we need totally V*tile_dim threads

    for(tile_row=0 ; tile_row < (V/tile_dim); tile_row++) //have used single tile row approach for now
    {
        relaxation_per_intmdt_vertex<<<block,thread>>>(d_adj + (offset*tile_row),d_to,d_from,tile_row,offset,tile_dim,V);
        cudaDeviceSynchronize();
    }
}

void fw_per_inmdt_vertex_mod(Graph& g,int num_threads_per_block,int *d_adj,int *d_to,int *d_from,int tile_row_start,int tile_rows_per_batch)
{
    int tile_row;int tile_dim=g.get_tile_dim(),V=g.getVertices();
    int offset= tile_dim * V; //the size of one full row of tiles (was tile_dim*tile_dim earlier)
    
    dim3 thread(num_threads_per_block,1,1);
    dim3 block(((offset)+num_threads_per_block-1)/num_threads_per_block,1,1); //we need totally V*tile_dim threads

    int update=0;
    for(tile_row=0 ; tile_row < tile_rows_per_batch; tile_row++) //have used single tile row approach for now
    {   //here tile_row is relative of the rows of the batch, so tile_row=0 means first row of batch and not absolute tile_row=0
        relaxation_per_intmdt_vertex<<<block,thread>>>(d_adj + (offset*tile_row),d_to,d_from,tile_row_start + update,offset,tile_dim,V);
        cudaDeviceSynchronize();
        update++;
    }
    //the tile_row_start+update value helps mimic the operation as if it were running with the whole adjacency matrix in VRAM
    //ie it helps us obtain element indices as if the whole adjacency matrix was in VRAM
}

void Floyd_Warshall(Graph& g,int num_threads_per_block,long long max_capacity)
{
    int *d_adj;int *d_to;int *d_from;int intmdt_vertex_idx;
    int V= g.getVertices();int tile_dim=g.get_tile_dim();
    int offset= tile_dim*tile_dim; //same as tile_size

    cudaError_t error=cudaMalloc((void**)&d_adj,V*V*sizeof(int));
    if(error!= cudaSuccess)
    {
        printf("There was an error in VRAM allocation\n");
        printf("Error:%s\n",cudaGetErrorString(error)); //to get the error corresponding to the int returned by cudaMalloc()
    }
    else printf("VRAM Memory Allocated Successfully!\n");
    cudaMalloc((void**)&d_to,V*sizeof(int));
    cudaMalloc((void**)&d_from,V*sizeof(int));

    //below is the main modified region which controls the entire batching process
    long long num_weights= V*V;int num_batches;long long max_tiles_per_batch;
    int total_tiles= (V/tile_dim)*(V/tile_dim);
    if(num_weights<=max_capacity) 
    {
        num_batches=1;//proceed normally
        max_tiles_per_batch = total_tiles;
    }
    else
    {
        int num_rows;
        //find the best-fit num of rows that just about reach max capacity, we will use this many rows in a single batch
        num_rows = max_capacity / (V * tile_dim); //am assuming that for all our datasets (with tile_size 2500) we will get atleast a single row to fit into max_capacity (120GB)
        num_batches= (num_weights) / (num_rows*V*tile_dim);// ie assuming that the size of a single row itself will NOT exceed the size of VRAM
        if((num_weights) % (num_rows*V*tile_dim) > 0){num_batches=num_batches+1;}//here the last batch will have the leftover tile row/rows
        max_tiles_per_batch= (V/tile_dim) * num_rows; 
        cout<<"Number batches formed:"<<num_batches<<endl;
        cout<<"Best fit num_tile_rows per batch:"<<num_rows<<endl;
        cout<<"Max tiles per batch:"<<max_tiles_per_batch<<endl;
    }
    int start_tile=0,end_tile=-1;
    if(num_batches==1)
    {   
        start_tile=0;end_tile= total_tiles -1;
        copy_adj_mat(g,d_adj,offset,0,start_tile,end_tile,0);
        for(intmdt_vertex_idx=0 ; intmdt_vertex_idx<V ; intmdt_vertex_idx++)
        {
            store_intmdt_vertex_dists(g,num_threads_per_block,intmdt_vertex_idx,d_adj,d_to,d_from,offset);

            fw_per_inmdt_vertex(g,num_threads_per_block,d_adj,d_to,d_from);
        }
        copy_adj_mat(g,d_adj,offset,1,start_tile,end_tile,0);
    }
    else
    {
        //NOTE: This approach is built such that it will fail if the size of a single tile_row exceeds the max capacity, thus do keep this in mind!
        for(intmdt_vertex_idx=0 ; intmdt_vertex_idx<V ; intmdt_vertex_idx++)
        {
            start_tile=0;end_tile=-1;
            //this one storage call will be used for all batches across this particular intmdt vertex
            store_dists_to_intmdt_vertex_mod(g,num_threads_per_block,intmdt_vertex_idx,d_to);
            store_dists_from_intmdt_vertex_mod(g,num_threads_per_block,intmdt_vertex_idx,d_from); 
            int batch_no;
            for (batch_no=0 ; batch_no < num_batches ; batch_no++)
            {   //handle last case here too by capping end_tile accordingly
                start_tile= end_tile + 1;
                end_tile= start_tile + max_tiles_per_batch -1;
                if(end_tile >= total_tiles)end_tile=total_tiles-1; //capping the end tile if it exceeds total_tiles in the last batch
                // cout<<"Start tile: "<<start_tile<<" End Tile: "<<end_tile<<endl;
                int tile_row_start= start_tile / (V/tile_dim);  //VERIFY THAT THEY ARE CORRECT FOR EVERY BATCH
                int tile_row_end= end_tile / (V/tile_dim);
                int tile_rows_per_batch = tile_row_end - tile_row_start + 1;

                copy_adj_mat(g,d_adj,offset,0,start_tile,end_tile,batch_no); //copy the batch to VRAM

                fw_per_inmdt_vertex_mod(g,num_threads_per_block,d_adj,d_to,d_from, tile_row_start, tile_rows_per_batch);//find the shortest path for all the vertices in the batch

                copy_adj_mat(g,d_adj,offset,1,start_tile,end_tile,batch_no); //copy the batch back to RAM
            }
        }
    }
    cudaFree(d_adj);cudaFree(d_to);cudaFree(d_from);
}







