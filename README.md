# Floyd-Warshall-CUDA
Attempting to Parallelize the Floyd Warshall Algorithm

The goal is to create a pipeline which can Parallelize the Floyd Warshall All Pair Shortest Path Algorithm using CUDA and adapt to accommodate graphs whose size could possibly exceed that of the GPU VRAM. A simple yet original tiled approach has been used.

### Overview of the Floyd Warshall All Pair Shortest Path Algorithm
For each pair of vertices, the Floyd Warshall algorithm checks all possible intermediate vertices in the graph and finds the vertex providing the smallest path when chosen as intermediate. Such operation has a time complexity of O(N^3) and space complexity of O(N^2), due to the adjacency matrix involved of size NxN, where the input graph has N nodes.

### Implementation Details
Firstly, the adjacency matrix is broken down into *tiles* of fixed size. This simplifes the transfer of data between RAM and VRAM later on. Then the input graph's CSV file is read and the adjacency matrix is populated.
The program proceeds to check whether the specified VRAM capacity, set by user, can contain the input graph's adjacency matrix or not. According to this, subsequent actions are determined:
#### When graph fits into VRAM
+ Full adjacency matrix is shifted to VRAM in a tile-wise fashion via repeated cudaMemcpy() calls
+ For each intermediate vertex, *d_to* and *d_from* arrays are extracted from the VRAM-resident matrix and stored in VRAM. For the kth intermediate vertex: <br>
    *d_to* => array storing the kth column of matrix<br>
    *d_from* => array storing the kth row of matrix<br>
+ *d_to* and *d_from* extraction is parallelized by running their extraction kernels in two separate CUDA streams
+ These arrays are used to relax the entire matrix for the kth intermediate vertex. Currently, a single row of tiles is dealt with at a time, with all the weights that the row covers, being relaxed in parallel. Note that number of tile rows involved above, could potentially be increased for better performance
+ Above process repeats until all N^2 intermediate vertices have been covered
+ Relaxed matrix is shifted back to RAM, tile_wise
#### When graph exceeds specified VRAM limit
+ We first calculate the optimal number of tile rows that can fit into the specified VRAM limit, say *x* rows
+ The adjacency matrix is broken into batches, each having *x* tile rows, last batch may have less 
+ For each intermediate vertex k, *d_to* and *d_from* arrays are extracted and stored in VRAM. Their extraction, however, is sequential here. Please refer to the *Observations* section for a broader explanation.
+ In a batch-wise manner, each set of *x* tile rows is transferred to VRAM, relaxed using *d_to* and *d_from* and subsequently shifted back to RAM. This repeats until the full matrix has been relaxed using the intermediate vertex under consideration.
+ Using the above procedure, the full matrix is relaxed for every intermediate vertex until all N^2 of them have been covered

### Observations 
Incase the graph does not fit into VRAM, we are forced to perform *d_to* and *d_from* extraction on a RAM-resident matrix and hence the lack of CUDA parallelization. This sequential operation indeed slows down the algorithm, furthermore, the overhead of per-batch cudaMemcpy() adds on.
For the time being, have maintained this approach.

### Visual description of program flow
![CUDA_functions_kernels_flow3](./../../Documents/DSA_via_CUDA/Paper_Contents/CUDA_functions_kernels_flow_3.jpg)

### Hardware Used
The obtained results are from runs on an NVIDIA GeForce RTX 4060 Laptop GPU with 8GB GDDR6 Memory.

### Datasets
Each graph is stored in a csv file in a [source,destination,weight] format. Credit for the datasets and the functions used to read them, go to my kind seniors. 
|Name|No. Edges|No. Vertices|
|---|---|---|
|Graph 1|1,00,000|5,000|
|Graph 2|1,50,000|7,500|
|Graph 3|2,00,000|10,000|
|Graph 4|2,50,000|12,500|
|Graph 5|3,00,000|15,000|
|Graph 6|3,50,000|17,500|
|Graph 7|4,00,000|20,000|
|Graph 8|4,50,000|22,500|
|Graph 9|5,00,000|25,000|
|Graph 10|5,50,000|27,500|
|Graph 11|6,00,000|30,000|

Please refer to ***Metrics.txt*** for the timing results.



