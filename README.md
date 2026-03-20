# Floyd-Warshall-CUDA
Attempting to Parallelize the Floyd Warshall Algorithm

The goal is to create a pipeline which can Parallelize the Floyd Warshall All Pair Shortest Path Algorithm using CUDA and adapt to accommodate graphs whose size could possibly exceed that of the GPU VRAM. A simple yet original tiled approach has been used.

### Overview of the Floyd Warshall All Pair Shortest Path Algorithm
For each pair of vertices, the Floyd Warshall algorithm checks all possible intermediate vertices in the graph and finds the vertex providing the smallest path when chosen as intermediate. Such operation has a time complexity of O(N^3) and space complexity of O(N^2), due to the adjacency matrix involved of size NxN, where the input graph has N nodes.

### Implementation Details
Firstly, the adjacency matrix is broken down into *tiles* of fixed size. This simplifes the transfer of data between RAM and VRAM later on. Then the input graph's CSV file is read and the adjacency matrix is populated.
The program follows to check whether the specified VRAM capacity, set by user, can contain the input graph's adjacency matrix or not. According to this, subsequent actions are determined:
#### When graph fits into VRAM
+ Full adjacency matrix is shifted to VRAM in a tile-wise fashion via repeated cudaMemcpy() calls
+ For each intermediate vertex, a *d_to* and *d_from* arrays are extracted from the VRAM-resident matrix and stored in VRAM. For the kth intermediate vertex: <br>
    d_to => array storing the kth column of matrix<br>
    d_from => array storing the kth row of matrix
+ *d_to* and *d_from* are extracted in parallel by running their extraction kernels in two separate CUDA streams








FW APSP breif explanation
Implementation (two sub parts, when graph fits into VRAM and when not so)
(mention that no. of ints is a custome parameter the programmer can set according to requirement)
Hardware Used (RTX 4060)
Datasets (mention only sizes and give due credit to Bhaiyas and for the readgrapgfrom csv function, mention the expected format of dataset)




Kevin T M

