# Floyd-Warshall-CUDA
Attempting to Parallelize the Floyd Warshall Algorithm

The goal is to create a pipeline which can Parallelize the Floyd Warshall All Pair Shortest Path Algorithm using CUDA and adapt to accommodate graphs whose size could possibly exceed that of the GPU VRAM. A simple yet original tiled approach has been used.

### Overview of the Floyd Warshall All Pair Shortest Path Algorithm
For each pair of vertices, the Floyd Warshall algorithm checks all possible intermediate vertices in the graph and finds the vertex providing the smallest path when chosen as intermediate. Such operation has a time complexity of O(N^3) and space complexity of O(N^2), due to the adjacency matrix involved of size NxN, where the input graph has N nodes.

TC SC to eb mentioend


#### Implementation Details





FW APSP breif explanation
Implementation (two sub parts, when graph fits into VRAM and when not so)
(mention that no. of ints is a custome parameter the programmer can set according to requirement)
Hardware Used (RTX 4060)
Datasets (mention only sizes and give due credit to Bhaiyas and for the readgrapgfrom csv function, mention the expected format of dataset)




Kevin T M

