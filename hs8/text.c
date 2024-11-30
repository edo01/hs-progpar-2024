
/* Task 1
 *
 *
 * 
 * 
*/



/* Task 2
 *
 * Execution commande:
 *
 * ./run -k sample --gpu --variant cuda_grad --size 256
 * ./run -k sample --gpu --variant cuda_grad --size 512
 * ./run -k sample --gpu --variant cuda_grad --size 1024
 * ./run -k sample --gpu --variant cuda_grad --size 2048
 * ./run -k sample --gpu --variant cuda_grad --size 4096
 * 
 * Global Work Size (X) = grid.x * block.x
 * Global Work Size (Y) = grid.y * block.y
 * Local Work Size = block.x * block.y
 * Total Threads=grid.x×grid.y×block.x×block.y
 * 
 * Image Size::
 * 256 × 256
 *    Grid Dimensions: 
 *    Global Work Size: 
 *    Local Work Size: 
 *    Threads in Parallel: 
 * 
 * 512 × 512
 *    Grid Dimensions: 
 *    Global Work Size: 
 *    Local Work Size: 
 *    Threads in Parallel: 
 * 
 * 1024 × 1024
 *    Grid Dimensions: 
 *    Global Work Size: 
 *    Local Work Size: 
 *    Threads in Parallel: 
 *
 * 2048 × 2048
 *    Grid Dimensions: 
 *    Global Work Size: 
 *    Local Work Size: 
 *    Threads in Parallel: 
 * 
 * 4096 × 4096
 *    Grid Dimensions: 
 *    Global Work Size: 
 *    Local Work Size: 
 *    Threads in Parallel: 
 *
 * Task 2.1 Warp Size – stripes Kernel
 *
 * A warp consists of 32 threads executing the same instruction simultaneously. 
 * If threads encounter a conditional instruction (e.g., if), 
 * those that evaluate to false will "sleep," causing warp divergence and reducing efficiency.
 * 
 * All threads in a warp execute the same instruction at the same cycle.
 * Divergence occurs when threads follow different branches, slowing execution.
 * Align thread counts to multiples of 32 to maximize efficiency and minimize divergence.
 * 
 * In stripes, the condition if (j & MASK) demonstrates how divergence impacts performance: 
 * frequent condition changes increase divergence and reduce GPU efficiency.
 *
*/

/* Task 2.2
 * Test results:
 *
 * 
 * 
 * Below are the reasons: 
 * 
 * 1. Each pixel computationin the Manselbrot is independent, GPUs are designed to handle
 *    thousands of threads simultaneously, making them perfect for problems like Mandelbrot.
 * 2. Each thread independently computes its own pixel, minimizing communication 
 *    and avoiding inter-thread dependencies.
 * 3. GPUs excel at handling such repetitive arithmetic computations due to their high number of ALUs

*/