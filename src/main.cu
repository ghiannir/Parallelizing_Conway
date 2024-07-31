#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define INFILE "../input/input.txt"
#define N 500
#define ITER 500


__global__ void game_iterations(int *dev_mat, int *dev_streak, int *dev_counter, int iterations)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

	#TODO
	

	__syncthreads();

}



int main(int argc, char *argv){
    int mat[N][N];
    char c;
    FILE *fin = fopen(INFILE, "r");
    int counter[N][N];
    int streak[N][N];
    
    for(int i=0; i < N; i++){
        for (int j=0; j < N; j++){
            // reading of the input file and initialization of the matrix
            fscanf(fin, "%d", &c);
            if(c == 'X')
                mat[j][i] = 1;
            else
                mat[j][i] = 0;
        }
    }

    // statistics array initialization into dev mem
    int *dev_counter;
    int *dev_streak;

    cudaMalloc((void **)&dev_counter, N * N * sizeof(int));
    cudaMalloc((void **)&dev_streak, N * N * sizeof(int));

    cudaMemset(dev_counter, 0x0, N * N * sizeof(int));
    cudaMemset(dev_streak, 0x0, N * N * sizeof(int));

    // copy input to device mem
    int *dev_mat;

    cudaMalloc((void**)&dev_mat, N * N sizeof(int))
    cudaMemcpy(dev_mat, mat, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // device block distribution
    dim3 blocks, threads;

    blocks=dim3(1, 1, 1);
    threads=dim3(N * N, 1);

    game_iterations<<<blocks , threads>>>(dev_mat, dev_streak, dev_counter, ITER);

	cudaMemcpy( mat, dev_mat, N * N * sizeof(int),cudaMemcpyDeviceToHost );
	cudaMemcpy( counter, dev_counter, N * N * sizeof(int),cudaMemcpyDeviceToHost );
	cudaMemcpy( streak, dev_streak, N * N * sizeof(int),cudaMemcpyDeviceToHost );

    // TODO: print or save results

    cudaFree(dev_counter);
    cudaFree(dev_mat);
    cudaFree(dev_streak);

    fclose(fin);
    fclose(fout);

    return;
}