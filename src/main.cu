#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define INFILE "../input/input.txt"
#define N 1000
#define ITER 500


__global__ void game_iterations(int *dev_mat, int *dev_streak, int *dev_counter, int iterations, int dim)
{   
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x > N || y > N)
        return;

    int idx = x * N + y;

    int sum;
    int streak = 0;
    int prev = 0;
    int curr;

    for(int i=0; i < iterations; i++){
        curr = dev_mat[idx];
        // statistics upgrade
        if(streak > dev_streak[idx])
            dev_streak[idx] = streak;
        if(curr && prev)
            streak++;
        if(!curr && prev)
            streak = 0;
        if(curr)
            dev_counter[idx]++;
        prev = curr;

        __syncthreads();

        // board update
        sum = tot_neighbours(idx, blockDim.x, dev_mat);

        if(!prev && sum == 3)
            dev_mat[idx] = 1;
        else if (prev && (sum >= 4 || sum == 1)){
            dev_mat[idx] = 0;
        }

	    
    }

}


__device__ int tot_neighbours(int idx, int block_dim, int *dev_mat){
    int sum = dev_mat[idx-1] + dev_mat[idx+1];
    for(int i=-1; i <= 1; i++){
        sum += dev_mat[idx - block_dim + i];
        sum += dev_mat[idx + block_dim + i];
    }
    return sum;
}


void printer(int *mat, int *streak, int *counter){
    printf("Final state of the board:\n");
    for(int i=0; i < N; i++){
        for(int j=0; j < N; j++){
            printf("%d ", mat[i*N+j]);
        }
        printf("\n");
    }
    
    printf("Overall count of alive generation for single cell:\n");
    for(int i=0; i < N; i++){
        for(int j=0; j < N; j++){
            printf("%d ", counter[i*N+j]);
        }
        printf("\n");
    }

    printf("Maximum consecutive alive generations:\n");
    for(int i=0; i < N; i++){
        for(int j=0; j < N; j++){
            printf("%d ", streak[i*N+j]);
        }
        printf("\n");
    }
}


int main(int argc, char *argv){
    int mat[N*N];
    char c;
    FILE *fin = fopen(INFILE, "r");
    int counter[N*N];
    int streak[N*N];
    
    for(int i=0; i < N; i++){
        for (int j=0; j < N; j++){
            // reading of the input file and initialization of the matrix
            fscanf(fin, "%c%*c", &c);
            if(c == 'X')
                mat[N * i + j] = 1;
            else
                mat[N * i + j] = 0;
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

    // TODO: device block distribution
    dim3 blockSize(N, N);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (N + blockSize.y - 1) / blockSize.y);

    // launch kernel on GPU
    game_iterations<<<gridSize , blockSize>>>(dev_mat, dev_streak, dev_counter, ITER, N);
    
    // gather results
	cudaMemcpy( mat, dev_mat, N * N * sizeof(int),cudaMemcpyDeviceToHost );
	cudaMemcpy( counter, dev_counter, N * N * sizeof(int),cudaMemcpyDeviceToHost );
	cudaMemcpy( streak, dev_streak, N * N * sizeof(int),cudaMemcpyDeviceToHost );

    // TODO: print or save results
    printer(mat, counter, streak);

    cudaFree(dev_counter);
    cudaFree(dev_mat);
    cudaFree(dev_streak);

    fclose(fin);

    return;
}