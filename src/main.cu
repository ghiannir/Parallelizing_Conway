#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define INFILE "../input/input.txt"
// #define N 1000
// #define ITER 500


__global__ void game_iterations(int *dev_mat, int *dev_streak, int *dev_counter, int iterations, int dim){   
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x > dim || y > dim)
        return;

    int idx = x * dim + y;

    int sum;
    int prev = 0;
    int curr;

    for(int i=0; i < iterations; i++){
        curr = dev_mat[idx];
        // statistics upgrade
        if(curr && prev)
            dev_streak[idx]++;
        if(curr)
            dev_counter[idx]++;
        prev = curr;

        __syncthreads();

        // board update
        sum = tot_neighbours(idx, dim, dev_mat);

        if(!prev && sum == 3)
            dev_mat[idx] = 1;
        else if (prev && (sum >= 4 || sum == 1)){
            dev_mat[idx] = 0;
        }

	    
    }

}

// TODO: farla piu leggibile
__device__ int tot_neighbours(int idx, int block_dim, int *dev_mat){
    int sum = 0;

    // flags for border cells
    int left=0, right=0, up=0, down=0;
    
    if(idx%block_dim == 0)
        left = 1; 
    else if((idx+1)%block_dim == 0)
        right = 1;
    if(idx-block_dim < 0)
        up = 1;
    else if(idx+block_dim >= n*n)
        down=1;
    // sum all existing nearby blocks vlaues
    if(!up){
        sum += dev_mat[idx-block_dim];
        if(!left)
            sum += dev_mat[idx-block_dim-1];
        if(!up)
            sum += dev_mat[idx-block_dim+1];
    }
    if(!down){
        sum += dev_mat[idx+block_dim];
        if(!left)
            sum += dev_mat[idx+block_dim-1];
        if(!up)
            sum += dev_mat[idx+block_dim+1];
    }
    if(!left)
        sum += dev_mat[idx-1];
    if(!up)
        sum += dev_mat[idx+1];
    return sum;
}


void printer(int *mat, int *streak, int *counter, int N);


int main(void){
    int n;
    char *num_elements = getenv("N");
    sscanf(num_elements, "%d", &n);
    int iter;
    char *num_iter = getenv("ITER");
    sscanf(num_iter, "%d", &iter);

    int *mat;
    FILE *fin = fopen(INFILE, "r");
    int *counter;
    int *streak;

    // matrix allocation
    mat = (int *)malloc(n*n*sizeof(int));
    counter = (int *)malloc(n*n*sizeof(int));
    streak = (int *)malloc(n*n*sizeof(int));
    
    for(int i=0; i < n; i++){
        for (int j=0; j < n; j++){
            // reading of the input file and initialization of the matrix
            if(fgetc(fin) == 'X')
                mat[n * i + j] = 1;
            else
                mat[n * i + j] = 0;
        }
    }

    // statistics array initialization into dev mem
    int *dev_counter;
    int *dev_streak;

    cudaMalloc((void **)&dev_counter, n * n * sizeof(int));
    cudaMalloc((void **)&dev_streak, n * n * sizeof(int));

    cudaMemset(dev_counter, 0x0, n * n * sizeof(int));
    cudaMemset(dev_streak, 0x0, n * n * sizeof(int));

    // copy input to device mem
    int *dev_mat;

    cudaMalloc((void**)&dev_mat, n * n sizeof(int))
    cudaMemcpy(dev_mat, mat, n * n * sizeof(int), cudaMemcpyHostToDevice);

    // TODO: device block distribution
    dim3 blockSize(n, n);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);

    // launch kernel on GPU
    // TODO: time measurement
    game_iterations<<<gridSize , blockSize>>>(dev_mat, dev_streak, dev_counter, iter, n);
    
    // gather results
	cudaMemcpy(mat, dev_mat, n * n * sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(counter, dev_counter, n * n * sizeof(int),cudaMemcpyDeviceToHost);
	cudaMemcpy(streak, dev_streak, n * n * sizeof(int),cudaMemcpyDeviceToHost);

    cudaFree(dev_counter);
    cudaFree(dev_mat);
    cudaFree(dev_streak);

    // print or save results
    printer(mat, counter, streak, n);

    free(mat);
    free(counter);
    free(streak);

    fclose(fin);

    return 0;
}


void printer(int *mat, int *streak, int *counter, int N){
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