#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

#define INFILE "../input/input.txt"
#define OUTMAT "../output/mat.txt"
#define OUTCNT "../output/cnt.txt"
#define OUTSTREAK "../output/streak.txt"
// #define N 1000
// #define ITER 500

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
    else if(idx+block_dim >= block_dim*block_dim)
        down=1;
    // sum all existing nearby blocks vlaues
    if(!up){
        sum += dev_mat[idx-block_dim];
        if(!left)
            sum += dev_mat[idx-block_dim-1];
        if(!right)
            sum += dev_mat[idx-block_dim+1];
    }
    if(!down){
        sum += dev_mat[idx+block_dim];
        if(!left)
            sum += dev_mat[idx+block_dim-1];
        if(!right)
            sum += dev_mat[idx+block_dim+1];
    }
    if(!left)
        sum += dev_mat[idx-1];
    if(!right)
        sum += dev_mat[idx+1];
    return sum;
}



__global__ void game_iterations(int *dev_mat, int *dev_streak, int *dev_counter, int iterations, int dim){   
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x > dim || y > dim)
        return;

    int idx = x * dim + y;

    // add cooperative grid
    // cg::grid_group g = cg::this_grid();

    int sum;
    int curr=dev_mat[idx];
    int counter=curr;
    int streak=0;

    for(int i=0; i < iterations; i++){
        // board update
        sum = tot_neighbours(idx, dim, dev_mat);

        if(!curr && sum == 3){
            curr = 1;
            counter++;
        }
        else if (curr && (sum >= 4 || sum <= 1)){
            curr = 0;
        }
        else if(curr){
            counter++;
            streak++;
        } 
        __syncthreads();
        // g.sync();
        dev_mat[idx] = curr;
        __syncthreads();
        // g.sync();
    }

    dev_counter[idx] = counter;
    dev_streak[idx] = streak;

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
    FILE *fout = fopen("../output/original.txt", "w");
    int *counter;
    int *streak;

    // matrix allocation
    mat = (int *)malloc(n*n*sizeof(int));
    counter = (int *)malloc(n*n*sizeof(int));
    streak = (int *)malloc(n*n*sizeof(int));
    
    char c;
    for(int i=0; i < n; i++){
        for (int j=0; j < n; j++){
            // reading of the input file and initialization of the matrix
            c = fgetc(fin);
            while(c!='O' && c !='X')
                c = fgetc(fin);
            if(c == 'X')
                mat[n * i + j] = 1;
            else
                mat[n * i + j] = 0;
            fprintf(fout, "%d ", mat[n*i+j]);
        }
        fprintf(fout, "\n");
    }

    fclose(fout);

    // statistics array initialization into dev mem
    int *dev_counter;
    int *dev_streak;

    cudaMalloc((void **)&dev_counter, n * n * sizeof(int));
    cudaMalloc((void **)&dev_streak, n * n * sizeof(int));

    cudaMemset(dev_counter, 0x0, n * n * sizeof(int));
    cudaMemset(dev_streak, 0x0, n * n * sizeof(int));

    // copy input to device mem
    int *dev_mat;

    cudaMalloc((void**)&dev_mat, n * n * sizeof(int));
    cudaMemcpy(dev_mat, mat, n * n * sizeof(int), cudaMemcpyHostToDevice);

    // TODO: device block distribution
    int m = 32;
    dim3 blockSize(m, m, 1);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y, 1);

    // launch kernel on GPU
    // TODO: time measurement
    game_iterations<<<gridSize , blockSize>>>(dev_mat, dev_streak, dev_counter, iter, n);
    // Launch the kernel using cudaLaunchCooperativeKernel
    // void* kernelArgs[] = {(void*)dev_mat, (void*)dev_streak, (void*)dev_counter, (void*)&iter, (void*)&n};
    // cudaLaunchCooperativeKernel((void*)game_iterations, gridDim, blockDim, kernelArgs);
    
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


void printer(int *mat, int *counter, int *streak, int N){
    FILE *f_mat, *f_cnt, *f_streak;

    f_mat = fopen(OUTMAT, "w");
    printf("Printing final state of the board...\n");
    for(int i=0; i < N; i++){
        for(int j=0; j < N; j++){
            fprintf(f_mat, "%d ", mat[i*N+j]);
        }
        fprintf(f_mat, "\n");
    }

    f_cnt = fopen(OUTCNT, "w");
    printf("Printing overall count of alive generation for single cell...\n");
    for(int i=0; i < N; i++){
        for(int j=0; j < N; j++){
            fprintf(f_cnt, "%d ", counter[i*N+j]);
        }
        fprintf(f_cnt, "\n");
    }

    f_streak = fopen(OUTSTREAK, "w");
    printf("Printing maximum consecutive alive generations...\n");
    for(int i=0; i < N; i++){
        for(int j=0; j < N; j++){
            fprintf(f_streak, "%d ", streak[i*N+j]);
        }
        fprintf(f_streak, "\n");
    }

    fclose(f_mat);
    fclose(f_cnt);
    fclose(f_streak);
}