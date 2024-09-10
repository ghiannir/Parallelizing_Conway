// Working implementation

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define INFILE "../input/input.txt"
#define OUTMAT "../output/mat.txt"
#define OUTCNT "../output/cnt.txt"
#define OUTSTREAK "../output/streak.txt"
#define STATS "../output/stats.csv"


__device__ int tot_neighbours(int idx, int block_dim, int *dev_mat){
    int sum = 0;

    // Cell coordinates
    int x = idx / block_dim;
    int y = idx % block_dim;

    for (int k = -1; k < 2; k++)
        for (int i = -1; i < 2; i++)
            if (x + k >= 0 && y + i >= 0 && x + k < block_dim  && y + i < block_dim && (k!=0 || i!=0)) 
                sum += dev_mat[block_dim * (x + k) + (y + i)];
    

    return sum;
}



__global__ void game_iterations(int *dev_mat, int *dev_streak, int *dev_counter, int *prev, int dim){   
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if(x > dim || y > dim)
        return;

    int idx = x * dim + y;

    int sum;
    int curr=dev_mat[idx];

    sum = tot_neighbours(idx, dim, prev);

    if (!curr && sum == 3) {
        curr = 1;
        dev_counter[idx]++;
    }
    else if (curr && (sum >= 4 || sum <= 1)) {
        curr = 0;
    }
    else if (curr) {
        dev_counter[idx]++;
        dev_streak[idx]++;
    }
    dev_mat[idx] = curr;
}

__global__ void update(int *dev_mat, int *prev, int dim) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > dim || y > dim)
        return;

    int idx = x * dim + y;

    prev[idx] = dev_mat[idx];
}


void printer(int *mat, int *streak, int *counter, int N);



int save_stats(int iterations, int table_size, float time, char * slurm_job_id) {
    FILE *file;

    file = fopen(STATS, "a");

    if (file == NULL) {
        printf("Error opening statistics file!\n");
        return 1;
    }

    fprintf(file, "%s,%d,%d,%.3f\n", slurm_job_id, iterations, table_size, time);

    fclose(file);
    return 0;
}



int main(int argc, char * argv[]) {

    if (argc != 2) {
        printf("Number of arguments passed is %d, but should be 2\n", argc);
        return 1;
    }

    int n;
    char *num_elements = getenv("N");
    sscanf(num_elements, "%d", &n);
    int iter;
    char *num_iter = getenv("ITER");
    sscanf(num_iter, "%d", &iter);

    int *mat;
    FILE *fin = fopen(INFILE, "r");
    // FILE *fout = fopen("../output/original.txt", "w");
    int *counter;
    int *streak;

    // matrix allocation
    mat = (int *) malloc(n * n * sizeof(int));
    counter = (int *) malloc(n * n * sizeof(int));
    streak = (int *) malloc(n * n * sizeof(int));
    
    // Initializing matrix
    int value;
    for (int i=0; i < n; i++) {
        for (int j=0; j < n; j++) {
            
            if (fscanf(fin, "%d", &value) == 1) {
                mat[n * i + j] = value;
            } else {
                printf("Error printing matrix at indexes (%d, %d)\n", i, j);
                return 1;
            }
            
        }
    }
    fclose(fin);


    // statistics array initialization into dev mem
    int *dev_counter;
    int *dev_streak;
    int *prev;

    cudaMalloc((void **) &dev_counter, n * n * sizeof(int));
    cudaMalloc((void **) &dev_streak, n * n * sizeof(int));
    cudaMalloc((void **) &prev, n * n * sizeof(int));

    cudaMemset(dev_counter, 0x0, n * n * sizeof(int));
    cudaMemset(dev_streak, 0x0, n * n * sizeof(int));
    cudaMemset(prev, 0x0, n * n * sizeof(int));

    // copy input to device mem
    int *dev_mat;

    cudaMalloc((void**) &dev_mat, n * n * sizeof(int));
    cudaMemcpy(dev_mat, mat, n * n * sizeof(int), cudaMemcpyHostToDevice);

    // TODO: device block distribution
    int m = 32;
    dim3 blockSize(m, m, 1);
    dim3 gridSize((n + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y, 1);

    // Setting up timer
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start); // create event objects
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    // launch kernel on GPU
    // TODO: time measurement
    for (int i=0; i < iter; i++) {
        update<<<gridSize, blockSize>>>(dev_mat, prev, n);

        game_iterations<<<gridSize , blockSize>>>(dev_mat, dev_streak, dev_counter, prev, n);
    }

    // Reading timer
    cudaEventRecord(stop, 0); // record end event
    cudaEventSynchronize(stop); // wait for all device work to complete
    cudaEventElapsedTime(&elapsedTime, start, stop); //time between events
    cudaEventDestroy(start); //destroy start event
    cudaEventDestroy(stop); 


    printf("Total execution time %f ms\n", elapsedTime);

    // gather results
	cudaMemcpy(mat, dev_mat, n * n * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(counter, dev_counter, n * n * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(streak, dev_streak, n * n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_counter);
    cudaFree(dev_mat);
    cudaFree(dev_streak);

    // print or save results
    printer(mat, counter, streak, n);

    free(mat);
    free(counter);
    free(streak);


    if (save_stats(iter, n, elapsedTime, argv[1]) != 0) {
        printf("Error saving stats\n");
    }

    return 0;
}


void printer(int *mat, int *counter, int *streak, int N){
    FILE *f_mat, *f_cnt, *f_streak;

    f_mat = fopen(OUTMAT, "w");
    printf("Printing final state of the board...\n");
    for (int i=0; i < N; i++) {
        for (int j=0; j < N; j++) {
            fprintf(f_mat, "%d ", mat[i*N+j]);
        }
        fprintf(f_mat, "\n");
    }

    f_cnt = fopen(OUTCNT, "w");
    printf("Printing overall count of alive generation for single cell...\n");
    for (int i=0; i < N; i++) {
        for (int j=0; j < N; j++) {
            fprintf(f_cnt, "%d ", counter[i*N+j]);
        }
        fprintf(f_cnt, "\n");
    }

    f_streak = fopen(OUTSTREAK, "w");
    printf("Printing maximum consecutive alive generations...\n");
    for (int i=0; i < N; i++) {
        for (int j=0; j < N; j++) {
            fprintf(f_streak, "%d ", streak[i*N+j]);
        }
        fprintf(f_streak, "\n");
    }

    fclose(f_mat);
    fclose(f_cnt);
    fclose(f_streak);
}
