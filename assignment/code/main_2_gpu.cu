#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <omp.h>

#define INFILE "../input/input.txt"
#define OUTMAT "../output/mat.txt"
#define OUTCNT "../output/cnt.txt"
#define OUTSTREAK "../output/streak.txt"
#define STATS "../output/stats_cuda_multi_gpu.csv"


struct Error {
    int code;
    const char* reason;
};


__device__ int tot_neighbours(int idx, int dim, int *dev_mat, int gpu_index) {
    int sum = 0;

    int y = (int) idx / dim + gpu_index;
    int x = idx % dim;

    for (int k = -1; k <= 1; k++)
        for (int i = -1; i <= 1; i++)
            if (x + i >= 0 && y + k >= 0 && x + i < dim  && y + k < (dim/2 + 1) && (k!=0 || i!=0))
                sum += dev_mat[dim * (y + k) + (x + i)];
            
    return sum;
}



__global__ void game_iterations(int *dev_mat, int *dev_streak, int *dev_counter, int *prev, int dim, int gpu_index){   
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dim || y >= dim/2) return;
    
    int idx_with_halo = (y + gpu_index) * dim + x;
    int idx = y * dim + x;

    int sum;
  
    int current_cell = prev[idx_with_halo];

    sum = tot_neighbours(idx, dim, prev, gpu_index);

    if (!current_cell && sum == 3) {
        current_cell = 1;
        dev_counter[idx]++;
    }
    else if (current_cell && (sum >= 4 || sum <= 1)) {
        current_cell = 0;
    }
    else if (current_cell) {
        dev_counter[idx]++;
        dev_streak[idx]++;
    }
    dev_mat[idx_with_halo] = current_cell;

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

// Function used for debugging
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

void matrix_printer(int * matrix, int dim_x, int dim_y) {
    printf("----------- PRINTING MATRIX ------------\n");
    for (int i = 0; i< dim_y; i++) {
        for (int j = 0; j< dim_x; j++) {
            printf("%d ", matrix[i*dim_x + j]);
        }
        printf("\n");
    }
}


struct Error run_cuda_program(
                    int* matrix,
                    int* counter, 
                    int* streak,
                    int n,
                    int iter,
                    float * elapsedTimePtr) {

    int num_gpus_cluster;
    cudaGetDeviceCount(&num_gpus_cluster);

    if (num_gpus_cluster < 2) {
        return { .code = 1, .reason = "There are not 2 GPU's available" };
    }
    
    printf("Found at least 2 GPU's\n");
    

    int *dev_matrix[2];
    int *dev_counter[2];
    int *dev_streak[2];
    int *previous_matrix[2];


    int m = 32;
    dim3 blockDim(m, m, 1);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, ((n/2 + 1) + blockDim.y - 1) / blockDim.y);

    // Setting up timer
    cudaEvent_t start, stop;

    

    // launch kernel on GPU
    #pragma omp parallel firstprivate(iter) shared(start, stop, elapsedTimePtr)
    {
        int tid = omp_get_thread_num();

		gpuErrchk(cudaSetDevice(tid));
        gpuErrchk(cudaMalloc((void **) &dev_counter[tid], n * (n/2) * sizeof(int)));
        gpuErrchk(cudaMalloc((void **) &dev_streak[tid], n * (n/2) * sizeof(int)));
        gpuErrchk(cudaMalloc((void **) &previous_matrix[tid], n * (n/2 + 1) * sizeof(int)));
        gpuErrchk(cudaMalloc((void**) &dev_matrix[tid], n * (n/2 + 1) * sizeof(int)));
        gpuErrchk(cudaMemset(dev_counter[tid], 0x0, n * (n/2) * sizeof(int)));
        gpuErrchk(cudaMemset(dev_streak[tid], 0x0, n * (n/2) * sizeof(int)));
        gpuErrchk(cudaMemcpy(previous_matrix[tid], matrix + tid*n*(n/2 - 1), n * (n/2 + 1) * sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(dev_matrix[tid], matrix + tid*n*(n/2 - 1), n * (n/2 + 1) * sizeof(int), cudaMemcpyHostToDevice));
        
        
        if (tid == 0) {
            gpuErrchk(cudaEventCreate(&start));
            gpuErrchk(cudaEventCreate(&stop));
            gpuErrchk(cudaEventRecord(start));
        }


        for (int i=0; i < iter; i++) {
            
            game_iterations<<<gridDim , blockDim>>>(dev_matrix[tid], dev_streak[tid], dev_counter[tid], previous_matrix[tid], n, tid);
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
            
            int *temp_matrix = dev_matrix[tid];
			dev_matrix[tid] = previous_matrix[tid];
			previous_matrix[tid] = temp_matrix;
            
            #pragma omp barrier
            if (tid == 0) {
                gpuErrchk(cudaMemcpyPeer(previous_matrix[1],1, previous_matrix[0] + n*(n/2 - 1), 0, n * sizeof(int)));
                gpuErrchk(cudaMemcpyPeer(previous_matrix[0] + n*(n/2), 0, previous_matrix[1] + n, 1, n * sizeof(int)));
            }
            #pragma omp barrier

        }

        if (tid == 0) {
            gpuErrchk(cudaEventRecord(stop));
            gpuErrchk(cudaEventSynchronize(stop));
            gpuErrchk(cudaEventElapsedTime(elapsedTimePtr, start, stop));
            gpuErrchk(cudaEventDestroy(start));
            gpuErrchk(cudaEventDestroy(stop)); 
    
        }

        
        gpuErrchk(cudaMemcpy(matrix + n*(n/2)*tid, previous_matrix[tid] + n*tid, n * (n/2) * sizeof(int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(counter + n*(n/2)*tid, dev_counter[tid], n * (n/2) * sizeof(int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaMemcpy(streak + n*(n/2)*tid, dev_streak[tid], n * (n/2) * sizeof(int), cudaMemcpyDeviceToHost));
        gpuErrchk(cudaFree(dev_counter[tid]));
        gpuErrchk(cudaFree(dev_matrix[tid]));
        gpuErrchk(cudaFree(dev_streak[tid]));
        gpuErrchk(cudaFree(previous_matrix[tid]));
    }
    
    

    printf("Total execution time %f ms\n", *elapsedTimePtr);

    return {.code = 0, .reason = ""};
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

    float elapsedTime = 0;
    
    omp_set_num_threads(2);
    struct Error error = run_cuda_program(mat, counter, streak, n, iter, &elapsedTime);

    if (error.code != 0) {
        printf("Error running CUDA problem with reason: %s\n", error.reason);
        return 1;
    }
    printf("Total execution time with 2 GPU %f ms\n", elapsedTime);
    // print or save results
    printer(mat, counter, streak, n);

    free(mat);
    free(counter);
    free(streak);


    if (save_stats(iter, n, elapsedTime, argv[1]) != 0) {
        printf("Error saving stats\n");
        return 1;
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
