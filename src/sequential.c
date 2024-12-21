#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define INFILE "../input/input.txt"
#define OUTMAT "../output/mat_seq.txt"
#define OUTCNT "../output/cnt_seq.txt"
#define OUTSTREAK "../output/streak_seq.txt"
#define STATS "../output/stats_sequential.csv"
// #define N 1000
// #define ITER 500

int tot_neighbours(int idx, int block_dim, int *dev_mat);


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

int main(int argc, char * argv[]){

    if (argc != 3) {
        printf("Number of arguments passed is %d, but should be 2\n", argc);
        return 1;
    }

    int n = atoi(argv[1]);
    int iter = atoi(argv[2]);

    int *mat;
    FILE *fin = fopen(INFILE, "r");
    int *counter;
    int *streak;
    // matrix allocation
    mat = (int *)malloc(n*n*sizeof(int));
    counter = (int *)malloc(n*n*sizeof(int));
    streak = (int *)malloc(n*n*sizeof(int));
    int *prev = (int *)malloc(n*n*sizeof(int));


    int value;
    for (int i=0; i < n; i++) {
        for (int j=0; j < n; j++) {
            
            if (fscanf(fin, "%d", &value) == 1) {
                mat[n * i + j] = value;
            } else {
                printf("Error printing matrix at indexes (%d, %d)\n", i, j);
            }
            prev[n*i+j] = mat[n*i+j];
            counter[n*i+j] = mat[n*i+j];
            streak[n*i+j] = 0;
        }
    }
    fclose(fin);

    int sum, idx, curr;
    clock_t begin = clock();
    
    for(int i=0; i < iter; i++){
        memcpy(prev, mat, n*n*sizeof(int));        
        // board update
        for(int j=0; j < n; j++){
            for(int z=0; z < n; z++){
                idx = j*n+z;

                sum = tot_neighbours(idx, n, prev);

                curr = mat[idx];

                if(!curr && sum == 3){
                    curr = 1;
                    counter[idx]++;
                }
                else if (curr && (sum >= 4 || sum <= 1)){
                    curr = 0;
                }
                else if(curr){
                    counter[idx]++;
                    streak[idx]++;
                }
                mat[idx] = curr;
            }
        }
    }

    clock_t end = clock();
    double time_spent = (double)(end - begin) * 1000 / CLOCKS_PER_SEC;


    // print or save results
    printer(mat, counter, streak, n);

    free(mat);
    free(counter);
    free(streak);
    free(prev);

    // if (save_stats(iter, n, time_spent, argv[1]) != 0) {
    //     printf("Error saving stats\n");
    // }

    return 0;
}


void printer(int *mat, int *counter, int *streak, int N){
    FILE *f_mat, *f_cnt, *f_streak;

    f_mat = fopen(OUTMAT, "w");
    // printf("Printing final state of the board...\n");
    for(int i=0; i < N; i++){
        for(int j=0; j < N; j++){
            fprintf(f_mat, "%d ", mat[i*N+j]);
        }
        fprintf(f_mat, "\n");
    }

    f_cnt = fopen(OUTCNT, "w");
    // printf("Printing overall count of alive generation for single cell...\n");
    for(int i=0; i < N; i++){
        for(int j=0; j < N; j++){
            fprintf(f_cnt, "%d ", counter[i*N+j]);
        }
        fprintf(f_cnt, "\n");
    }

    f_streak = fopen(OUTSTREAK, "w");
    // printf("Printing maximum consecutive alive generations...\n");
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


int tot_neighbours(int idx, int block_dim, int *dev_mat){
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