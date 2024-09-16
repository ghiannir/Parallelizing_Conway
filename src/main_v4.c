#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>


#define INFILE "../input/input.txt"
#define OUTFILE "../output/original.txt"
#define OUTMAT "../output/mat.txt"
#define OUTCNT "../output/cnt.txt"
#define OUTSTREAK "../output/streak.txt"
#define STATS "../output/stats_openmp.csv"

int tot_neighbours(int idx, int block_dim, int *dev_mat){
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


void buildMatrix(int n, int * matrix, FILE * input_file) {
    int value;
    for (int i=0; i < n; i++) {
        for (int j=0; j < n; j++) {
            
            if (fscanf(input_file, "%d", &value) == 1) {
                matrix[n * i + j] = value;
            } else {
                printf("Error printing matrix at indexes (%d, %d)\n", i, j);
            }
            
        }
    }
}


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


    int *counter;
    int *streak;
    char c;
    int *mat;
    int *prev;
    mat = (int *)malloc(n*n*sizeof(int));
    prev = (int *)malloc(n*n*sizeof(int));
    counter = (int *)malloc(n*n*sizeof(int));
    streak = (int *)malloc(n*n*sizeof(int));

    buildMatrix(n , mat, fopen(INFILE, "r"));

    printf("Max number of threads: %d\n", omp_get_max_threads());

    int n_threads = 16;
    omp_set_num_threads(n_threads);

    clock_t begin = clock();

    for (int i = 0; i < iter; i++) {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < n * n; j++) {
            prev[j] = mat[j];
        }

        #pragma omp parallel for schedule(static)
        for (int j = 0; j < n * n; j++) {
            int sum;
            int curr=mat[j];

            sum = tot_neighbours(j, n, prev);

            if(!curr && sum == 3){
                curr = 1;
                counter[j]++;
            }
            else if (curr && (sum >= 4 || sum <= 1)){
                curr = 0;
            }
            else if(curr){
                counter[j]++;
                streak[j]++;
            }
            mat[j] = curr;
        }
    }

    clock_t end = clock();
    double time_spent = (double)(end - begin) * 1000 / CLOCKS_PER_SEC;


    printer(mat, counter, streak, n);

    if (save_stats(iter, n, time_spent, argv[1]) != 0) {
        printf("Error saving stats\n");
    }

    free(mat);
    free(counter);
    free(streak);
    free(prev);

    return 0;
}


