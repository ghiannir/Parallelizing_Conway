#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define INFILE "../input/input_omp.txt"
#define OUTFILE "../output/original.txt"
#define OUTMAT "../output/mat.txt"
#define OUTCNT "../output/cnt.txt"
#define OUTSTREAK "../output/streak.txt"

// TODO: farla piu leggibile, si possono togliere gli if inserendo solo somme (i campi nelle celle sono 0 o 1)
int tot_neighbours(int idx, int size, int *table){
    int sum = 0;

    // flags for border cells
    int left=0, right=0, up=0, down=0;
    
    if((idx % size) == 0)
        left = 1; 
    else if((idx + 1) % size == 0)
        right = 1;

    if((idx - size) < 0)
        up = 1;
    else if((idx + size) >= (size * size))
        down=1;

    // sum all existing nearby blocks vlaues
    if(!up){
        sum += table[idx - size];
        if(!left)
            sum += table[idx - size - 1];
        if(!right)
            sum += table[idx - size + 1];
    }
    if(!down){
        sum += table[idx + size];
        if(!left)
            sum += table[idx + size - 1];
        if(!right)
            sum += table[idx + size + 1];
    }
    if(!left)
        sum += table[idx - 1];
    if(!right)
        sum += table[idx + 1];
    return sum;
}


// void printer(int *mat, int *counter, int *streak, int N){
//     FILE *f_mat, *f_cnt, *f_streak;

//     f_mat = fopen(OUTMAT, "w");
//     if (f_mat == NULL) {
//         printf("Error opening file %s\n", OUTMAT);
//         return;
//     }
//     printf("Printing final state of the board...\n");
//     for(int i = 0; i < N; i++){
//         for(int j = 0; j < N; j++){
//             fprintf(f_mat, "%d ", mat[i*N+j]);
//         }
//         fprintf(f_mat, "\n");
//     }
//     fclose(f_mat);
//     printf("Finished printing final state of the board.\n");

//     f_cnt = fopen(OUTCNT, "w");
//     if (f_cnt == NULL) {
//         printf("Error opening file %s\n", OUTCNT);
//         return;
//     }
//     printf("Printing overall count of alive generation for single cell...\n");
//     for(int i = 0; i < N; i++){
//         for(int j = 0; j < N; j++){
//             fprintf(f_cnt, "%d ", counter[i*N+j]);
//         }
//         fprintf(f_cnt, "\n");
//     }
//     fclose(f_cnt);
//     printf("Finished printing overall count of alive generation for single cell.\n");

//     f_streak = fopen(OUTSTREAK, "w");
//     if (f_streak == NULL) {
//         printf("Error opening file %s\n", OUTSTREAK);
//         return;
//     }
//     printf("Printing maximum consecutive alive generations...\n");
//     for(int i = 0; i < N; i++){
//         for(int j = 0; j < N; j++){
//             fprintf(f_streak, "%d ", streak[i*N+j]);
//         }
//         fprintf(f_streak, "\n");
//     }
//     fclose(f_streak);
//     printf("Finished printing maximum consecutive alive generations.\n");
// }


void buildMatrix(int n, int * matrix, FILE * input_file) {
    for(int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            // reading of the input file and initialization of the matrix
            char c = fgetc(input_file);
            while(c != '0' && c != '1')
                c = fgetc(input_file);
            if(c == '1')
                matrix[n * i + j] = 1;
            else
                matrix[n * i + j] = 0;
            // fprintf(output_file, "%d ", matrix[n*i+j]);
        }
        // fprintf(output_file, "\n");
    }
}


int main(int argc, char *argv[]){
    // printf("Entering main function...\n");

    if (argc != 5) {
        printf("Usage: %s <N> <ITER> <n_threads> <output_file>\n", argv[0]);
        return 1;
    }
    
    int n = atoi(argv[1]);
    int iter = atoi(argv[2]);
    int n_threads = atoi(argv[3]);

    // printf("Arguments received: N=%d, ITER=%d, n_threads=%d\n", n, iter, n_threads);

    int *counter;
    int *streak;
    int *mat;
    int *prev;

    mat = (int *)malloc(n*n*sizeof(int));
    prev = (int *)malloc(n*n*sizeof(int));
    counter = (int *)malloc(n*n*sizeof(int));
    streak = (int *)malloc(n*n*sizeof(int));

    if (mat == NULL || prev == NULL || counter == NULL || streak == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    // printf("Building matrix...\n");
    buildMatrix(n , mat, fopen(INFILE, "r"));

    // printf("Max number of threads: %d\n", omp_get_max_threads());

    omp_set_num_threads(n_threads);

    double start_time = omp_get_wtime();
    for (int i = 0; i < iter; i++) {
        #pragma omp parallel for schedule(static)
        for (int j = 0; j < n * n; j++) {
            prev[j] = mat[j];
        }

        #pragma omp parallel for schedule(static)
        for (int j = 0; j < n * n; j++) {
            int sum;
            int curr = mat[j];

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
    double end_time = omp_get_wtime();

    FILE *pcsv = fopen(argv[4], "a");
    fprintf(pcsv, "\n%d, %d, %d, %f", n, iter, n_threads, end_time-start_time);

    free(mat);
    free(counter);
    free(streak);
    free(prev);


    return 0;
}