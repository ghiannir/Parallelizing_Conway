#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define INFILE "../input/input_omp.txt"
#define OUTFILE "../output/original.txt"
#define OUTMAT "../output/mat.txt"
#define OUTCNT "../output/cnt.txt"
#define OUTSTREAK "../output/streak.txt"

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