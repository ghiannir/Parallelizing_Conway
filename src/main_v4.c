#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


#define INFILE "../input/input.txt"
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


void buildMatrix(int n, int * matrix, FILE * input_file, FILE * output_file) {
    for(int i=0; i < n; i++){
        for (int j=0; j < n; j++){
            // reading of the input file and initialization of the matrix
            char c = fgetc(input_file);
            while(c!='O' && c !='X')
                c = fgetc(input_file);
            if(c == 'X')
                matrix[n * i + j] = 1;
            else
                matrix[n * i + j] = 0;
            fprintf(output_file, "%d ", matrix[n*i+j]);
        }
        fprintf(output_file, "\n");
    }
}


int main(void){
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

    buildMatrix(n , mat, fopen(INFILE, "r"), fopen(OUTFILE, "w"));

    printf("Max number of threads: %d\n", omp_get_max_threads());

    int n_threads = 24;
    omp_set_num_threads(n_threads);



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

    printer(mat, counter, streak, n);

    free(mat);
    free(counter);
    free(streak);
    free(prev);

    return 0;
}


