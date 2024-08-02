#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INFILE "../input/input.txt"
#define OUTMAT "../output/mat_seq.txt"
#define OUTCNT "../output/cnt_seq.txt"
#define OUTSTREAK "../output/streak_seq.txt"
// #define N 1000
// #define ITER 500

int tot_neighbours(int idx, int block_dim, int *dev_mat);


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
    int *prev = (int *)malloc(n*n*sizeof(int));

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
            prev[n*i+j] = mat[n*i+j];
            counter[n*i+j] = mat[n*i+j];
            streak[n*i+j] = 0;
        }

    }

    int sum, idx, curr;

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