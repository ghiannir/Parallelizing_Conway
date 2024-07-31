#include <stdio.h>
#include <stdlib.h>

#define INFILE "../input/input.txt"
#define N 1000
#define ITER 500

void printer(int *mat, int *streak, int *counter){
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


int main(int argc, char *argv){
    printf("ciao");
    int mat[N*N];
    char c;
    FILE *fin = fopen(INFILE, "r");
    int counter[N*N];
    int streak[N*N];

    for(int i=0; i < N; i++){
        for (int j=0; j < N; j++){
            // reading of the input file and initialization of the matrix
            fscanf(fin, "%c%*c", &c);
            if(c == 'X')
                mat[N * i + j] = 1;
            else
                mat[N * i + j] = 0;
        }
    }

    printer(mat, counter, streak);

    fclose(fin);

    return 0;
}