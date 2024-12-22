#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

#define INFILE "../input/input.txt"
#define OUTFILE "../output/original.txt"
#define OUTMAT "../output/mat_mpi.txt"
#define OUTCNT "../output/cnt_mpi.txt"
#define OUTSTREAK "../output/streak_mpi.txt"
#define DEBUG

// TODO: farla piu leggibile, si possono togliere gli if inserendo solo somme (i campi nelle celle sono 0 o 1)
int tot_neighbours(int i, int size, int *table){
    int sum = 0;
    int idx = i+size;
    // flags for border cells
    int left=0, right=0;
    
    if((idx % size) == 0)
        left = 1; 
    if((idx + 1) % size == 0)
        right = 1;

    sum += table[idx-size];
    sum += table[idx+size];
    if(!left){
        sum += table[idx-1];
        sum += table[idx-size-1];
        sum += table[idx+size-1];
    }
    if(!right){
        sum += table[idx+1];
        sum += table[idx-size+1];
        sum += table[idx+size+1];
    }
    return sum;
}


void printer(int *mat, int *counter, int *streak, int N, int R){
    FILE *f_mat, *f_cnt, *f_streak;

    f_mat = fopen(OUTMAT, "a");
    if (f_mat == NULL) {
        printf("Error opening file %s\n", OUTMAT);
        return;
    }
    // printf("Printing final state of the board...\n");
    for(int i = 0; i < R; i++){
        for(int j = 0; j < N; j++){
            fprintf(f_mat, "%d ", mat[i*N+j]);
        }
        fprintf(f_mat, "\n");
    }
    fclose(f_mat);
    // printf("Finished printing final state of the board.\n");

    f_cnt = fopen(OUTCNT, "a");
    if (f_cnt == NULL) {
        printf("Error opening file %s\n", OUTCNT);
        return;
    }
    // printf("Printing overall count of alive generation for single cell...\n");
    for(int i = 0; i < R; i++){
        for(int j = 0; j < N; j++){
            fprintf(f_cnt, "%d ", counter[i*N+j]);
        }
        fprintf(f_cnt, "\n");
    }
    fclose(f_cnt);
    // printf("Finished printing overall count of alive generation for single cell.\n");

    f_streak = fopen(OUTSTREAK, "a");
    if (f_streak == NULL) {
        printf("Error opening file %s\n", OUTSTREAK);
        return;
    }
    // printf("Printing maximum consecutive alive generations...\n");
    for(int i = 0; i < R; i++){
        for(int j = 0; j < N; j++){
            fprintf(f_streak, "%d ", streak[i*N+j]);
        }
        fprintf(f_streak, "\n");
    }
    fclose(f_streak);
    // printf("Finished printing maximum consecutive alive generations.\n");
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

void game_of_life(int n, int *prev, int *mat, int *counter, int *streak, int rows_per_process){

    int idx, sum, curr;
    
    for(int j=0; j < rows_per_process; j++){
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


int main(int argc, char *argv[]){
    // printf("Entering main function...\n");
    MPI_Status Stat;

    MPI_Init(&argc,&argv);
    int i, rank, numtasks;
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#ifndef DEBUG
    if (argc != 4) {
        printf("Usage: %s <N> <ITER> <output_file>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
#else
    if (argc != 3) {
        printf("Usage: %s <N> <ITER>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
#endif
    
    int n = atoi(argv[1]);
    int iter = atoi(argv[2]);

    int rows_per_process = n / numtasks;
    int extra_rows = n % numtasks;

    if (rank < extra_rows) {
        rows_per_process++;
    }
    // printf("Rank %d has %d rows\n", rank, rows_per_process);
    // allocate memory for local matrices
    int *local_mat = (int *)malloc((rows_per_process) * n * sizeof(int));
    int *local_prev = (int *)malloc((rows_per_process+2) * n * sizeof(int));
    int *local_counter = (int *)malloc((rows_per_process) * n * sizeof(int));
    int *local_streak = (int *)malloc((rows_per_process) * n * sizeof(int));

    if (local_mat == NULL || local_prev == NULL || local_counter == NULL || local_streak == NULL) {
        printf("Memory allocation failed.\n");
        MPI_Finalize();
        return 1;
    }

    // initialize local matrices to 0
    memset(local_counter, 0, (rows_per_process) * n * sizeof(int));
    memset(local_streak, 0, (rows_per_process) * n * sizeof(int));
    memset(local_prev, 0, (rows_per_process+2) * n * sizeof(int));
    memset(local_mat, 0, (rows_per_process) * n * sizeof(int));

    int *counter;
    int *streak;
    int *mat;

    if (rank==0){
        
        mat = (int *)malloc(n*n*sizeof(int));
        counter = (int *)malloc(n*n*sizeof(int)); 
        streak = (int *)malloc(n*n*sizeof(int));
        memset(counter, 0, n * n * sizeof(int));
        memset(streak, 0, n * n * sizeof(int));
        memset(mat, 0, n * n * sizeof(int));

        if (mat == NULL) {
            printf("Memory allocation failed.\n");
            MPI_Finalize();
            return 1;
        }

        buildMatrix(n , mat, fopen(INFILE, "r"));
        // printf("Here matrix has dimensions %d, with n=%d\n", sizeof(mat), n);

        // memcpy(counter, mat, n*n*sizeof(int));
        // memset(streak, 0, n*n*sizeof(int));

        // initialize local matrices
        memcpy(local_mat, mat, (rows_per_process) * n * sizeof(int));
        memcpy(local_counter, mat, (rows_per_process) * n * sizeof(int));
        memset(local_streak, 0, (rows_per_process) * n * sizeof(int));
        

        // sendo local matrices of other processes
        int offset = 0;
        for (int r=1; r<numtasks; r++) {
            int rows = (r-1 < extra_rows) ? rows_per_process : rows_per_process - 1;
            if (extra_rows == 0) {
                rows++;
            }
            offset += rows*n;
            rows = (r < extra_rows) ? rows_per_process : rows_per_process - 1;
            if (extra_rows == 0) {
                rows++;
            }
            MPI_Send(&mat[offset], rows * n, MPI_INT, r, 0, MPI_COMM_WORLD);
        }
    }
    if(rank!=0) {
        MPI_Recv(local_mat, rows_per_process * n, MPI_INT, 0, 0, MPI_COMM_WORLD, &Stat);
        // update local counter
        memcpy(local_counter, local_mat, (rows_per_process) * n * sizeof(int));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    // start iterations
    double start_time = MPI_Wtime();
    for(int i=0; i<iter; i++){
        // update local matrix
        memcpy(local_prev+n, local_mat, (rows_per_process) * n * sizeof(int));
        //  exchange ghost cells
        if(rank==numtasks-1){
            MPI_Send(&local_mat[0], n, MPI_INT, rank-1, 0, MPI_COMM_WORLD);
            // printf("Process %d receiving...", rank);
            MPI_Recv(&local_prev[0], n, MPI_INT, rank-1, 0, MPI_COMM_WORLD, &Stat);
        }
        else if(rank==0){
            MPI_Send(&local_mat[(rows_per_process-1) * n], n, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
            // printf("Process %d receiving...", rank);
            MPI_Recv(&local_prev[(rows_per_process+1) * n], n, MPI_INT, rank+1, 0, MPI_COMM_WORLD, &Stat);
        }
        else{
            MPI_Send(&local_mat[0], n, MPI_INT, rank-1, 0, MPI_COMM_WORLD);
            MPI_Send(&local_mat[(rows_per_process-1) * n], n, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
            // printf("Process %d receiving...", rank);
            MPI_Recv(&local_prev[0], n, MPI_INT, rank-1, 0, MPI_COMM_WORLD, &Stat);
            MPI_Recv(&local_prev[(rows_per_process+1) * n], n, MPI_INT, rank+1, 0, MPI_COMM_WORLD, &Stat);
            
        }
        printf("Rank %d local_prev matrix\n", rank, rows_per_process);
        char fn[20];
        sprintf(fn, "out_mat%d.txt", rank);
        FILE *fout = fopen(fn, "w");
        for (int i = 0; i < rows_per_process+2; i++) {
            for (int j = 0; j < n; j++) {
                fprintf(fout, "%d ", local_prev[i*n+j]);
            }
            fprintf(fout, "\n");
        }
        fclose(fout);
        // printf("\nRank %d received initial row:", rank);
        // for (int i = 0; i < n; i++) {
        //     printf("%d ", local_prev[i]);
        // }
        // printf("\n");
        // printf("\nRank %d received final row:", rank);
        // for (int i = 0; i < n; i++) {
        //     printf("%d ", local_prev[i+(rows_per_process+1)*n]);
        // }
        // printf("\n");

        // update local matrix
        game_of_life(n, local_prev, local_mat, local_counter, local_streak, rows_per_process);

        MPI_Barrier(MPI_COMM_WORLD);
    }
    double end_time = MPI_Wtime();
#ifndef DEBUG
    int *p = malloc(sizeof(int));
    *p = 1;
    if (rank==0){
        printer(local_mat, local_counter, local_streak, n, rows_per_process);
        MPI_Send(p, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    }
    else{
        MPI_Recv(p, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, &Stat);
        printer(local_mat, local_counter, local_streak, n, rows_per_process);
        if(rank!=numtasks-1)
            MPI_Send(p, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
    }
    free(p);
#endif
    
    int offset = 0;
    int total_size = 0;

    int *recvcounts = (int*)malloc(numtasks * sizeof(int));
    int *displs = (int*)malloc(numtasks * sizeof(int));

    for (int i = 0; i < numtasks; i++) {
        int rows = (i < extra_rows) ? rows_per_process : rows_per_process - 1;
        if (extra_rows == 0) {
                rows++;
            }
        recvcounts[i] = rows*n;
        displs[i] = offset;
        offset += recvcounts[i];
        total_size += recvcounts[i];
    }

    // Alloca i buffer di ricezione con la dimensione corretta
    // if (rank == 0) {
    //     mat = (int*)malloc(total_size * sizeof(int));
    //     counter = (int*)malloc(total_size * sizeof(int));
    //     streak = (int*)malloc(total_size * sizeof(int));
    //     memset(counter, 0, n * n * sizeof(int));
    //     memset(streak, 0, n * n * sizeof(int));
    //     memset(mat, 0, n * n * sizeof(int));
    // }

    //Utilizza MPI_Gatherv per raccogliere i dati
    // if (rank==0){
    //     mat = (int *)malloc(n*n*sizeof(int));
    //     counter = (int *)malloc(n*n*sizeof(int)); 
    //     streak = (int *)malloc(n*n*sizeof(int));
        
    //     for(int i=0; i<n; i++){
    //         for(int j=0; j < n; j++){
    //             printf("%d ", mat[i*n+j]);
    //         }
    //         printf("\n");
    //     }
    // }
    printf("Rank %d is gathering with %d rows\n", rank, rows_per_process);
    // MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gatherv(&local_mat[0], rows_per_process * n, MPI_INT, &mat[0], recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gatherv(&local_counter[0], rows_per_process * n, MPI_INT, &counter[0], recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gatherv(&local_streak[0], rows_per_process * n, MPI_INT, &streak[0], recvcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Gather(&local_mat[0], rows_per_process * n, MPI_INT, &mat[0], rows_per_process * n, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Gather(&local_counter[0], rows_per_process * n, MPI_INT, &counter[0], rows_per_process * n, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Gather(&local_streak[0], rows_per_process * n, MPI_INT, &streak[0], rows_per_process * n, MPI_INT, 0, MPI_COMM_WORLD);
    // int *p = malloc(sizeof(int));
    // *p = 1;
    // MPI_Request request1, request2, request3;
    // MPI_Barrier(MPI_COMM_WORLD);
    // if (rank==0){
    //     MPI_Igather(&local_mat[0], rows_per_process * n, MPI_INT, &mat[0], rows_per_process * n, MPI_INT, 0, MPI_COMM_WORLD, &request1);
    //     MPI_Igather(&local_counter[0], rows_per_process * n, MPI_INT, &counter[0], rows_per_process * n, MPI_INT, 0, MPI_COMM_WORLD, &request2);
    //     MPI_Igather(&local_streak[0], rows_per_process * n, MPI_INT, &streak[0], rows_per_process * n, MPI_INT, 0, MPI_COMM_WORLD, &request3);
    //     MPI_Send(p, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
    //     MPI_Wait(&request1, MPI_STATUS_IGNORE);
    //     MPI_Wait(&request2, MPI_STATUS_IGNORE);
    //     MPI_Wait(&request3, MPI_STATUS_IGNORE);
    // }
    // else{
    //     MPI_Recv(p, 1, MPI_INT, rank-1, 0, MPI_COMM_WORLD, &Stat);
    //     MPI_Igather(&local_mat[0], rows_per_process * n, MPI_INT, &mat[0], rows_per_process * n, MPI_INT, 0, MPI_COMM_WORLD, &request1);
    //     MPI_Igather(&local_counter[0], rows_per_process * n, MPI_INT, &counter[0], rows_per_process * n, MPI_INT, 0, MPI_COMM_WORLD, &request2);
    //     MPI_Igather(&local_streak[0], rows_per_process * n, MPI_INT, &streak[0], rows_per_process * n, MPI_INT, 0, MPI_COMM_WORLD, &request3);
    //     if(rank!=numtasks-1)
    //         MPI_Send(p, 1, MPI_INT, rank+1, 0, MPI_COMM_WORLD);
    // }
    // MPI_Barrier(MPI_COMM_WORLD);
    // free(p);

    // // Libera la memoria allocata
    // free(recvcounts);
    // free(displs);

    // gather local matrices
    // if(rank==0){
    //     memcpy(&mat[0], local_mat, (rows_per_process) * n * sizeof(int));
    //     memcpy(&counter[0], local_counter, (rows_per_process) * n * sizeof(int));
    //     memcpy(&streak[0], local_streak, (rows_per_process) * n * sizeof(int));
    //     // printer(mat, counter, streak, n);
    //     printf("Here matrix has dimensions %d", sizeof(mat));
    //     int m_offset = (rows_per_process) * n;
    //     int c_offset = (rows_per_process) * n;
    //     int s_offset = (rows_per_process) * n;
    //     printf("0 is gathering...\n");
    //     for (int r = 1; r < numtasks; r++) {
    //         int rows = (r < extra_rows) ? rows_per_process : rows_per_process - 1;
    //         printf("%d is sending...\n", r);
            
    //         // Ricevi m
    //         printf("Receiving m from %d...space %ld\n", r, sizeof(mat));
    //         MPI_Recv(&mat[m_offset], rows * n, MPI_INT, r, 0, MPI_COMM_WORLD, &Stat);
    //         int count;
    //         MPI_Get_count(&Stat, MPI_INT, &count);
    //         if (count != rows * n) {
    //             printf("Error: received message size %d does not match expected size %d for m\n", count, rows * n);
    //         } else {
    //             printf("Received m from %d successfully.\n", r);
    //         }
    //         m_offset += rows * n;

    //         // Ricevi c
    //         printf("Receiving c from %d...space %ld\n", r, sizeof(counter));
    //         MPI_Recv(&counter[c_offset], rows * n, MPI_INT, r, 0, MPI_COMM_WORLD, &Stat);
    //         MPI_Get_count(&Stat, MPI_INT, &count);
    //         if (count != rows * n) {
    //             printf("Error: received message size %d does not match expected size %d for c\n", count, rows * n);
    //         } else {
    //             printf("Received c from %d successfully.\n", r);
    //         }
    //         c_offset += rows * n;

    //         // Ricevi s
    //         printf("Receiving s from %d...space %ld\n", r, sizeof(streak));
    //         MPI_Recv(&streak[s_offset], rows * n, MPI_INT, r, 0, MPI_COMM_WORLD, &Stat);
    //         MPI_Get_count(&Stat, MPI_INT, &count);
    //         if (count != rows * n) {
    //             printf("Error: received message size %d does not match expected size %d for s\n", count, rows * n);
    //         } else {
    //             printf("Received s from %d successfully.\n", r);
    //         }
    //         s_offset += rows * n;
    //     }
    // }
    if (rank==0){
#ifdef DEBUG
        printer(mat, counter, streak, n, n);
#endif
        free(mat);
        free(counter);
        free(streak);
    }
    // else {
    //     printf("Sending data from %d...\n", rank);
    //     MPI_Send(&local_mat[0], rows_per_process * n, MPI_INT, 0, 0, MPI_COMM_WORLD);
    //     printf("Sent local_mat from %d.\n", rank);
    //     MPI_Send(&local_counter[0], rows_per_process * n, MPI_INT, 0, 0, MPI_COMM_WORLD);
    //     printf("Sent local_counter from %d.\n", rank);
    //     MPI_Send(&local_streak[0], rows_per_process * n, MPI_INT, 0, 0, MPI_COMM_WORLD);
    //     printf("Sent local_streak from %d.\n", rank);
    // }
    MPI_Barrier(MPI_COMM_WORLD); 
    free(local_mat);
    free(local_counter);
    free(local_streak);
    free(local_prev);
    

    if (rank==0){
        printf("Elapsed time %f", end_time-start_time);
    }

    MPI_Finalize();
#ifndef DEBUG
    FILE *pcsv = fopen(argv[4], "a");
    fprintf(pcsv, "\n%d, %d, %d, %f", n, iter, n_threads, end_time-start_time);
    fclose(pcsv);
#endif

    return 0;
}