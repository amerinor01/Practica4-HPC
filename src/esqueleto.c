#include <stdio.h>
#include "cblas.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "memoryfun.h"
#include <mpi.h>
#include <time.h>

#define SEED 2022
#define TAG 22

#define ALPHA 1
#define BETA 0

#define TO_SECONDS 1000000
#define ROOT 0

#define DEBUG 0

int main(int argc, char **argv){

int i,j,l;              /* Indices para bucles */
int n;                  /* Tamanyo de las matrices */
int np,mid;             /* numero de procesos (np), identificador del proceso (mid) */
int nlocal;             /* columnas que corresponden a cada proceso */ 

double *A,*B,*C;                    /*Punteros de las matrices globales*/
double *Alocal,*Blocal,*Clocal; 	/*Punteros de las matrices locales*/

/*Metrics vars*/
clock_t inicio, fin;                                                                                                                                                                                        
double  duration;  

srand(SEED);
MPI_Status st;
 
/* Comprobación numero de argumentos correctos. Se pasa m */
if (argc!=2){
   printf("Error de Sintaxis. Uso: mpi_gemm n \n");
   exit(1);
}

/* Lectura de parametros de entrada */
n=atoi(argv[1]); 

/* Inicializar el entorno MPI */
MPI_Init(&argc, &argv);
/* ¿Cuántos procesos somos? */
MPI_Comm_size(MPI_COMM_WORLD, &np);
/* ¿Cuál es mi identificador? */
MPI_Comm_rank(MPI_COMM_WORLD, &mid);
/* n debe ser múltiplo de np. Al menos para empezar. Cuando tengáis más experiencia, esto se puede adaptar. */
nlocal=n/np;  

/* Reserva de espacio para las matrices locales utilizando las rutinas en memoryfun.c */
Alocal=dmatrix(n,nlocal);
Clocal=dmatrix(n,nlocal);
B=dmatrix(n,n);    
   
if (!mid){
    /*Alloc Global non-shared variables*/
    C=dmatrix(n,n);	
    A=dmatrix(n,n);

	/* Relleno de las matrices. Uso de macro propia o memset para inicializar a 0*/
	for (i = 0; i < n; ++i)
        for(j = 0; j < n; ++j)
            M(A,i,j,n) = n*i+j+1; 
	for (i = 0; i < n; ++i)
        for(j = 0; j < n; ++j)
            M(B,i,j,n)=n*n+i*n+j+1;
    inicio = clock();
}

MPI_Scatter(A, n*nlocal, MPI_DOUBLE, Alocal, nlocal*n, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
MPI_Bcast(B,n*n,MPI_DOUBLE,ROOT,MPI_COMM_WORLD);

/* Cada proceso calcula el producto parcial de la matriz */
memset(Clocal,0.0,nlocal*n*sizeof(double));
cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,nlocal,n,n,ALPHA,Alocal,n,B,n,BETA,Clocal,n);

// Una vez calculada la matriz parcial se envía al proceso maestro, en este caso el 0, 
// para obtener el resultado global.
MPI_Gather(Clocal, n*nlocal, MPI_DOUBLE, C, n*nlocal, MPI_DOUBLE, ROOT, MPI_COMM_WORLD); 
fin = clock();

if (!mid){
    duration = (double)(fin- inicio)/TO_SECONDS;   
    printf("%7lf\n",duration);
}

/* Llegado a este punto el proceso 0 ha de tener toda la matriz, por lo que puede imprimirla */
#if DEBUG
if (mid==0)
    printMatrix(C, n, n);
#endif

/* Cerrar el entorno MPI */
MPI_Finalize();
return 0;
}

/* 
A =

     1     5     8     12
     2     6     9     13
     3     7    10     14
     4     8    11     15
	
B =

    17    21    25    29
    18    22    26    30
    19    23    27    31
    20    24    28    32


Resultado 


         538         650         762         874
         612         740         868         996
         686         830         974        1118
         760         920        1080        1240
		
		*/
		





