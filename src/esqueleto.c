#include <stdio.h>
#include "cblas.h"
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "memoryfun.h"
#include <omp.h>

#define SEED 2022
#define TAG 22

#define ALPHA 1
#define BETA 0

#define TO_SECONDS 1000000
#define ROOT 0

#define DEBUG 1

#define NUM_THREADS 4

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
 
/* Comprobación numero de argumentos correctos. Se pasa m */
if (argc!=2){
   printf("Error de Sintaxis. Uso: mpi_gemm n \n");
   exit(1);
}

/* Lectura de parametros de entrada */
n=atoi(argv[1]); 


/* Reserva de espacio para las matrices locales utilizando las rutinas en memoryfun.c */

/*Alloc Global non-shared variables*/
A=dmatrix(n,n);
B=dmatrix(n,n);    
C=dmatrix(n,n);	

	/* Relleno de las matrices. Uso de macro propia o memset para inicializar a 0*/
for (i = 0; i < n; ++i)
    for(j = 0; j < n; ++j)
        M(A,i,j,n) = n*i+j+1; 
for (i = 0; i < n; ++i)
    for(j = 0; j < n; ++j)
        M(B,i,j,n)=n*n+i*n+j+1;
inicio = clock();

#if DEBUG
    printf("MATRIX A\n");
    printMatrix(A,n,n);
    printf("\n");
    printf("MATRIX B\n");
    printMatrix(B,n,n);
    printf("\n");
#endif
/* Cada proceso calcula el producto parcial de la matriz */

double st = omp_get_wtime();

#pragma omp parallel private(i)
{  
    #pragma omp for
        for(i = 0; i < n; ++i){
            //printf("fila de A: \n");
            //printMatrix(A,n,n);  
            //printf("multiplicado por: \n");
            //printMatrix(&M(B,i,0,n),1,n);                                              
          cblas_dgemv(CblasRowMajor,CblasNoTrans,n,n,ALPHA,A,n,&M(B,i,0,1),n,BETA,&M(C,i,0,1),n);
         // printf("Soy el proceso: %d y tengo el valor i: %d \n",omp_get_thread_num(), i);
            //printf("con el resultado: \n");
            //printMatrix(&M(C,i,0,n),n,1);
            //printf("\n");

        }
}

//cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,nlocal,n,n,ALPHA,Alocal,n,B,n,BETA,Clocal,n);
           
// Una vez calculada la matriz parcial se envía al proceso maestro, en este caso el 0, 
// para obtener el resultado global.
double end = omp_get_wtime();
fin = clock();

//duration = (double)(fin- inicio)/TO_SECONDS;   
//printf("%7lf\n",duration);

/* Llegado a este punto el proceso 0 ha de tener toda la matriz, por lo que puede imprimirla */
#if DEBUG
    printMatrix(C, n, n);
#endif

/* Cerrar el entorno MPI */
return 0;
}

/* 
A =

     1     5     9     13
     2     6    10     14
     3     7    11     15
     4     8    12     16
	
B =

    17    21    25    29
    18    22    26    30
    19    23    27    31
    20    24    28    32


Resultado 


         538(250)         650(260)         762 (270)         874(280)
         612 (618)        740(644)         868 (670)         996(696)
         686 (986)        830 (1028)       974 (1070)        1118(1112)
         760 (1354)       920(1412)        1080(1470)        1240(1528)
		
		*/
		





