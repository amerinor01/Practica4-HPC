#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include "memoryfun.h"

void print_error(rutina,texto_err)
char rutina[],texto_err[];

/* Manejo standard de errores */
{
        void exit();

        printf("ERROR DE EJECUCION EN RUTINA : %s\n",rutina);
        printf("%s\n",texto_err);
        printf("...EJECUCION DE PROGRAMA SUSPENDIDA...\n");
        exit(1);
}


double *dvector(int nh)

/* Dimensiona y reserve espacio para un vector double de rango [n1..nh] */
{
        double *v;

        v=(double *) malloc((unsigned) (nh)*sizeof(double));
        if (!v) print_error("dvector","error de reserva de espacio");
        return v;
}

int *ivector(int nh)

/* Dimensiona y reserve espacio para un vector double de rango [n1..nh] */
{
        int *v;

        v=(int *) malloc((unsigned) (nh)*sizeof(int));
        if (!v) print_error("ivector","error de reserva de espacio");
        return v;
}

double *dmatrix (int nfh, int nch) 

/* Dimensiona y reserva espacio para una matriz double [nf1..nfh][nc1..nch] */
{
	double *m;
	
	m=(double *)malloc((unsigned) (nfh)*(nch)*sizeof(double)); 
	if (!m) print_error("dmatriz","error de reserve de espacio 1");
	return m;
}

int *imatrix (int nfh, int nch) 

/* Dimensiona y reserva espacio para una matriz int [nf1..nfh][nc1..nch] */
{
	int *m;
	
	
	m=(int *)malloc((unsigned) (nfh)*(nch)*sizeof(int));
	if (!m) print_error("imatriz","error de reserve de espacio 1");
	return m;
}

void printMatrix(double* m, int n, int l){
        int i,j;
        for (i=0;i<n;i++){
                for (j=0;j<l;j++)
                        printf("%15.2lf ",M(m,i,j,l));
                printf("\n");
       }
}

void printMatrix2(double* m, int n, int n_local){
        int i,j;
        for (i=0;i<n_local;i++){
                for (j=0;j<n;j++)
                        printf("%15.2lf ",M(m,i,j,n_local));
                printf("\n");
       }
}