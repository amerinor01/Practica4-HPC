#define M(a,i,j,lda) (a)[(i)*lda+(j)]

void print_error(char *rutina, char *texto_err);
double *dvector(int nh);
int *ivector(int nh);
double *dmatrix(int nfh, int nch);
int *imatrix(int nfh, int nch);

void printMatrix(double* m,int n, int l);
void printMatrix2(double* m, int n, int n_local);