#ifndef REF_H
#define REF_H

#define  REALSIZE     8                                  /* in units of byte */
typedef double real;                 /* can be long double, double, or float */
typedef struct { real Re; real Im; } complex_struct;         /* for complex number */

                                     /* Mathematical functions and constants */
#undef M_PI
#if (REALSIZE==16)
#define sin  sinl
#define cos  cosl
#define fabs  fabsl
#define M_PI 3.1415926535897932384626433832795L
#else
#define M_PI 3.1415926535897932385E0
#endif

void ref_fft(complex_struct x[], int n, int flag);
void ref_fft2D(complex_struct x[], int n1, int n2, int flag);
void ref_fft3D(complex_struct x[], int n1, int n2, int n3, int flag);

#endif // REF_H
