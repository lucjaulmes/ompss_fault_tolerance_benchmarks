


#define BLOCK_FACTOR_CPU 1
void initialize_kernel(size_t len, double *a, double *b, double *c) {
   unsigned int i;
   size_t elen = len / BLOCK_FACTOR_CPU;
   for ( i = 0; i < BLOCK_FACTOR_CPU; i += 1) {
      double *aptr = &(a[i * elen]);
      double *bptr = &(b[i * elen]);
      double *cptr = &(c[i * elen]);
      //#pragma omp target device (smp) copy_deps
      //#pragma omp task firstprivate(elen) out([elen] aptr, [elen] bptr, [elen] cptr)
      {
         int j;
         for (j=0; j < elen; j++){
            aptr[j] = 1.0;
            bptr[j] = 2.0;
            cptr[j] = 0.0;
            aptr[j] = 2.0E0 * aptr[j];
         }
      }
   }
}
