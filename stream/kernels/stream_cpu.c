#define BLOCK_FACTOR 16
void stream_kernel(size_t len, double * restrict a, double * restrict b, double * restrict c) {
   unsigned int i;
   size_t elen = len / BLOCK_FACTOR;
   for ( i = 0; i < BLOCK_FACTOR; i += 1) {
      double *aptr = &(a[i * elen]);
      double *bptr = &(b[i * elen]);
      double *cptr = &(c[i * elen]);
      //task here
      #pragma omp task firstprivate(elen) inout([elen]aptr, [elen]bptr, [elen] cptr)
      {
         int j;
         double scalar = 3.0;
         //copy
         for (j=0; j < elen; j++){
            cptr[j] = aptr[j];
         }
         //scale
         for (j=0; j < elen; j++){
            bptr[j] = scalar * cptr[j];
         }
         //add
         for (j=0; j < elen; j++){
            cptr[j] = aptr[j] + bptr[j];
         }
         //triad
         for (j=0; j < elen; j++){
            aptr[j] = bptr[j] + scalar * cptr[j];
         }
      }
   }
}
