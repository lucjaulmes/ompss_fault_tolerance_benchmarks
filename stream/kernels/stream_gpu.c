#define BLOCK_FACTOR 4
void stream_kernel(size_t len, double *a, double *b, double *c) {
   unsigned int i;
   size_t elen = len / BLOCK_FACTOR;
   unsigned int node = 0; nanos_get_node_num(&node);
   for ( i = 0; i < BLOCK_FACTOR; i += 1) {
//      double *aptr = &(a[i * elen]);
//      double *bptr = &(b[i * elen]);
//      double *cptr = &(c[i * elen]);
#pragma omp target device (cuda) copy_deps
//#pragma omp task firstprivate(elen) inout([elen] aptr, [elen] bptr, [elen] cptr)
#pragma omp task firstprivate(i, len, elen, node) inout(([len] a)[i*elen;elen], ([len] b)[i*elen;elen], ([len] c)[i*elen;elen])
      {
            //printf("START node %d wd: %d stream kernel addresses: i=%d, elen=%d, a=%p, b=%p, c=%p, a[%zu]=%p, b[%zu]=%p, c[%zu]=%p\n", node, nanos_get_wd_id(nanos_current_wd()), i, elen, a, b, c, i*elen, &a[i*elen], i*elen, &b[i*elen], i*elen, &c[i*elen]);
            //gpuErrchk( cudaPeekAtLastError() );
            //printf("START node %d wd: %d stream kernel addresses: i=%d, elen=%d, a=%p, b=%p, c=%p, a[%zu]=%p, b[%zu]=%p, c[%zu]=%p\n", node, nanos_get_wd_id(nanos_current_wd()), i, elen, a, b, c, i*elen, &a[i*elen], i*elen, &b[i*elen], i*elen, &c[i*elen]);
            double scalar = 3.0;

         //   const int threadsPerBlock = 512;
         //   dim3 dimBlock;
         //   dimBlock.x = threadsPerBlock;
         //   dimBlock.y = dimBlock.z = 1;

         //   dim3 dimGrid;  
         //   dimGrid.x = elen/threadsPerBlock; 
         const int threadsPerBlock = 768;
         dim3 dimBlock;
         dimBlock.x = threadsPerBlock;
         dimBlock.y = dimBlock.z = 1;

         dim3 dimGrid;
         int calls = 1;
         int desiredGrid = elen / threadsPerBlock;
         if ( desiredGrid <= 65535 ) {
            dimGrid.x = desiredGrid;
         } else {
            calls += ( desiredGrid / 65535 );
            dimGrid.x = 65535;// elen/threadsPerBlock; 
         }

         int j;
         for (j = 0; j < calls; j += 1) {
            copy_kernel<<<dimGrid,dimBlock>>>(elen, 1, &a[i*elen+j*65535*threadsPerBlock], &c[i*elen+j*65535*threadsPerBlock]);
            //gpuErrchk( cudaPeekAtLastError() );
            scale_kernel<<<dimGrid,dimBlock>>>(elen, 1, &b[i*elen+j*65535*threadsPerBlock], &c[i*elen+j*65535*threadsPerBlock], scalar);
            //gpuErrchk( cudaPeekAtLastError() );
            add_kernel<<<dimGrid,dimBlock>>>(elen, 1, &a[i*elen+j*65535*threadsPerBlock], &b[i*elen+j*65535*threadsPerBlock], &c[i*elen+j*65535*threadsPerBlock]);
            //gpuErrchk( cudaPeekAtLastError() );
            triad_kernel<<<dimGrid,dimBlock>>>(elen, 1, &a[i*elen+j*65535*threadsPerBlock], &b[i*elen+j*65535*threadsPerBlock], &c[i*elen+j*65535*threadsPerBlock], scalar);
            //gpuErrchk( cudaPeekAtLastError() );
            //printf("END node %d wd: %d stream kernel addresses: i=%d, elen=%d, a=%p, b=%p, c=%p, a[%zu]=%p, b[%zu]=%p, c[%zu]=%p\n", node, nanos_get_wd_id(nanos_current_wd()), i, elen, a, b, c, i*elen, &a[i*elen], i*elen, &b[i*elen], i*elen, &c[i*elen]);
         }
      }
   }
}
