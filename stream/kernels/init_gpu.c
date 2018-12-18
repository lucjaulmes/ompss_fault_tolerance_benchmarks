void initialize_kernel_gpu(size_t len, double *a, double *b, double *c) {
   unsigned int i;
   size_t elen = len / BLOCK_FACTOR;
   unsigned int node = 0; nanos_get_node_num(&node);
   for ( i = 0; i < BLOCK_FACTOR; i += 1) {
      //fprintf(stderr, "create task init\n");
#pragma omp target device (cuda) copy_deps
#pragma omp task firstprivate(i, len, elen, node) out(([len] a)[i*elen;elen], ([len] b)[i*elen;elen], ([len] c)[i*elen;elen])
      {
         //if ( node == 1 ) {
         //printf("node %d wd: %d init kernel addresses: i=%d, elen=%d, a=%p, b=%p, c=%p, a[%zu]=%p, b[%zu]=%p, c[%zu]=%p\n", node, nanos_get_wd_id(nanos_current_wd()), i, elen, a, b, c, i*elen, &a[i*elen], i*elen, &b[i*elen], i*elen, &c[i*elen]);
         //gpuErrchk( cudaPeekAtLastError() );
         //printf("node %d wd: %d init kernel addresses: i=%d, elen=%d, a=%p, b=%p, c=%p, a[%zu]=%p, b[%zu]=%p, c[%zu]=%p\n", node, nanos_get_wd_id(nanos_current_wd()), i, elen, a, b, c, i*elen, &a[i*elen], i*elen, &b[i*elen], i*elen, &c[i*elen]);
         //}
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

         //printf("dimGirid.x=%d, i should do %d calls, desiredGrid %d\n", dimGrid.x, calls, desiredGrid );
         int j;
         for(j = 0; j < calls; j += 1) {
            init_kernel<<<dimGrid,dimBlock>>>(elen, 1, &a[i*elen+j*65535*threadsPerBlock], &b[i*elen+j*65535*threadsPerBlock], &c[i*elen+j*65535*threadsPerBlock]);
         }
         //gpuErrchk( cudaPeekAtLastError() );
         //printf("END node %d wd: %d init kernel addresses: i=%d, elen=%d, a=%p, b=%p, c=%p, a[%zu]=%p, b[%zu]=%p, c[%zu]=%p\n", node, nanos_get_wd_id(nanos_current_wd()), i, elen, a, b, c, i*elen, &a[i*elen], i*elen, &b[i*elen], i*elen, &c[i*elen]);
      }
   }
}

