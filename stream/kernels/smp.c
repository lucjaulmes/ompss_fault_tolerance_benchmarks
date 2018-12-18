
void init_kernel(double *a, double *b, double *c, unsigned long bs, unsigned long j)
{
	#pragma omp task out(a[j;bs], b[j;bs], c[j;bs]) label(init_kernel)
	{
		#pragma simd
		for (unsigned long i = j; i < j + bs; i++)
		{
			a[i] = 1.0;
			b[i] = 2.0;
			c[i] = 0.0;
			a[i] = 2.0 * a[i];
		}
	}
}


void copy_kernel(double *a, double *c, unsigned long bs, unsigned long j)
{
	#pragma omp task in(a[j;bs]) out(c[j;bs]) label(copy_kernel)
	{
		#pragma simd
		for (unsigned long i = j; i < j + bs; i++)
			c[i] = a[i];
	}
}


void scale_kernel(double *b, double *c, double scalar, unsigned long bs, unsigned long j)
{
	#pragma omp task in(c[j;bs]) out(b[j;bs]) label(scale_kernel)
	{
		#pragma simd
		for (unsigned long i = j; i < j + bs; i++)
			b[i] = scalar * c[i];
	}
}


void add_kernel(double *a, double *b, double *c, unsigned long bs, unsigned long j)
{
	#pragma omp task in(a[j;bs], b[j;bs]) out(c[j;bs]) label(add_kernel)
	{
		#pragma simd
		for (unsigned long i = j; i < j + bs; i++)
			c[i] = a[i] + b[i];
	}
}


void triad_kernel(double *a, double *b, double *c, double scalar, unsigned long bs, unsigned long j)
{
	#pragma omp task in(b[j;bs], c[j;bs]) out(a[j;bs]) label(triad_kernel)
	{
		#pragma simd
		for (unsigned long i = j; i < j + bs; i++)
			a[i] = b[i] + scalar * c[i];
	}
}
