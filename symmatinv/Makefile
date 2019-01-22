OMP_SRC:=inverse.c
TARGET:=inverse

include ../common.mk

LDLIBS+=-lm
CPPFLAGS+=-DDOUBLE_PREC

# some CBLAS/LAPACK is required. Setup given for a normal machine and for BSC Supercomputers using ATLAS.
ifeq ($(filter-out power,$(BSC_MACHINE)),) # CTE-Power9 or empty
LDLIBS+=-lcblas -llapack -lblas
endif


# mn
ifeq ($(BSC_MACHINE),mn4)
CBLAS_HOME=/apps/ATLAS/3.10.3/GCC-8.1
CPPFLAGS+=-I$(CBLAS_HOME)/include
LDFLAGS+=-L$(CBLAS_HOME)/lib -Wl,-rpath,$(CBLAS_HOME)/lib
LDLIBS+=-llapack -lcblas -latlas -lsatlas
endif

# nord (ATLAS just for cblas.h)
ifeq ($(BSC_MACHINE),nord3)
LAPACK_HOME=/apps/LAPACK/3.5.0/GCC
CBLAS_HOME=/apps/ATLAS/3.10.2
CPPFLAGS+=-I$(CBLAS_HOME)/include
LDFLAGS+=-L$(LAPACK_HOME)/lib -Wl,-rpath,$(LAPACK_HOME)/lib
LDLIBS+=-llapack -lcblas -lgfortran
endif
