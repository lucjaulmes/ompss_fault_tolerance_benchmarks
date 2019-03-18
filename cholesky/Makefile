TARGET:=cholesky
OMP_SRC:=nonnested.c

include ../common.mk

LDLIBS+=-lm

# some CBLAS/LAPACK is required. Setup given for a normal machine and for BSC Supercomputers using ATLAS.
ifeq ($(filter-out power,$(BSC_MACHINE)),) # CTE-Power9 or empty
LDLIBS+=-lcblas -llapack -lblas
endif

# mn
ifeq ($(BSC_MACHINE),mn4)
CBLAS_HOME=/apps/ATLAS/3.10.3/GCC
CPPFLAGS += -I$(CBLAS_HOME)/include
LDFLAGS+=-L$(CBLAS_HOME)/lib -Wl,-rpath,$(CBLAS_HOME)/lib
LDLIBS+=-lcblas -llapack -latlas -lsatlas
endif

# nord (ATLAS just for cblas.h)
ifeq ($(BSC_MACHINE),nord3)
LAPACK_HOME=/apps/LAPACK/3.5.0/GCC
CBLAS_HOME=/apps/ATLAS/3.10.2
CPPFLAGS += -I$(CBLAS_HOME)/include
LDFLAGS+=-L$(CBLAS_HOME)/lib -L$(LAPACK_HOME)/lib -Wl,-rpath,$(LAPACK_HOME)/lib
LDLIBS+=-lblas -lcblas -latlas -lgfortran $(LAPACK_HOME)/lib/liblapack.a
endif

# power8
ifeq ($(BSC_MACHINE),power8)
OPENBLAS_HOME=/apps/EB/software/OpenBLAS/0.2.19-gompic-2016.10-LAPACK-3.6.1
CPPFLAGS+=-I$(OPENBLAS_HOME)/include
LDFLAGS+=-L$(OPENBLAS_HOME)/lib
LDLIBS+=-lopenblas
endif
