TARGET := dgemm
OMP_SRC := dgemm.c

include ../common.mk

CPPFLAGS+=-DNDEBUG -DVALIDATE
LDLIBS+=-lm

# Get a CBLAS implementation
ifeq ($(BSC_MACHINE),)
LDLIBS+=-lcblas -llapack
endif

# CTE-Power9
ifeq ($(BSC_MACHINE),power)
OPENBLAS_HOME=/apps/OPENBLAS/0.2.20/GCC/OPENMPI
CPPFLAGS+=-I$(OPENBLAS_HOME)/include
LDFLAGS+=-L$(OPENBLAS_HOME)/lib
LDLIBS+=-lopenblas
endif

# mn
ifeq ($(BSC_MACHINE),mn4)
CBLAS_HOME=/apps/ATLAS/3.10.3/GCC-8.1
CPPFLAGS+=-I$(CBLAS_HOME)/include
LDFLAGS+=-L$(CBLAS_HOME)/lib -Wl,-rpath,$(CBLAS_HOME)/lib
LDLIBS+=-lcblas -latlas
endif

# nord
ifeq ($(BSC_MACHINE),nord3)
CBLAS_HOME=/apps/ATLAS/3.10.2
CPPFLAGS+=-I$(CBLAS_HOME)/include
LDFLAGS+=-L$(CBLAS_HOME)/lib
LDLIBS+=-lcblas -latlas
endif

# power8
ifeq ($(BSC_MACHINE),power8)
OPENBLAS_HOME=/apps/EB/software/OpenBLAS/0.2.19-gompic-2016.10-LAPACK-3.6.1
CPPFLAGS+=-I$(OPENBLAS_HOME)/include
LDFLAGS+=-L$(OPENBLAS_HOME)/lib
LDLIBS+=-lopenblas
endif
