TARGET := dgemm
OMP_SRC := dgemm.c

include ../common.mk

CPPFLAGS+=-DNDEBUG -DVALIDATE
LDLIBS+=-lm

# Get a CBLAS implementation
LDLIBS+=-lcblas -llapack

# mn
ifeq ($(BSC_MACHINE),mn4)
CBLAS_HOME=/apps/ATLAS/3.10.3/GCC
CPPFLAGS+=-I$(CBLAS_HOME)/include
LDFLAGS+=-L$(CBLAS_HOME)/lib
LDLIBS+=-latlas
endif

# nord
ifeq ($(BSC_MACHINE),nord3)
CBLAS_HOME=/apps/ATLAS/3.10.2
CPPFLAGS+=-I$(CBLAS_HOME)/include
LDFLAGS+=-L$(CBLAS_HOME)/lib
LDLIBS+=-lcblas -latlas
endif
