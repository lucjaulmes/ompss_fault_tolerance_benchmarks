CC=gcc
MCC=smpcc --cc=$(CC)

CPPFLAGS=-D_GNU_SOURCE -DDOUBLE_PREC -I$(CBLAS_HOME)/include -I$(CATCHROI_HOME)/include
CFLAGS=-O$(O) -frecord-gcc-switches $(OMPSSFLAGS)
OMPSSFLAGS=--ompss -k --Wn,-Wno-incompatible-pointer-types
LDFLAGS=-L$(CATCHROI_HOME)/lib -Wl,-rpath,$(CATCHROI_HOME)/lib
LDLIBS=-lm -llapack -lcatchroi

CATCHROI_HOME?=$(dir $(abspath $(dir $(lastword $(MAKEFILE_LIST)))))libcatchroi


# mn
ifeq ($(BSC_MACHINE),mn4)
CBLAS_HOME=/apps/ATLAS/3.10.3/GCC
LDFLAGS+=-L$(CBLAS_HOME)/lib -Wl,-rpath,$(CBLAS_HOME)/lib
LDLIBS+=-lcblas -latlas -lsatlas
endif

# power 9
ifeq ($(BSC_MACHINE),power)
CBLAS_HOME=/apps/ATLAS/3.10.3/GCC/OPENMPI/
LDFLAGS+=-L$(CBLAS_HOME)/lib -Wl,-rpath,$(CBLAS_HOME)/lib
LDLIBS+=-lcblas -latlas -lsatlas
endif

# nord (ATLAS just for cblas.h)
ifeq ($(BSC_MACHINE),nord3)
LAPACK_HOME=/apps/LAPACK/3.5.0/GCC
CBLAS_HOME=/apps/ATLAS/3.10.2
LDFLAGS+=-L$(LAPACK_HOME)/lib -Wl,-rpath,$(LAPACK_HOME)/lib
LDLIBS+=-lblas -lgfortran
endif

SRC=inverse.c
OUTPUT=inverse

all:perf instr debug
perf :$(OUTPUT)
instr:$(OUTPUT)_instr
debug:$(OUTPUT)_debug
seq  :$(OUTPUT)_seq

$(OUTPUT):$(OUTPUT)_perf
	@mv $< $@

O=3
%_debug:O=0
%_debug:CFLAGS+=-g

%_seq:MCC=$(CC)
%_seq:OMPSSFLAGS=-Wno-unknown-pragmas

%_perf :OMPSSFLAGS+=-K --output-dir=.build_perf
%_instr:OMPSSFLAGS+=-K --output-dir=.build_instr
%_debug:OMPSSFLAGS+=-K --output-dir=.build_debug

%_instr:LDFLAGS+=--instrument
%_debug:LDFLAGS+=--debug

$(OUTPUT)_%: $(SRC) | dirs
	$(MCC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) $< -o $@ $(LDLIBS)

dirs:
	@mkdir -p .build_perf .build_instr .build_debug

clean:
	@rm -rf .build_perf .build_instr .build_debug *.o
	@rm -vf $(OUTPUT) $(OUTPUT)_instr $(OUTPUT)_debug

.PHONY: all clean instr dirs debug perf instr
.INTERMEDIATE:*.o
