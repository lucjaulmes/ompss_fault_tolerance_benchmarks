CC=smpcc --cc=gcc
CPPFLAGS = -I$(CBLAS_HOME)/include -I$(CATCHROI_HOME)/include
CFLAGS = -std=c11 -Wall -O$(O) -rdynamic -g
LDFLAGS = -L$(CBLAS_HOME)/lib -L$(CATCHROI_HOME)/lib -Wl,-rpath,$(CATCHROI_HOME)/lib
LDLIBS = -llapack -lm -lcatchroi

OUT=cholesky


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
LDLIBS+=-lblas -lcblas -latlas -lgfortran $(LAPACK_HOME)/lib/liblapack.a
endif

CATCHROI_HOME?=$(dir $(abspath $(dir $(lastword $(MAKEFILE_LIST)))))libcatchroi

all: perf instr debug seq
seq: $(OUT)_seq
perf: $(OUT)
instr: $(OUT)_instr
debug: $(OUT)_debug

$(OUT):$(OUT)_perf
	@mv $< $@

%_seq:CC=gcc

O=3
%_instr:LDFLAGS+=--instrument
%_debug:LDFLAGS+=--debug
%_debug:CFLAGS+=
%_debug:O=0

%_seq  :CFLAGS+=-Wno-unknown-pragmas -Wno-unused-variable
%_perf :CFLAGS+=--ompss -Wno-unused-variable -Wno-unused-function -K --output-dir=.build_perf
%_instr:CFLAGS+=--ompss -Wno-unused-variable -Wno-unused-function -K --output-dir=.build_instr
%_debug:CFLAGS+=--ompss -Wno-unused-variable -Wno-unused-function -K --output-dir=.build_debug


$(OUT)_%: nonnested.c dirs
	$(LINK.c) $< $(LDLIBS) -o $@


dirs:
	@mkdir -p .build_perf .build_instr .build_debug

clean:
	@rm -rf  .build_perf .build_instr .build_debug *.o
	@rm -fv $(OUT)_seq $(OUT) $(OUT)_debug $(OUT)_instr
