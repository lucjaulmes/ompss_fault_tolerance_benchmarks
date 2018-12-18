#N2|NlogN|N
BIGO?=N2

# Compilers. Will use CC in the end.
MCC ?= smpcc
CC = $(MCC) --cc=$(BACKEND_CC)

CPPFLAGS = -Iinclude -I$(CATCHROI_HOME)/include -DBIGO=$(BIGO) -Dalign_value=aligned -D_Float128=__float128
CFLAGS = -O$(O) -std=gnu99 -Wall -Wextra -fno-optimize-sibling-calls
OMPSSFLAGS = --ompss --Wn,-Wno-unused-variable,-Wno-unused-parameter
LDFLAGS = -L$(CATCHROI_HOME)/lib -Wl,-rpath,$(CATCHROI_HOME)/lib
LDLIBS = -lm -lcatchroi

MPI=1

ifeq ($(MPI),1)
MPICC ?= mpicc

#MPI_HOME ?= /usr/lib64/mpi/gcc/openmpi
MPI_HOME ?= /apps/INTEL/2017.4/impi/2017.3.196/intel64

BACKEND_CC := $(MPICC)
CPPFLAGS += -I$(MPI_HOME)/include -DUSE_MPI=1
LDLIBS += -lmpi
LDFLAGS += -L$(MPI_HOME)/lib64 -L$(MPI_HOME)/lib
else
BACKEND_CC := gcc
endif

CATCHROI_HOME?=$(dir $(abspath $(dir $(lastword $(MAKEFILE_LIST)))))libcatchroi

O=3
%_debug:O=0
%_debug:CFLAGS+=-g
debug:CFLAGS+=-g

%_perf.o: OMPSSFLAGS+=--keep-all-files --output-dir=.build_$(BIGO)_perf
%_debug.o:OMPSSFLAGS+=--keep-all-files --output-dir=.build_$(BIGO)_debug --debug
%_instr.o:OMPSSFLAGS+=--keep-all-files --output-dir=.build_$(BIGO)_instr --instrument

perf: OMPSSFLAGS+=
debug:OMPSSFLAGS+=--debug
instr:OMPSSFLAGS+=--instrument

all: perf instr debug
perf: bin/nbody_ompss_$(BIGO) dirs
instr: bin/nbody_ompss_$(BIGO)_instr dirs
debug: bin/nbody_ompss_$(BIGO)_debug dirs

# Common source files. Wildcard % takes care of selecting build: here, ompss.
# Try e.g make BIGO=N bin/nbody_intel_offload_N, it will use src/nbody_intel_offload.c
SRC=common.c main.c kernel_smp.c
ifeq ($(MCC),icc)
SRC+=iomp_syms.c
endif

bin/nbody_%_$(BIGO): nbody_%_$(BIGO)_perf.o $(SRC:.c=_$(BIGO)_perf.o)
	$(MCC) --cc=$(MPICC) $(OMPSSFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/nbody_%_$(BIGO)_instr: nbody_%_$(BIGO)_instr.o $(SRC:.c=_$(BIGO)_instr.o)
	$(MCC) --cc=$(MPICC) $(OMPSSFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

bin/nbody_%_$(BIGO)_debug: nbody_%_$(BIGO)_debug.o $(SRC:.c=_$(BIGO)_debug.o)
	$(MCC) --cc=$(MPICC) $(OMPSSFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)


vpath %.c src kernels

common%.o:CC=$(BACKEND_CC)
main%.o:CC=$(BACKEND_CC)
kernel_%.o:CFLAGS+=$(OMPSSFLAGS)
nbody_%.o:CFLAGS+=$(OMPSSFLAGS)

%_$(BIGO)_debug.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

%_$(BIGO)_instr.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

%_$(BIGO)_perf.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c -o $@ $<

clean:
	@rm -rf *.o .build_N2_perf .build_N2_debug .build_N2_instr bin .build_NlogN_perf .build_NlogN_debug .build_NlogN_instr bin .build_N_perf .build_N_debug .build_N_instr bin
	@rm -vf bin/*

dirs:
	@mkdir -p .build_$(BIGO)_perf .build_$(BIGO)_debug .build_$(BIGO)_instr bin data

.INTERMEDIATE:%.o
.PHONY: all clean dirs perf instr debug


