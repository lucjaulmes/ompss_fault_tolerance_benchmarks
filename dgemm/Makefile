CC=gcc
MCC=smpcc --cc=$(CC)

CPPFLAGS=-DNDEBUG -DVALIDATE -I$(CBLAS_HOME)/include -I$(CATCHROI_HOME)/include
CFLAGS=-Wall -Wextra -std=c11 -O$(O) $(OMPSSFLAGS)
OMPSSFLAGS=--ompss --Wn,-Wno-unused-parameter,-Wno-unused-but-set-variable -fno-optimize-sibling-calls
LDFLAGS=-L$(CBLAS_HOME)/lib -L$(CATCHROI_HOME)/lib -Wl,-rpath,$(CATCHROI_HOME)/lib
LDLIBS=-lcblas -latlas -lpthread -lm -lcatchroi

OUTPUT = dgemm

CATCHROI_HOME?=$(dir $(abspath $(dir $(lastword $(MAKEFILE_LIST)))))libcatchroi

# mn
ifeq ($(BSC_MACHINE),mn4)
CBLAS_HOME=/apps/ATLAS/3.10.3/GCC
endif

# nord
ifeq ($(BSC_MACHINE),nord3)
CBLAS_HOME=/apps/ATLAS/3.10.2
endif

# power 9
ifeq ($(BSC_MACHINE),power)
CBLAS_HOME=/apps/ATLAS/3.10.3/GCC/OPENMPI/
endif

all:perf debug instr seq

O=3
%_debug:O=0

%_seq:  OMPSSFLAGS+=-Wno-unknown-pragmas
%_perf: OMPSSFLAGS+=-K --output-dir=.build_perf
%_instr:OMPSSFLAGS+=-K --output-dir=.build_instr --instrument
%_debug:OMPSSFLAGS+=-K --output-dir=.build_debug --debug -g

$(OUTPUT)_%: dgemm.c dirs
	$(MCC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) $<  -o $@ $(LDLIBS)

$(OUTPUT):$(OUTPUT)_perf
	@mv $< $@

dirs:
	@mkdir -p .build_perf .build_instr .build_debug

perf:  $(OUTPUT)
seq:   $(OUTPUT)_seq
instr: $(OUTPUT)_instr
debug: $(OUTPUT)_debug
all: perf instr debug

clean:
	@rm -rf *.o .build_perf .build_instr .build_debug
	@rm -vf $(OUTPUT) $(OUTPUT)_instr $(OUTPUT)_debug

.PHONY: all clean perf debug instr seq
