CC=smpcc --cc=gcc

O=3
CPPFLAGS=-DNDEBUG -I$(CATCHROI_HOME)/include
CFLAGS=-Wall -O$(O) -k --ompss -fno-optimize-sibling-calls #-Wno-unused-variable -Wno-unused-local-typedefs -Wno-parentheses
LDFLAGS=-L$(CATCHROI_HOME)/lib -Wl,-rpath,$(CATCHROI_HOME)/lib
LDLIBS=-lm -lcatchroi

OUTPUT=sparseLU
CATCHROI_HOME?=$(dir $(abspath $(dir $(lastword $(MAKEFILE_LIST)))))libcatchroi

perf: $(OUTPUT)
instr: $(OUTPUT)_instr
debug: $(OUTPUT)_debug

sparseLU:checkSparseLU.c
	$(LINK.c) $^ $(LDLIBS) -o $@

sparseLU_instr:checkSparseLU.c
	$(LINK.c) --instrument $^ $(LDLIBS) -o $@

sparseLU_debug:O=0
sparseLU_debug:CFLAGS+=-g -rdynamic
sparseLU_debug:checkSparseLU.c
	$(LINK.c) --debug $^ $(LDLIBS) -o $@

clean:
	@rm -fv $(OUTPUT) $(OUTPUT)_instr $(OUTPUT)_debug

.PHONY: all clean instr clean

