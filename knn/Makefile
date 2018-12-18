CC=gcc
MCC=smpcc --cc=$(CC)

CPPFLAGS=-DNDEBUG -I$(CATCHROI_HOME)/include -D_Float128=__float128
CFLAGS=-Wall -O$(O) $(D) -std=gnu11 -fno-optimize-sibling-calls -Wno-unused-variable -Wno-unused-local-typedefs -Wno-parentheses --ompss
LDFLAGS=-L$(CATCHROI_HOME)/lib -Wl,-rpath,$(CATCHROI_HOME)/lib
LDLIBS=-lpthread -lm -lcatchroi

OUTPUT = knn

CATCHROI_HOME?=$(dir $(abspath $(dir $(lastword $(MAKEFILE_LIST)))))libcatchroi

all: perf instr debug
perf: $(OUTPUT) dirs
instr: $(OUTPUT)_instr dirs
debug: $(OUTPUT)_debug dirs

O=3
%_instr:CFLAGS += --instrument
%_debug:CFLAGS += --debug
%_debug:O=0
%_debug:D=-g

%_perf: CFLAGS += --keep-all-files --output-dir=.build_perf
%_instr:CFLAGS += --keep-all-files --output-dir=.build_instr
%_debug:CFLAGS += --keep-all-files --output-dir=.build_debug

$(OUTPUT)_%: knn.c
	$(MCC) $(CPPFLAGS) $(CFLAGS) $(LDFLAGS) $<  -o $@ $(LDLIBS)

$(OUTPUT):$(OUTPUT)_perf
	@mv $< $@

dirs:
	@mkdir -p .build_perf .build_instr .build_debug

clean:
	@rm -rvf $(OUTPUT) $(OUTPUT)_instr $(OUTPUT)_debug
	@rm -rf .build_perf .build_instr .build_debug *.o

.PHONY: all clean instr perf debug
