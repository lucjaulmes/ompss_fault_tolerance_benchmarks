MCC = smpcc --cc=$(CC)
OPT=3
CFLAGS  = -O$(OPT) -ffast-math $(OMPSSFLAGS) -Wall -Wextra -fno-optimize-sibling-calls
OMPSSFLAGS = --ompss -D_OMPSS -D_Float128=__float128
CPPFLAGS = -I$(CATCHROI_HOME)/include
LDFLAGS = -L$(CATCHROI_HOME)/lib -Wl,-rpath,$(CATCHROI_HOME)/lib
LIBS = -lm -lcatchroi
SOURCES = cg.c main.c
SEQ_SOURCES = matrix.c mmio.c
OBJECTS = $(SOURCES:.c=.o) $(SEQ_SOURCES:.c=_seq.o)
EXECUTABLE = cg
EXECUTABLES = $(EXECUTABLE) $(EXECUTABLE)_instr $(EXECUTABLE)_debug $(EXECUTABLE)_seq

CATCHROI_HOME?=$(dir $(abspath $(dir $(lastword $(MAKEFILE_LIST)))))libcatchroi


do: perf

perf: $(EXECUTABLE)

instr: OMPSSFLAGS += --instrument
instr: $(EXECUTABLE)_instr

debug: OMPSSFLAGS += --debug -K
debug: CFLAGS += -g
debug: OPT = 0
debug: $(EXECUTABLE)_debug

seq:$(EXECUTABLE)_seq
seq %_seq.o:MCC = gcc
%_seq.o:OMPSSFLAGS =
seq:OMPSSFLAGS =  -Wno-unknown-pragmas


$(EXECUTABLES): $(OBJECTS)
	$(MCC) $(CFLAGS) $(LDFLAGS) $(OBJECTS) -o $@ $(LIBS)

%.o: %.c
	$(MCC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

%_seq.o: %.c
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

clean:
	rm -rf *.o $(EXECUTABLES)

.INTERMEDIATE: $(SOURCES:.c=.o)
.PHONY: perf instr
