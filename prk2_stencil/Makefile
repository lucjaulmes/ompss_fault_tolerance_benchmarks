CC = gcc
MCC = smpcc --cc=$(CC)

O=3
CPPFLAGS=-D_Float128=__float128 -DVALIDATE=1 -I$(CATCHROI_HOME)/include -K
CFLAGS=-Wall -Wextra -Wpedantic -O$(O) --keep-all-files --ompss -Wno-maybe-uninitialized -Wno-unused-parameter -Wno-unused-but-set-variable -std=c11 -g -fno-optimize-sibling-calls
LDFLAGS=-L$(CATCHROI_HOME)/lib -Wl,-rpath,$(CATCHROI_HOME)/lib
LDLIBS=-lcatchroi -lm

EXEC=prk2_stencil
CATCHROI_HOME?=$(dir $(abspath $(dir $(lastword $(MAKEFILE_LIST)))))libcatchroi

perf: $(EXEC) dirs
instr: $(EXEC)_instr  dirs
debug: $(EXEC)_debug  dirs
all: perf instr debug

SRC=stencil.c

$(EXEC): $(SRC:.c=_perf.o)
	$(MCC) $(CFLAGS) $(LDFLAGS) $(LDLIBS) $^ -o $@

$(EXEC)_instr: $(SRC:.c=_instr.o)
	$(MCC) $(CFLAGS) $(LDFLAGS) --instrument $(LDLIBS) $^ -o $@

$(EXEC)_debug: $(SRC:.c=_debug.o)
	$(MCC) $(CFLAGS) $(LDFLAGS) --debug $(LDLIBS) $^ -o $@

%_perf.o: %.c
	$(MCC) $(CFLAGS) $(CPPFLAGS) --output-dir=.build_perf -c $< -o $@

%_instr.o: %.c
	$(MCC) $(CFLAGS) $(CPPFLAGS) --output-dir=.build_instr --instrument -c $< -o $@

%_debug.o: %.c
	$(MCC) $(CFLAGS) $(CPPFLAGS) --output-dir=.build_debug --debug -c $< -o $@

clean:
	@rm -vf *.o $(EXEC) $(EXEC)_instr $(EXEC)_debug

dirs:
	@mkdir -p .build_perf .build_instr .build_debug

.PHONY:clean dirs
.INTERMEDIATE: $(notdir $(SRC:.c=_perf.o)) $(notdir $(SRC:.c=_instr.o)) $(notdir $(SRC:.c=_debug.o))

