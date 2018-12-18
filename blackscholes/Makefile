CC		= gcc
MCC		= smpcc --cc=$(CC)
TARGET=blackscholes

CFLAGS=-O2 -std=gnu11 -rdynamic -Wall -Wextra -g -rdynamic -fno-optimize-sibling-calls
OMPSSFLAGS = --ompss --no-copy-deps --Wn,-Wno-unused-but-set-variable,-Wno-unused-parameter -D_Float128=__float128
CPPFLAGS += -DERR_CHK=0 -I$(CATCHROI_HOME)/include
LDFLAGS += -L$(CATCHROI_HOME)/lib -Wl,-rpath,$(CATCHROI_HOME)/lib
LDLIBS += -lm -lcatchroi

CATCHROI_HOME?=$(dir $(abspath $(dir $(lastword $(MAKEFILE_LIST)))))libcatchroi

# un-strict aliasing needed for the ways #atomic on doubles is translated into __sync_cas
%_perf: OMPSSFLAGS+=--keep-all-files --output-dir=.build_perf
%_debug:OMPSSFLAGS+=--keep-all-files --output-dir=.build_debug --debug
%_instr:OMPSSFLAGS+=--keep-all-files --output-dir=.build_instr --instrument

$(TARGET)_seq inputgen: MCC=$(CC)
$(TARGET)_seq inputgen: OMPSSFLAGS=-Wno-unknown-pragmas

CFLAGS+=$(OMPSSFLAGS)

all:   perf instr debug seq inputgen
seq:   $(TARGET)_seq
perf:  $(TARGET)
instr: $(TARGET)_instr
debug: $(TARGET)_debug

%:%.c
	$(LINK.c) $< -o $@

$(TARGET):$(TARGET)_perf
	@mv $< $@

# Force distinct object names
$(TARGET)_%:$(TARGET)_%.o
	$(MCC) $(CFLAGS) $(TARGET_ARCH) $(LDFLAGS) $< $(LDLIBS) -o $@

$(TARGET)_%.o:$(TARGET).c | dirs
	$(MCC) $(CFLAGS) $(CPPFLAGS) $(TARGET_ARCH) -c $< -o $@

.PHONY: clean dirs all seq perf instr debug

dirs:
	@mkdir -p .build_perf .build_debug .build_instr

clean:
	@rm -rfv *.o $(TARGET) $(TARGET)_seq $(TARGET)_perf $(TARGET)_instr $(TARGET)_debug inputgen

.INTERMEDIATE:%.o

