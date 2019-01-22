## This is the common parts of the Makefile. Define these first, before including:
#TARGET := cholesky        # will be post-fixed
#OMP_SRC := nonnested.c    # C source files with #pragma omp annotations and/or with main()
#SEQ_SRC :=			       # remaining C source files
#ADDITIONAL_TARGETS := ... # some more targets to be built/installed with all/install phonies

# NB: making object files and having separate --output-dir values per target is important to build in parallel
# even if here is only a single source, because when running `smpcc -o foo bar.c` the files bar.o and smpcc_bar.c
# will be used by all targets.

CC = gcc
CXX = g++
MCC = smpcc --cc=$(CC)
MCXX = smpcxx --cxx=$(CXX)

CFLAGS = -O$(O) -std=gnu11 -Wall -Wextra -rdynamic -g -fno-optimize-sibling-calls
CXXFLAGS = -O$(O) -std=gnu++11 -Wall -Wextra -rdynamic -g -fno-optimize-sibling-calls
OMPSSFLAGS = --ompss --Wn,-Wno-unused-parameter,-Wno-unused-but-set-variable,-Wno-unused-variable,-Wno-unused-function,-Wno-discarded-array-qualifiers -D_Float128=__float128
CPPFLAGS = -I$(CATCHROI_HOME)/include
LDFLAGS = -L$(CATCHROI_HOME)/lib -Wl,-rpath,$(CATCHROI_HOME)/lib
LDLIBS = -lcatchroi

ALL_TARGETS := bin/$(TARGET) bin/$(TARGET)_instr bin/$(TARGET)_debug bin/$(TARGET)_seq

CATCHROI_HOME ?= $(dir $(lastword $(MAKEFILE_LIST)))libcatchroi
DESTDIR ?= $(CATCHROI_HOME)

all: $(ALL_TARGETS)
extras:

perf: bin/$(TARGET)
instr: bin/$(TARGET)_instr
debug: bin/$(TARGET)_debug
seq: bin/$(TARGET)_seq


O = 3
%_debug:O = 0

%_seq.o   %_seq  :MCC := $(CC)
%_seq.o   %_seq  :OMPSSFLAGS = -Wno-unknown-pragmas -Wno-unused-variable
%_instr.o %_instr:OMPSSFLAGS += --instrument
%_debug.o %_debug:OMPSSFLAGS += --debug

%_perf.o :OMPSSFLAGS += --keep-all-files --output-dir=.build_perf
%_instr.o:OMPSSFLAGS += --keep-all-files --output-dir=.build_instr --instrument
%_debug.o:OMPSSFLAGS += --keep-all-files --output-dir=.build_debug --debug

define COMPILATION_RULES
$(MCC) $(CFLAGS) $(OMPSSFLAGS) $(CPPFLAGS) $< -c -o $@
endef

define LINKING_RULES
$(MCC) $(CFLAGS) $(OMPSSFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)
endef

# main target is perf and has no suffix
bin/$(TARGET): $(OMP_SRC:.c=_perf.o) $(SEQ_SRC:.c=_seq.o) | bin
	$(LINKING_RULES)

# all other targets {main target}_{suffix} have {source file}_{suffix}.o objects
bin/$(TARGET)_%: $(OMP_SRC:.c=_%.o) $(SEQ_SRC:.c=_seq.o) | bin
	$(LINKING_RULES)

# build objects
%_seq.o: %.c
	$(COMPILATION_RULES)

%_perf.o: %.c | .build_perf
	$(COMPILATION_RULES)

%_instr.o: %.c | .build_instr
	$(COMPILATION_RULES)

%_debug.o: %.c | .build_debug
	$(COMPILATION_RULES)

# make directories.
.build_%:
	@mkdir -p $@

bin:
	@mkdir -p $@

# some meta rules
clean:
	@rm -fv $(ALL_TARGETS)
	@rm -rf .build_perf .build_instr .build_debug *.o
	@rmdir --ignore-fail-on-non-empty bin

install: $(ALL_TARGETS)
	@mkdir -p $(DESTDIR)/bin
	@cp -v $^ $(DESTDIR)/bin/

uninstall:
	@rm -v $(addprefix $(DESTDIR)/bin/,$(notdir $(ALL_TARGETS)))
	@rmdir --ignore-fail-on-non-empty $(DESTDIR)/bin

.PRECIOUS: .build_perf .build_instr .build_debug bin
.INTERMEDIATE: $(SEQ_SRC:.c=_seq.o) $(OMP_SRC:.c=_perf.o) $(OMP_SRC:.c=_instr.o) $(OMP_SRC:.c=_debug.o) $(OMP_SRC:.c=_seq.o)
.PHONY: perf instr debug seq clean install uninstall extras

## After including file, adjust variables, e.g.:
#LDLIBS+=-lm
