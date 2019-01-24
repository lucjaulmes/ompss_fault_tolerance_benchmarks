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
OMPSSFLAGS = --ompss --Wn,-Wno-unused-parameter,-Wno-unused-but-set-variable,-Wno-unused-variable,-Wno-unused-function,-Wno-discarded-array-qualifiers
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

define C_COMPILATION
$(MCC) $(CFLAGS) $(OMPSSFLAGS) $(CPPFLAGS) $< -c -o $@
endef

define CXX_COMPILATION
$(MCXX) $(CXXFLAGS) $(OMPSSFLAGS) $(CPPFLAGS) $< -c -o $@
endef


define C_LINKING
$(MCC) $(CFLAGS) $(OMPSSFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)
endef

define CXX_LINKING
$(MCXX) $(CXXFLAGS) $(OMPSSFLAGS) $(LDFLAGS) $^ -o $@ $(LDLIBS)
endef



# main target is perf and has no suffix
bin/$(TARGET): $(addsuffix _perf.o, $(basename $(OMP_SRC))) $(addsuffix _seq.o, $(basename $(SEQ_SRC))) | bin
ifeq ($(filter-out %.c , $(OMP_SRC) $(SEQ_SRC)),)
	$(C_LINKING)
else
	$(CXX_LINKING)
endif

# all other targets {main target}_{suffix} have {source file}_{suffix}.o objects
bin/$(TARGET)_%: $(addsuffix _%.o, $(basename $(OMP_SRC))) $(addsuffix _seq.o, $(basename $(SEQ_SRC))) | bin
ifeq ($(filter-out %.c , $(OMP_SRC) $(SEQ_SRC)),)
	$(C_LINKING)
else
	$(CXX_LINKING)
endif


# build objects
%_seq.o: %.c
	$(C_COMPILATION)

%_perf.o: %.c | .build_perf
	$(C_COMPILATION)

%_instr.o: %.c | .build_instr
	$(C_COMPILATION)

%_debug.o: %.c | .build_debug
	$(C_COMPILATION)

%_seq.o: %.cc
	$(CXX_COMPILATION)

%_perf.o: %.cc | .build_perf
	$(CXX_COMPILATION)

%_instr.o: %.cc | .build_instr
	$(CXX_COMPILATION)

%_debug.o: %.cc | .build_debug
	$(CXX_COMPILATION)

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
.INTERMEDIATE: $(addsuffix _seq.o, $(basename $(SEQ_SRC)))
.INTERMEDIATE: $(addsuffix _perf.o, $(basename $(OMP_SRC)))
.INTERMEDIATE: $(addsuffix _instr.o, $(basename $(OMP_SRC)))
.INTERMEDIATE: $(addsuffix _debug.o, $(basename $(OMP_SRC)))
.INTERMEDIATE: $(addsuffix _seq.o, $(basename $(OMP_SRC)))
.PHONY: perf instr debug seq clean install uninstall extras

## After including file, adjust variables, e.g.:
#LDLIBS+=-lm
