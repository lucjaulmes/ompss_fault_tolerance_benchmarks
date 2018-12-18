CXX=g++
MCXX=smpcxx

CPPFLAGS=-I$(CATCHROI_HOME)/include
CXXFLAGS=-Wall -Wextra -std=c++11 -O$(O) $(D) $(MCXXFLAGS) -fno-optimize-sibling-calls
LDFLAGS=-L$(CATCHROI_HOME)/lib -Wl,-rpath,$(CATCHROI_HOME)/lib
LIBS=-lcatchroi

CATCHROI_HOME?=$(dir $(abspath $(dir $(lastword $(MAKEFILE_LIST)))))libcatchroi

# optimization levels: default and debug
O=3
heat_debug:D=-g -rdynamic
heat_debug:O=0

%_seq   %_seq.o:  MCXXFLAGS=-Wno-unknown-pragmas
heat    %_perf.o: MCXXFLAGS=--ompss --Wn,-Wno-unused-parameter,-Wno-unused-but-set-parameter,-Wno-unused-local-typedefs --keep-all-files --output-dir=.build_perf
%_instr %_instr.o:MCXXFLAGS=--ompss --Wn,-Wno-unused-parameter,-Wno-unused-but-set-parameter,-Wno-unused-local-typedefs --keep-all-files --output-dir=.build_instr --instrument
%_debug %_debug.o:MCXXFLAGS=--ompss --Wn,-Wno-unused-parameter,-Wno-unused-but-set-parameter,-Wno-unused-local-typedefs --keep-all-files --output-dir=.build_debug --debug



# object recipes
%_seq.o: %.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $^

%_perf.o: %.cc dirs
	$(MCXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

%_instr.o: %.cc dirs
	$(MCXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

%_debug.o: %.cc dirs
	$(MCXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<


# target recipes. NB most objects are seq
TARGETS=heat heat_seq heat_instr

default: heat
all: $(TARGETS)

seq: heat_seq
debug: heat_debug
instr: heat_instr
perf: heat



OMP_SRC:=algorithms.cc heat.cc
SEQ_SRC:=algorithms_inner.cc matrix.cc misc.cc

heat_seq: $(SEQ_SRC:.cc=_seq.o) $(OMP_SRC:.cc=_seq.o)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

heat: $(SEQ_SRC:.cc=_seq.o) $(OMP_SRC:.cc=_perf.o)
	$(MCXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LIBS)

heat_instr: $(SEQ_SRC:.cc=_seq.o) $(OMP_SRC:.cc=_instr.o)
	$(MCXX) $(CXXFLAGS) -_instrument $(LDFLAGS) -o $@ $^ $(LIBS)




.PHONY: clean distclean all default install dirs seq debug instr perf
.INTERMEDIATE:$(OMP_SRC:.cc=_debug.o) $(OMP_SRC:.cc=_instr.o) $(OMP_SRC:.cc=_perf.o) $(OMP_SRC:.cc=_seq.o) $(SEQ_SRC:.cc=_seq.o)

dirs:
	@mkdir -p .build_perf .build_instr .build_debug

clean:
	@rm -f *.o *~

distclean: clean
	@rm -fv $(TARGETS) .build_*/$(MCXX)_*.cc

# make anywhere you want using: make DESTDIR=/desired/path/here install
DESTDIR=bin
install:
	mkdir -p $(DESTDIR)
	-mv $(TARGETS) $(DESTDIR)/
