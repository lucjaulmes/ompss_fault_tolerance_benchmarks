CXX=g++
MCXX=smpcxx
CXXFLAGS=-Wall -Wextra -std=gnu++11 -O$(O) -fno-optimize-sibling-calls $(OMPSSFLAGS)
OMPSSFLAGS=--ompss --Wn,-Wno-unused-parameter,-Wno-unused-but-set-parameter,-Wno-unused-local-typedefs
CPPFLAGS=-I$(CATCHROI_HOME)/include
LDFLAGS=-L$(CATCHROI_HOME)/lib -Wl,-rpath,$(CATCHROI_HOME)/lib
LDLIBS=-lcatchroi

CATCHROI_HOME?=$(dir $(abspath $(dir $(lastword $(MAKEFILE_LIST)))))libcatchroi
DESTDIR?=$(CATCHROI_HOME)


# target recipes. NB most objects are seq
OMP_SRC:=algorithms.cc heat.cc
SEQ_SRC:=algorithms_inner.cc matrix.cc misc.cc

BINARIES=heat heat_seq heat_instr heat_debug


all: $(BINARIES)

perf: heat
instr: heat_instr
debug: heat_debug
seq: heat_seq


# optimization levels: default and debug
O=3
heat_debug:CXXFLAGS+=-g -rdynamic
heat_debug:O=0

%_seq   %_seq.o:  OMPSSFLAGS=-Wno-unknown-pragmas
heat    %_perf.o: OMPSSFLAGS+=--keep-all-files --output-dir=.build_perf
%_instr %_instr.o:OMPSSFLAGS+=--keep-all-files --output-dir=.build_instr --instrument
%_debug %_debug.o:OMPSSFLAGS+=--keep-all-files --output-dir=.build_debug --debug


# object recipes
%_seq.o: %.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $^

%_perf.o: %.cc | dirs
	$(MCXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

%_instr.o: %.cc | dirs
	$(MCXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<

%_debug.o: %.cc | dirs
	$(MCXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $<


heat_seq: $(SEQ_SRC:.cc=_seq.o) $(OMP_SRC:.cc=_seq.o)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

heat: $(SEQ_SRC:.cc=_seq.o) $(OMP_SRC:.cc=_perf.o)
	$(MCXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

heat_instr: $(SEQ_SRC:.cc=_seq.o) $(OMP_SRC:.cc=_instr.o)
	$(MCXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)




.PHONY: clean distclean all default install dirs seq debug instr perf
.INTERMEDIATE:$(OMP_SRC:.cc=_debug.o) $(OMP_SRC:.cc=_instr.o) $(OMP_SRC:.cc=_perf.o) $(OMP_SRC:.cc=_seq.o) $(SEQ_SRC:.cc=_seq.o)

dirs:
	@mkdir -p .build_perf .build_instr .build_debug

clean:
	@rm -f *.o *~

distclean: clean
	@rm -fv $(BINARIES) .build_*/$(MCXX)_*.cc

# make anywhere you want using: make DESTDIR=/desired/path/here install
DESTDIR=bin
install:
	mkdir -p $(DESTDIR)
	-mv $(BINARIES) $(DESTDIR)/
