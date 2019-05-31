PLUGIN=catch_tasks_deps
NANOX_SRC=/usr/src/pm.bsc.es/nanox
# nanox install dir, with no trailing space
DESTDIR=/usr/local/lib64

CATCHROI_HOME ?= $(dir $(dir $(abspath $(lastword $(MAKEFILE_LIST)))))libcatchroi

CXX=g++

CPPFLAGS=-I$(NANOX_SRC)/src/plugins -I$(NANOX_SRC) -I$(NANOX_SRC)/src/arch/unix-os -I$(NANOX_SRC)/src/arch/cluster \
		 -I$(NANOX_SRC)/src/arch/common -I$(NANOX_SRC)/src/arch/gpu -I$(NANOX_SRC)/src/arch/fpga -I$(NANOX_SRC)/src/arch/opencl \
		 -I$(NANOX_SRC)/src/arch/smp -I$(NANOX_SRC)/src/arch/spu -I$(NANOX_SRC)/src/core -I$(NANOX_SRC)/src/support \
		 -I$(CATCHROI_HOME)/include -D_NANOS_INTERNAL -DLIBDIR='$(DESTDIR)' -include "new_decl.hpp" -DPIC

CXXFLAGS=-Wall -Wextra -Wshadow -Wmissing-declarations -Wno-unused-parameter -Wno-missing-field-initializers -Werror -std=c++11 -fPIC

LDFLAGS=-shared -Wl,-soname,libnanox-instrumentation-$(PLUGIN).so -L$(CATCHROI_HOME)/lib -Wl,-rpath,$(CATCHROI_HOME)/lib
LDLIBS=-pthread

debug debug_instr:CPPFLAGS+=-DNANOS_DEBUG_ENABLED
instr debug_instr:CPPFLAGS+=-DNANOS_INSTRUMENTATION_ENABLED

debug debug_instr:CXXFLAGS+=-O0 -g2 -fno-inline
instr perf:CXXFLAGS+=-O3


all:debug instr debug_instr perf

# NB: directory names must match those of install for -rpath
debug:       debug/libnanox-instrumentation-$(PLUGIN).so
instr:       instrumentation/libnanox-instrumentation-$(PLUGIN).so
debug_instr: instrumentation-debug/libnanox-instrumentation-$(PLUGIN).so
perf:        performance/libnanox-instrumentation-$(PLUGIN).so


%/libnanox_instrumentation_$(PLUGIN).o:$(PLUGIN).cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c -o $@ $^

%/libnanox-instrumentation-$(PLUGIN).so: %/libnanox_instrumentation_$(PLUGIN).o | dirs
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -Wl,-rpath,$(DESTDIR)/$(@D) -o $@ $(LDLIBS) $^


dirs:
	@mkdir -p instrumentation debug instrumentation-debug performance

clean:
	@rm -rf instrumentation/ debug/ instrumentation-debug/ performance/

install: | all
	@cp -v debug/libnanox-instrumentation-$(PLUGIN).so                 $(DESTDIR)/debug/libnanox-instrumentation-$(PLUGIN).so
	@cp -v instrumentation/libnanox-instrumentation-$(PLUGIN).so       $(DESTDIR)/instrumentation/libnanox-instrumentation-$(PLUGIN).so
	@cp -v instrumentation-debug/libnanox-instrumentation-$(PLUGIN).so $(DESTDIR)/instrumentation-debug/libnanox-instrumentation-$(PLUGIN).so
	@cp -v performance/libnanox-instrumentation-$(PLUGIN).so           $(DESTDIR)/performance/libnanox-instrumentation-$(PLUGIN).so

uninstall:
	@rm -vf $(DESTDIR)/debug/libnanox-instrumentation-$(PLUGIN).so
	@rm -vf $(DESTDIR)/instrumentation/libnanox-instrumentation-$(PLUGIN).so
	@rm -vf $(DESTDIR)/instrumentation-debug/libnanox-instrumentation-$(PLUGIN).so
	@rm -vf $(DESTDIR)/performance/libnanox-instrumentation-$(PLUGIN).so

.PHONY: all debug instr debug_instr perf dirs clean install uninstall
