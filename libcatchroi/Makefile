CC=gcc
CFLAGS=-g -O3 -fPIC -Wall -Wextra -std=gnu11
ARFLAGS=rcs

libs=libcatchroi.a libcatchroi.so
headers=catchroi.h

make:$(libs)
%.so:LDFLAGS+=-shared -pthread

CATCHROI_HOME?=$(abspath $(dir $(lastword $(MAKEFILE_LIST))))
DESTDIR?=$(CATCHROI_HOME)

lib%.a:%.o inject_err.o
	$(AR) $(ARFLAGS) $@ $^

lib%.so:%.o inject_err.o
	$(LINK.o) $^ -o $@

clean:
	@rm -vf $(libs)

install:$(libs) $(headers)
	@mkdir -p $(DESTDIR)/lib $(DESTDIR)/include
	@cp -v $(libs) $(DESTDIR)/lib
	@cp -v $(headers) $(DESTDIR)/include

uninstall:
	@rm -v $(addprefix $(DESTDIR)/lib/, $(libs)) $(addprefix $(DESTDIR)/include/, $(headers))
	@rmdir --ignore-fail-on-non-empty $(DESTDIR)/lib $(DESTDIR)/include

.PHONY: make clean install uninstall
