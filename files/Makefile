OPTS = -Wall
CC = g++
LDFLAGS = -Lecos -lm
CFLAGS = -Wall -fPIC
INCLUDE = -Iecos/include  -Iecos/external/SuiteSparse_config -I/usr/include/python3.9 -I../extern/pybind11/include



bindings: bindings.o codegen.a ecos/libecos.a lib.o
	g++ -O3 $(CFLAGS) -shared -std=c++11 $(INCLUDE) $^ -o gfold`python3-config --extension-suffix`

lib.o: lib.cpp
	g++ $(CFLAGS) $(INCLUDE) -c $<

bindings.o: bindings.cpp
	g++ $(CFLAGS) $(INCLUDE) -c $<

codegen.a:  solver_intf.o codegen.o linop.o param.o lib.o
	ar rcs $@ $^

solver_intf.o: solver_intf.c
	$(CC) $(CFLAGS) $(INCLUDE) -c $<

codegen.o: codegen.c codegen.h
	$(CC) $(CFLAGS) $(INCLUDE) -c $<

linop.o: linop.c codegen.h
	$(CC) $(CFLAGS) $(INCLUDE) -c $<

param.o: param.c codegen.h
	$(CC) $(CFLAGS) $(INCLUDE) -c $<

ecos/libecos.a:
	cd ecos && make libecos.a

.PHONY: clean
clean:
	cd ecos && make clean
	rm -rf *.o example_problem
	rm -rf *.a

.PHONY: re
re: clean example_problem
