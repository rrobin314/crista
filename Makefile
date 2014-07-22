CBLASPATH=/usr/local/atlas-3.9.23
MPICC=mpicc
CC=gcc
CFLAGS=-Wall
IDFLAGS=-I$(CBLASPATH)/include/
LDFLAGS=-L$(CBLASPATH)/lib/ -latlas -lcblas -lm
CRISTASOURCE=src/runCrista.c src/cristaLib.c src/cristaLib.h
CVSOURCE=src/runCrossValidatedCrista.c src/CVcristaLib.c src/CVcristaLib.h

.PHONY: clean test

all: dataGenerator runCrista runCrossValidatedCrista

dataGenerator: src/dataGenerator.c
	$(CC) $(CFLAGS) $(IDFLAGS) $(LDFLAGS) -o dataGenerator src/dataGenerator.c

runCrista: src/runCrista.o src/cristaLib.o
	$(MPICC) $(CFLAGS) $(IDFLAGS) $(LDFLAGS) src/runCrista.o src/cristaLib.o -o runCrista 

runCrossValidatedCrista: src/runCrossValidatedCrista.o src/CVcristaLib.o
	$(MPICC) $(CFLAGS) $(IDFLAGS) $(LDFLAGS) src/runCrossValidatedCrista.o src/CVcristaLib.o -o runCrossValidatedCrista

src/runCrista.o: $(CRISTASOURCE)
	$(MPICC) $(CFLAGS) -c src/runCrista.c -o src/runCrista.o

src/runCrossValidatedCrista.o: $(CVSOURCE)
	$(MPICC) $(CFLAGS) -c src/runCrossValidatedCrista.c -o src/runCrossValidatedCrista.o

src/cristaLib.o: $(CRISTASOURCE)
	$(MPICC) $(CFLAGS) -c src/cristaLib.c -o src/cristaLib.o

src/CVcristaLib.o: $(CVSOURCE)
	$(MPICC) $(CFLAGS) -c src/CVcristaLib.c -o src/CVcristaLib.o

clean:
	rm -vf src/*.o 1000x1000_Test*.dat makeTestResults.txt dataGenerator runCrista runCrossValidatedCrista

test:
	./dataGenerator test/TESTdataGenParams
	mpirun -np 2 runCrista test/TESTmasterParams test/TESTslaveParams
