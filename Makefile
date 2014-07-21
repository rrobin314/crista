CBLASPATH=/usr/local/atlas-3.9.23
MPICC=mpicc
CC=gcc
CFLAGS=-Wall
IDFLAGS=-I$(CBLASPATH)/include/
LDFLAGS=-L$(CBLASPATH)/lib/ -latlas -lcblas -lm

.PHONY: clean test

all: dataGenerator runCrista runCrossValidatedCrista

dataGenerator: 
	$(CC) $(CFLAGS) $(IDFLAGS) $(LDFLAGS) -o dataGenerator src/dataGenerator.c

runCrista: runCrista.o cristaLib.o
	$(MPICC) $(CFLAGS) $(IDFLAGS) $(LDFLAGS) src/runCrista.o src/cristaLib.o -o runCrista 

runCrossValidatedCrista: runCrossValidatedCrista.o CVcristaLib.o
	$(MPICC) $(CFLAGS) $(IDFLAGS) $(LDFLAGS) src/runCrossValidatedCrista.o src/CVcristaLib.o -o runCrossValidatedCrista

runCrista.o:
	$(MPICC) $(CFLAGS) -c src/runCrista.c -o src/runCrista.o

runCrossValidatedCrista.o:
	$(MPICC) $(CFLAGS) -c src/runCrossValidatedCrista.c -o src/runCrossValidatedCrista.o

cristaLib.o:
	$(MPICC) $(CFLAGS) -c src/cristaLib.c -o src/cristaLib.o

CVcristaLib.o:
	$(MPICC) $(CFLAGS) -c src/CVcristaLib.c -o src/CVcristaLib.o

clean:
	rm -vf src/*.o dataGenerator runCrista runCrossValidatedCrista

test:
	./dataGenerator test/TESTdataGenParams
	mpirun -np 2 runCrista test/TESTmasterParams test/TESTslaveParams
