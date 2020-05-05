MPI=-DMPI
MPICC= mpicxx
LDFLAGS = -lm

CC_FILES   := dcnn.cpp main.cpp matrix.cpp parse_file.cpp instrument.cpp
LOGS	   := logs

all: cnn-seq cnn-mpi

###########################################################

INSTRUMENT=1
DEBUG=0

OBJDIR=objs
CXX=g++ -m64
WFLGAS= -Wall -Wextra -Werror
CXXFLAGS=-O3 -g -std=c++1y -DTRACK=$(INSTRUMENT)

OBJS=$(OBJDIR)/main.o $(OBJDIR)/matrix.o $(OBJDIR)/dcnn.o $(OBJDIR)/parse_file.o $(OBJDIR)/instrument.o

.PHONY: dirs clean

default: all

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ cnn-seq cnn-mpi $(LOGS) *.ppm

cnn-seq: dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)


cnn-mpi: $(CC_FILES)
	$(MPICC) $(CXXFLAGS) $(MPI) -o $@ $(CC_FILES) $(LDFLAGS)

$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@


$(OBJDIR_MPI)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) $(MPI) -c -o $@

run-mpi:
		 mpiexec -np 4 -map-by core -bind-to core ./cnn-mpi ../data/mnist-train-data ../data/mnist-train-labels ../data/mnist-test-data ../data/mnist-test-labels
