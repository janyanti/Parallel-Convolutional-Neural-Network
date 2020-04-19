EXECUTABLE := cnn
LDFLAGS=-L/usr/local/depot/cuda-10.2/lib64/ -lcudart
CU_FILES   := cudaRenderer.cu
CU_DEPS    :=
CC_FILES   := main.cpp matrix.cpp 
LOGS	   	 := logs

all: $(EXECUTABLE)

###########################################################

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-O3 -Wall -g -std=c++11

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc

OBJS=$(OBJDIR)/main.o $(OBJDIR)/matrix.o

.PHONY: dirs clean

default: $(EXECUTABLE)

dirs:
		mkdir -p $(OBJDIR)/

clean:
		rm -rf $(OBJDIR) *~ $(EXECUTABLE) $(LOGS) *.ppm

$(EXECUTABLE): dirs $(OBJS)
		$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS) $(LDLIBS) $(LDFRAMEWORKS)


$(OBJDIR)/%.o: %.cpp
		$(CXX) $< $(CXXFLAGS) -c -o $@

$(OBJDIR)/%.o: %.cu
		$(NVCC) $< $(NVCCFLAGS) -c -o $@
