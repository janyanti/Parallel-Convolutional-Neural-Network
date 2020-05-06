EXECUTABLE := cnn
LDFLAGS=-L/usr/local/cuda-10.2/lib64/ -lcudart
CU_FILES   := dcnn.cu matrix.cu
CU_DEPS    :=
CC_FILES   := main.cpp parse_file.cpp
LOGS	   := logs

all: $(EXECUTABLE)

###########################################################

OBJDIR=objs
CXX=g++ -m64
CXXFLAGS=-Og -g -std=c++1y
#LIBS += GL glut cudart

LDLIBS  := $(addprefix -l, $(LIBS))
LDFRAMEWORKS := $(addprefix -framework , $(FRAMEWORKS))

NVCC=nvcc
NVCCFLAGS=-O3 -m64 --gpu-architecture compute_61

OBJS=$(OBJDIR)/main.o $(OBJDIR)/matrix.o $(OBJDIR)/dcnn.o $(OBJDIR)/parse_file.o

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
