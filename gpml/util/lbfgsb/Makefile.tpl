
### 1) Choose Matlab configuration 
MATLAB_ROOT = ${Matlab_Root}
MEX_BIN     = $(MATLAB_ROOT)/bin/mex

### 2) Choose L-BFGS-B configuration
LBFGSB_OBJS = ${LBFGSB_obj}

### 3) Choose FORTRAN compiler
FCC    = ${Fortran_Comp}
FLIBS  = ${Fortran_Libs}
FFLAGS = -O3 -fPIC -fexceptions -Wall -g -Wno-uninitialized

### 4) Choose C++ compiler 
CXX    = ${Cpp_Comp}
CFLAGS = -O3 -fPIC -pthread -Wall -Werror -ansi -ffast-math -fomit-frame-pointer

###############################################################################

TARGET  = lbfgsb
OBJS    = $(LBFGSB_OBJS) matlabexception.o matlabscalar.o matlabstring.o \
          matlabmatrix.o arrayofmatrices.o program.o matlabprogram.o \
          lbfgsb.o

%.o: %.cpp
	$(CXX) $(CFLAGS) -I$(MATLAB_ROOT)/extern/include -o $@ -c $^

%.o: %.f
	$(FCC) $(FFLAGS) -o $@ -c $^

mex: $(OBJS)
	$(MEX_BIN) -cxx CXX=$(CXX) CC=$(CXX) FC=$(FCC) LD=$(CXX) $(FLIBS) -lm -O -output $(TARGET) $^

copy:
	mv $(TARGET).mex* ..

tidy:
	rm -f *.o

clean: tidy
	rm -f ../$(TARGET).mex*

all: mex copy tidy
