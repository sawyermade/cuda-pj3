CC=nvcc
CFLAGS=-std=c++11 `pkg-config --cflags igraph`
INC=
LIB=`pkg-config --libs igraph`
BIN=
SRC=proj3-lcm.cu
OBJ=$(SRC:.cpp=.o)
EXEC=proj3-lcm
NVFLAGS=-arch=sm_35


all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) $(CLFAGS) $(NVFLAGS) $(INC) -o $@ $^ $(LIB)
	
# .cpp.o:
# 	$(CC) $(CFLAGS) $(INC) -c -o $@ $^

clean:
	rm -f $(EXEC)