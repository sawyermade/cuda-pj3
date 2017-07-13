CC=nvcc
CFLAGS=-g -std=c++11 `pkg-config --cflags igraph`
INC=
LIB=`pkg-config --libs igraph`
BIN=
SRC=port-cpu.cpp
OBJ=$(SRC:.cpp=.o)
EXEC=pj3
LDFLAGS=


all: $(EXEC)

$(EXEC): $(OBJ)
	$(CC) $(CLFAGS) $(INC) -o $@ $^ $(LIB)
	
.cpp.o:
	$(CC) $(CFLAGS) $(INC) -c -o $@ $^

clean:
	rm -f $(OBJ) $(EXEC)