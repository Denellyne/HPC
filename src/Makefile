CXX =gcc
CFLAGS =-O3 -march=native -lm -ffast-math -funroll-loops
INCLUDES :=$(shell find . -name '*.c')
OBJ :=$(patsubst %.c,%.o,$(INCLUDES))
DEBUG =-g -pg -fsanitize=address,leak -Wall -Wextra -pedantic
OUT =../bin/main
CORES =./cores/core.*

all: br
cr : clean build run
br : build run


build: $(OBJ)
	@echo "Linking..."
	@mkdir -p ../bin
	$(CXX) $(CFLAGS) $(OBJ) -o $(OUT) 

%.o: %.c 
	@echo "Compiling..."
	$(CXX) $(CFLAGS) -c $< -o $@  	
	@echo "Compiled:" $@


run: $(OBJ)
	@$(OUT)

debug: 
	@echo "Compiling..."
	@mkdir -p ../bin
	$(CXX) $(DEBUG) $(CFLAGS) $(INCLUDES) -o $(OUT)


dr:
	@echo "Compiling..."
	@mkdir -p ../bin
	$(CXX) $(DEBUG) $(CFLAGS) $(INCLUDES) -o $(OUT)
	@$(OUT)

clean:
	@echo "Deleting..."
	@rm -f $(OBJ) $(OUT) $(TEST) $(CORES)
