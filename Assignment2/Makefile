NVCC = nvcc
CXXFLAGS = -O3 -Xcompiler -fopenmp

TARGET = a2
SRC = main.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
