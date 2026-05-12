NVCC = nvcc
CXXFLAGS = -arch=sm_35 -O3 -Xcompiler -fopenmp

TARGET = a2
SRC = main.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) $(SRC) -o $(TARGET)

clean:
	rm -f $(TARGET)
