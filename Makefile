CC = nvcc
CFLAGS = --expt-relaxed-constexpr -Wno-deprecated-gpu-targets
LDFLAGS = -lncurses -ltinfo
TARGET = cudagol
SRC = main.cu

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
