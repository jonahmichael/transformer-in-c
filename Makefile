CC = gcc
CFLAGS = -Wall -Wextra
SRC = src/tensor.c src/math_ops.c src/attention.c src/encoder.c src/main.c
OUT = transformer

build:
	$(CC) $(CFLAGS) $(SRC) -o $(OUT) -lm

clean:
	rm -f $(OUT)
