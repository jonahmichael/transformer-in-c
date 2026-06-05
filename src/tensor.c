#include <stdlib.h>
#include "tensor.h"

Tensor* tensor_create(int rows, int cols) {
    Tensor* t = malloc(sizeof(Tensor));
    t->data = malloc(rows * cols * sizeof(float));
    t->rows = rows;
    t->cols = cols;
    return t;
}

void tensor_free(Tensor* t) {
    free(t->data);
    free(t);
}

float tensor_get(Tensor* t, int row, int col) {
    return t->data[row * t->cols + col];
}

void tensor_set(Tensor* t, int row, int col, float val) {
    t->data[row * t->cols + col] = val;
}