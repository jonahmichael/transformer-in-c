#include <stdlib.h>
#include "tensor.h"

Tensor* tensor_create(int rows, int cols) {
    Tensor* t = malloc(sizeof(Tensor));
    t->data = calloc(rows * cols, sizeof(float));   // we use calloc or else, there will be uninitialized values in the tensor, which will cause problems in the attention mechanism whereas calloc initializes all values to zero, which is what we want...

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