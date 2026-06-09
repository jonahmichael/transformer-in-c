#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    int rows;
    int cols;
    float *data;
} Tensor;

Tensor* tensor_create(int rows, int cols);
void tensor_free(Tensor* t);
float tensor_get(Tensor* t, int row, int col);
void tensor_set(Tensor* t, int row, int col, float val);

#endif