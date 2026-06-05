#ifndef MATH_OPS_H
#define MATH_OPS_H

#include "tensor.h"

Tensor* mat_mul(Tensor* A, Tensor* B);
Tensor* mat_add(Tensor* A, Tensor* B);
void softmax(float* x, int len);
void layer_norm(float* x, float* gamma, float* beta, int len);
float gelu(float x);

#endif