#ifndef UTILS_H
#define UTILS_H

#include "tensor.h"
#include "math_ops.h"

Tensor* positional_encoding(int seq_len, int d_model);

#endif