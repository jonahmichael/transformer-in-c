#ifndef ATTENTON_H
#define ATTENTON_H


#include "tensor.h"

Tensor* transpose(Tensor* K);
Tensor* attention(Tensor* Q, Tensor* K, Tensor* V);

#endif