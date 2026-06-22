#ifndef ENCODER_H
#define ENCODER_H

#include "attention.h"
#include "math_ops.h"

typedef struct {
    MultiHeadAttention* mha;
    float* gamma1;
    float* beta1;
    float* gamma2;
    float* beta2;
    Tensor* W1;
    Tensor* W2;
    float* b1;
    float* b2;
    int d_model;
    int d_ff;
} EncoderBlock;

EncoderBlock* encoder_create(int d_model, int d_ff, int h);
void encoder_free(EncoderBlock* enc);
Tensor* encoder_forward(EncoderBlock* enc, Tensor* input);

#endif
