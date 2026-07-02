#ifndef DECODER_H
#define DECODER_H

#include "attention.h"
#include "math_ops.h"

typedef struct {
    MultiHeadAttention* masked_mha;
    MultiHeadAttention* cross_mha;
    float* gamma1; float* beta1;
    float* gamma2; float* beta2;
    float* gamma3; float* beta3;
    Tensor* W1;
    Tensor* W2;
    float* b1;
    float* b2;
    int d_model;
    int d_ff;
} DecoderBlock;

DecoderBlock* decoder_create(int d_model, int d_ff, int num_heads);
void decoder_free(DecoderBlock* dec);
Tensor* decoder_forward(DecoderBlock* dec, Tensor* target, Tensor* enc_output);

#endif