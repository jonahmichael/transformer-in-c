#ifndef ENCODER_H
#define ENCODER_H
#include "tensor.h"
#include "attention.h"

typedef struct {
    MultiHeadAttention* mha;

    // Layer norm 1 (after MHA)
    float* gamma1;
    float* beta1;

    // Layer norm 2 (after FFN)
    float* gamma2;
    float* beta2;

    // Feed forward network weights
    Tensor* W1;   // (d_model, d_ff)
    Tensor* W2;   // (d_ff, d_model)
    float* b1;    // bias for layer 1
    float* b2;    // bias for layer 2

    int d_model;
    int d_ff;
} EncoderBlock;

EncoderBlock* encoder_create(int d_model, int d_ff, int h);
void encoder_free(EncoderBlock* enc);
Tensor* encoder_forward(EncoderBlock* enc, Tensor* input);

#endif