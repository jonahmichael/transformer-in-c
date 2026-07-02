#ifndef TRANSFORMER_H
#define TRANSFORMER_H

#include "encoder.h"
#include "decoder.h"

typedef struct {
    EncoderBlock** encoders;
    DecoderBlock** decoders;
    int N;
    int d_model;
    int d_ff;
    int h;
} Transformer;

Transformer* transformer_create(int N, int d_model, int d_ff, int h);
void transformer_init_random(Transformer* t);
void transformer_free(Transformer* t);
Tensor* transformer_forward(Transformer* t, Tensor* src, Tensor* tgt);

#endif