#ifndef ATTENTION_H
#define ATTENTION_H

#include <stdio.h>
#include "tensor.h"

// these two are for Attention Visualisation
extern int g_dump_attention;
extern FILE* g_attention_file;

/*
 * MultiHeadAttention block
 * h        = number of attention heads
 * d_model  = total model dimension
 * W_Q/K/V  = projection matrices for query, key, value
 * W_O      = output projection matrix
 */
typedef struct {
    int h;
    int d_model;
    Tensor* W_Q;
    Tensor* W_K;
    Tensor* W_V;
    Tensor* W_O;
} MultiHeadAttention;

Tensor* transpose(Tensor* K);
Tensor* attention(Tensor* Q, Tensor* K, Tensor* V, int causal);

// ---------- MHA RELATED STUFFS -------------

Tensor* get_head_slice(Tensor* X, int head, int d_k);
Tensor* concat_heads(Tensor** heads, int h, int d_k);
MultiHeadAttention* mha_create(int h, int d_model);
void mha_free(MultiHeadAttention* mha);
Tensor* mha_forward(MultiHeadAttention* mha, Tensor* Q, Tensor* K, Tensor* V, int causal);

void fill_random(Tensor* t);

#endif
