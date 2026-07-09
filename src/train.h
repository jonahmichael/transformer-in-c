#ifndef TRAIN_H
#define TRAIN_H

#include "tensor.h"
#include "transformer.h"

Tensor* embedding_lookup(int* tokens, Tensor* emb, int seq_len);
Tensor* linear_project(Tensor* decoder_out, Tensor* W_out);
float cross_entropy_loss(Tensor* logits, int* targets, int seq_len);
void train_copy_task(int epochs);

#endif