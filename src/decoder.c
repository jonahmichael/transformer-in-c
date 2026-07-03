#include <math.h>
#include <stdlib.h>
#include "decoder.h"

DecoderBlock* decoder_create(int d_model, int d_ff, int num_heads) {
    DecoderBlock* decoder = calloc(sizeof(DecoderBlock));
    decoder->masked_mha = mha_create(num_heads, d_model);
    decoder->cross_mha  = mha_create(num_heads, d_model);
    decoder->gamma1 = calloc(d_model * sizeof(float));
    decoder->beta1  = calloc(d_model * sizeof(float));
    decoder->gamma2 = calloc(d_model * sizeof(float));
    decoder->beta2  = calloc(d_model * sizeof(float));
    decoder->gamma3 = calloc(d_model * sizeof(float));
    decoder->beta3  = calloc(d_model * sizeof(float));
    decoder->W1 = tensor_create(d_model, d_ff);
    decoder->W2 = tensor_create(d_ff, d_model);
    decoder->b1 = calloc(d_ff * sizeof(float));
    decoder->b2 = calloc(d_model * sizeof(float));
    decoder->d_model = d_model;
    decoder->d_ff    = d_ff;
    return decoder;
}

void decoder_free(DecoderBlock* dec) {
    mha_free(dec->masked_mha);
    mha_free(dec->cross_mha);
    free(dec->gamma1); free(dec->beta1);
    free(dec->gamma2); free(dec->beta2);
    free(dec->gamma3); free(dec->beta3);
    tensor_free(dec->W1);
    tensor_free(dec->W2);
    free(dec->b1);
    free(dec->b2);
    free(dec);
}

Tensor* decoder_forward(DecoderBlock* dec, Tensor* target, Tensor* enc_output) {
    Tensor* mha1_out = mha_forward(dec->masked_mha, target, target, target,1);
    Tensor* add1 = mat_add(target, mha1_out);
    tensor_free(mha1_out);
    for (int i = 0; i < add1->rows; i++)
        layer_norm(add1->data + i * add1->cols, dec->gamma1, dec->beta1, add1->cols);

    Tensor* mha2_out = mha_forward(dec->cross_mha, add1, enc_output, enc_output, 0);
    Tensor* add2 = mat_add(add1, mha2_out);
    tensor_free(mha2_out);
    tensor_free(add1);
    for (int i = 0; i < add2->rows; i++)
        layer_norm(add2->data + i * add2->cols, dec->gamma2, dec->beta2, add2->cols);

    Tensor* ff1 = mat_mul(add2, dec->W1);
    for (int i = 0; i < ff1->rows; i++)
        for (int j = 0; j < ff1->cols; j++)
            ff1->data[i * ff1->cols + j] += dec->b1[j];
    for (int i = 0; i < ff1->rows * ff1->cols; i++)
        ff1->data[i] = gelu(ff1->data[i]);
    Tensor* ff2 = mat_mul(ff1, dec->W2);
    tensor_free(ff1);
    for (int i = 0; i < ff2->rows; i++)
        for (int j = 0; j < ff2->cols; j++)
            ff2->data[i * ff2->cols + j] += dec->b2[j];
    Tensor* add3 = mat_add(add2, ff2);
    tensor_free(ff2);
    tensor_free(add2);
    for (int i = 0; i < add3->rows; i++)
        layer_norm(add3->data + i * add3->cols, dec->gamma3, dec->beta3, add3->cols);

    return add3;
}