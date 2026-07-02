#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "math_ops.h"
#include "attention.h"
#include "encoder.h"
#include "decoder.h"

int main() {
    int d_model = 8, d_ff = 32, h = 2, seq_len = 3;

    // --- ENCODER ---
    EncoderBlock* enc = encoder_create(d_model, d_ff, h);
    fill_random(enc->mha->W_Q);
    fill_random(enc->mha->W_K);
    fill_random(enc->mha->W_V);
    fill_random(enc->mha->W_O);
    fill_random(enc->W1);
    fill_random(enc->W2);
    for (int i = 0; i < d_model; i++) {
        enc->gamma1[i] = 1.0f; enc->beta1[i] = 0.0f;
        enc->gamma2[i] = 1.0f; enc->beta2[i] = 0.0f;
    }
    Tensor* src = tensor_create(seq_len, d_model);
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < d_model; j++)
            tensor_set(src, i, j, (float)(i + j + 1));
    Tensor* enc_out = encoder_forward(enc, src);
    printf("Encoder output (%d x %d):\n", enc_out->rows, enc_out->cols);
    for (int i = 0; i < enc_out->rows; i++) {
        for (int j = 0; j < enc_out->cols; j++)
            printf("%.4f ", tensor_get(enc_out, i, j));
        printf("\n");
    }

    // --- DECODER ---
    DecoderBlock* dec = decoder_create(d_model, d_ff, h);
    fill_random(dec->masked_mha->W_Q);
    fill_random(dec->masked_mha->W_K);
    fill_random(dec->masked_mha->W_V);
    fill_random(dec->masked_mha->W_O);
    fill_random(dec->cross_mha->W_Q);
    fill_random(dec->cross_mha->W_K);
    fill_random(dec->cross_mha->W_V);
    fill_random(dec->cross_mha->W_O);
    fill_random(dec->W1);
    fill_random(dec->W2);
    for (int i = 0; i < d_model; i++) {
        dec->gamma1[i] = 1.0f; dec->beta1[i] = 0.0f;
        dec->gamma2[i] = 1.0f; dec->beta2[i] = 0.0f;
        dec->gamma3[i] = 1.0f; dec->beta3[i] = 0.0f;
    }
    Tensor* tgt = tensor_create(seq_len, d_model);
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < d_model; j++)
            tensor_set(tgt, i, j, (float)(i + j + 2));
    Tensor* dec_out = decoder_forward(dec, tgt, enc_out);
    printf("\nDecoder output (%d x %d):\n", dec_out->rows, dec_out->cols);
    for (int i = 0; i < dec_out->rows; i++) {
        for (int j = 0; j < dec_out->cols; j++)
            printf("%.4f ", tensor_get(dec_out, i, j));
        printf("\n");
    }

    tensor_free(src);
    tensor_free(enc_out);
    tensor_free(tgt);
    tensor_free(dec_out);
    encoder_free(enc);
    decoder_free(dec);
    return 0;
}