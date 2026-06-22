#include <math.h>
#include <stdlib.h>
#include "tensor.h"
#include "math_ops.h"
#include "encoder.h"
#include "attention.h"

EncoderBlock* encoder_create(int d_model, int d_ff, int h) {
    EncoderBlock* enc = malloc(sizeof(EncoderBlock));
    
    // MHA
    enc->mha = mha_create(h, d_model);
    
    // Layer norm parameters
    enc->gamma1 = malloc(sizeof(float) * d_model);
    enc->beta1  = malloc(sizeof(float) * d_model);
    enc->gamma2 = malloc(sizeof(float) * d_model);
    enc->beta2  = malloc(sizeof(float) * d_model);
    
    // FFN weights
    enc->W1 = tensor_create(d_model, d_ff);
    enc->W2 = tensor_create(d_ff, d_model);
    
    // FFN biases
    enc->b1 = malloc(sizeof(float) * d_ff);
    enc->b2 = malloc(sizeof(float) * d_model);
    
    // Config
    enc->d_model = d_model;
    enc->d_ff    = d_ff;
    
    return enc;
}

void encoder_free(EncoderBlock* enc) {
    mha_free(enc->mha);
    
    free(enc->gamma1);
    free(enc->beta1);
    free(enc->gamma2);
    free(enc->beta2);
    
    tensor_free(enc->W1);
    tensor_free(enc->W2);
    
    free(enc->b1);
    free(enc->b2);
    
    free(enc);
}

Tensor* encoder_forward(EncoderBlock* enc, Tensor* input) {
    // Step 1: Multi-head self-attention
    Tensor* mha_out = mha_forward(enc->mha, input, input, input);

    // Step 2: Residual + Layer Norm 1
    Tensor* add1 = mat_add(input, mha_out);
    tensor_free(mha_out);
    for (int i = 0; i < add1->rows; i++) {
        layer_norm(add1->data + i * add1->cols,
                   enc->gamma1, enc->beta1, add1->cols);
    }

    // Step 3: FFN - first linear layer
    Tensor* ff1 = mat_mul(add1, enc->W1);
    for (int i = 0; i < ff1->rows; i++)
        for (int j = 0; j < ff1->cols; j++)
            ff1->data[i * ff1->cols + j] += enc->b1[j];

    // Step 3.2: GELU activation
    for (int i = 0; i < ff1->rows * ff1->cols; i++)
        ff1->data[i] = gelu(ff1->data[i]);

    // Step 3.3: second linear layer
    Tensor* ff2 = mat_mul(ff1, enc->W2);
    tensor_free(ff1);
    for (int i = 0; i < ff2->rows; i++)
        for (int j = 0; j < ff2->cols; j++)
            ff2->data[i * ff2->cols + j] += enc->b2[j];

    // Step 4: Residual + Layer Norm 2
    Tensor* add2 = mat_add(add1, ff2);
    tensor_free(ff2);
    tensor_free(add1);
    for (int i = 0; i < add2->rows; i++) {
        layer_norm(add2->data + i * add2->cols,
                   enc->gamma2, enc->beta2, add2->cols);
    }

    return add2;
}