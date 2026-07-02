#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "math_ops.h"
#include "attention.h"
#include "encoder.h"

int main() {
    int d_model = 8, d_ff = 32, h = 2, seq_len = 3;

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

    Tensor* input = tensor_create(seq_len, d_model);
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < d_model; j++)
            tensor_set(input, i, j, (float)(i + j + 1));

    Tensor* output = encoder_forward(enc, input);

    printf("Encoder output (%d x %d):\n", output->rows, output->cols);
    for (int i = 0; i < output->rows; i++) {
        for (int j = 0; j < output->cols; j++)
            printf("%.4f ", tensor_get(output, i, j));
        printf("\n");
    }

    // Sanity check: verify layer norm properties
    printf("\nLayer Norm Sanity Check:\n");
    for (int i = 0; i < output->rows; i++) {
        float sum = 0.0f, sq_sum = 0.0f;
        for (int j = 0; j < output->cols; j++) {
            float val = tensor_get(output, i, j);
            sum += val;
            sq_sum += val * val;
        }
        float mean = sum / output->cols;
        float var = sq_sum / output->cols - mean * mean;
        printf("Row %d → mean: %.6f, variance: %.6f\n", i, mean, var);
    }

    tensor_free(input);
    tensor_free(output);
    encoder_free(enc);
    return 0;
}