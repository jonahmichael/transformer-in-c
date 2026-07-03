#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "math_ops.h"
#include "transformer.h"

int main() {
    int N = 2, d_model = 8, d_ff = 32, h = 2, seq_len = 3;

    // Create full transformer (2 encoder + 2 decoder layers)
    Transformer* t = transformer_create(N, d_model, d_ff, h);

    transformer_init_random(t);  // Initialize weights randomly


    // Create source and target inputs
    Tensor* src = tensor_create(seq_len, d_model);
    Tensor* tgt = tensor_create(seq_len, d_model);
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < d_model; j++) {
            tensor_set(src, i, j, (float)(i + j + 1));
            tensor_set(tgt, i, j, (float)(i + j + 2));
        }

    // Run full transformer forward pass
    Tensor* out = transformer_forward(t, src, tgt);

    printf("Transformer output (%d x %d):\n", out->rows, out->cols);
    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++)
            printf("%.4f ", tensor_get(out, i, j));
        printf("\n");
    }

    tensor_free(src);
    tensor_free(tgt);
    tensor_free(out);
    transformer_free(t);
    return 0;
}