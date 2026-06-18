#include <stdio.h>
#include "tensor.h"
#include "math_ops.h"
#include "attention.h"

int main() {
    // 2 heads, d_model=8, seq_len=3
    int h = 2, d_model = 8, seq_len = 3;

    // Create MHA
    MultiHeadAttention* mha = mha_create(h, d_model);

    fill_random(mha->W_Q);
    fill_random(mha->W_K);
    fill_random(mha->W_V);
    fill_random(mha->W_O);

    // Create Q, K, V of shape (seq_len, d_model)
    Tensor* Q = tensor_create(seq_len, d_model);
    Tensor* K = tensor_create(seq_len, d_model);
    Tensor* V = tensor_create(seq_len, d_model);

    // Fill with simple values
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < d_model; j++) {
            tensor_set(Q, i, j, (float)(i + j + 1));
            tensor_set(K, i, j, (float)(i + j + 1));
            tensor_set(V, i, j, (float)(i + j + 1));
        }

    // Run MHA forward pass
    Tensor* out = mha_forward(mha, Q, K, V);

    // Print output
    printf("MHA output (%d x %d):\n", out->rows, out->cols);
    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++)
            printf("%.4f ", tensor_get(out, i, j));
        printf("\n");
    }

    tensor_free(Q);
    tensor_free(K);
    tensor_free(V);
    tensor_free(out);
    mha_free(mha);
    return 0;
}