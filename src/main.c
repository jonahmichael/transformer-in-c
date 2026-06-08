#include <stdio.h>
#include "tensor.h"
#include "math_ops.h"
#include "attention.h"

int main() {
    // Create Q, K, V all shape (2, 4)
    // seq_len=2, d_k=4
    Tensor* Q = tensor_create(2, 4);
    Tensor* K = tensor_create(2, 4);
    Tensor* V = tensor_create(2, 4);

    // Fill with simple values
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 4; j++) {
            tensor_set(Q, i, j, (float)(i + j + 1));
            tensor_set(K, i, j, (float)(i + j + 1));
            tensor_set(V, i, j, (float)(i + j + 1));
        }
    }

    // Run attention
    Tensor* out = attention(Q, K, V);

    // Print output
    printf("Attention output:\n");
    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++) {
            printf("%.4f ", tensor_get(out, i, j));
        }
        printf("\n");
    }

    tensor_free(Q);
    tensor_free(K);
    tensor_free(V);
    tensor_free(out);
    return 0;
}