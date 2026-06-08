
#include <math.h>
#include <stdlib.h>
#include "tensor.h"
#include "math_ops.h"
#include "attention.h"



Tensor* transpose(Tensor* K) {
    Tensor* out = tensor_create(K->cols, K->rows);
    for (int i = 0; i < K->rows; i++) {
        for (int j = 0; j < K->cols; j++) {
            tensor_set(out, j, i, tensor_get(K, i, j));
        }
    }
    return out;
}

/*
 * Scaled dot-product attention
 * Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
 */
Tensor* attention(Tensor* Q, Tensor* K, Tensor* V) {
    // Step 1: QK^T
    Tensor* Kt = transpose(K);
    Tensor* scores = mat_mul(Q, Kt);

    // Step 2: scale by √d_k
    for (int i = 0; i < scores->rows; i++) {
        for (int j = 0; j < scores->cols; j++) {
            tensor_set(scores, i, j,
                tensor_get(scores, i, j) / sqrtf((float)K->cols));
        }
    }

    // Step 3: softmax each row
    for (int i = 0; i < scores->rows; i++) {
        softmax(scores->data + i * scores->cols, scores->cols);
    }

    // Step 4: multiply by V
    Tensor* out = mat_mul(scores, V);

    // Free intermediates
    tensor_free(Kt);
    tensor_free(scores);

    return out;
}