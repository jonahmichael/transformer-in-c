#include <stdio.h>
#include "tensor.h"
#include "math_ops.h"

int main() {
    // Create a 2x3 matrix A
    Tensor* A = tensor_create(2, 3);
    tensor_set(A, 0, 0, 1); tensor_set(A, 0, 1, 2); tensor_set(A, 0, 2, 3);
    tensor_set(A, 1, 0, 4); tensor_set(A, 1, 1, 5); tensor_set(A, 1, 2, 6);

    // Create a 3x2 matrix B
    Tensor* B = tensor_create(3, 2);
    tensor_set(B, 0, 0, 7);  tensor_set(B, 0, 1, 8);
    tensor_set(B, 1, 0, 9);  tensor_set(B, 1, 1, 10);
    tensor_set(B, 2, 0, 11); tensor_set(B, 2, 1, 12);

    // Multiply A x B → should give 2x2 matrix
    Tensor* C = mat_mul(A, B);

    printf("Result of mat_mul:\n");
    for (int i = 0; i < C->rows; i++) {
        for (int j = 0; j < C->cols; j++) {
            printf("%.1f ", tensor_get(C, i, j));
        }
        printf("\n");
    }

    tensor_free(A);
    tensor_free(B);
    tensor_free(C);
    return 0;
}