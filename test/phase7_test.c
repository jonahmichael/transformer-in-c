#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "math_ops.h"
#include "transformer.h"
#include "utils.h"

int main() {
    int N = 2, d_model = 8, d_ff = 32, h = 2, seq_len = 3;

    Transformer* t = transformer_create(N, d_model, d_ff, h);
    transformer_init_random(t);

    // Create source and target inputs
    Tensor* src = tensor_create(seq_len, d_model);
    Tensor* tgt = tensor_create(seq_len, d_model);
    for (int i = 0; i < seq_len; i++)
        for (int j = 0; j < d_model; j++) {
            tensor_set(src, i, j, (float)(i + j + 1));
            tensor_set(tgt, i, j, (float)(i + j + 2));
        }

    // Generate and print positional encoding
    Tensor* pe = positional_encoding(seq_len, d_model);
    printf("Positional Encoding (%d x %d):\n", pe->rows, pe->cols);
    for (int i = 0; i < pe->rows; i++) {
        for (int j = 0; j < pe->cols; j++)
            printf("%.4f ", tensor_get(pe, i, j));
        printf("\n");
    }

    // Add PE to src before transformer
    Tensor* src_pe = mat_add(src, pe);
    tensor_free(pe);

    // Run full transformer forward pass
    Tensor* out = transformer_forward(t, src_pe, tgt);

    printf("\nTransformer output (%d x %d):\n", out->rows, out->cols);
    for (int i = 0; i < out->rows; i++) {
        for (int j = 0; j < out->cols; j++)
            printf("%.4f ", tensor_get(out, i, j));
        printf("\n");
    }

    

    // dump input and output to file for python verification
    FILE* f = fopen("verify.txt", "w");
    fprintf(f, "INPUT\n");
    for (int i = 0; i < src_pe->rows; i++)
        for (int j = 0; j < src_pe->cols; j++)
            fprintf(f, "%.6f\n", tensor_get(src_pe, i, j));
    fprintf(f, "OUTPUT\n");
    for (int i = 0; i < out->rows; i++)
        for (int j = 0; j < out->cols; j++)
            fprintf(f, "%.6f\n", tensor_get(out, i, j));
    fclose(f);
    printf("Dumped to verify.txt\n");

    tensor_free(src);
    tensor_free(src_pe);
    tensor_free(tgt);
    tensor_free(out);
    transformer_free(t);
    return 0;
}