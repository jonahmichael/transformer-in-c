#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include "math_ops.h"
#include "transformer.h"
#include "attention.h"
#include "utils.h"
#include "train.h"

int main() {
    int N = 2, d_model = 8, d_ff = 32, h = 2, seq_len = 4;
    int vocab_size = 10;

    // create transformer
    Transformer* t = transformer_create(N, d_model, d_ff, h);
    transformer_init_random(t);

    // create embedding and input
    Tensor* emb = tensor_create(vocab_size, d_model);
    fill_random(emb);

    int tokens[4] = {1, 3, 2, 4};
    Tensor* src = embedding_lookup(tokens, emb, seq_len);
    Tensor* pe  = positional_encoding(seq_len, d_model);
    Tensor* src_pe = mat_add(src, pe);

    // enable attention dumping
    g_dump_attention = 1;
    g_attention_file = fopen("attention_weights.csv", "w");
    fprintf(g_attention_file, "row,col,weight\n");

    // run forward pass — attention weights get dumped automatically
    Tensor* out = transformer_forward(t, src_pe, src_pe);

    // close dump file
    fclose(g_attention_file);
    g_dump_attention = 0;

    printf("attention weights saved to attention_weights.csv\n");

    tensor_free(src);
    tensor_free(pe);
    tensor_free(src_pe);
    tensor_free(out);
    tensor_free(emb);
    transformer_free(t);
    return 0;
}