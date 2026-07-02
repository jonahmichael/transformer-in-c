#include <stdlib.h>
#include "transformer.h"

void transformer_init_random(Transformer* t) {
    for (int i = 0; i < t->N; i++) {
        // encoder weights
        fill_random(t->encoders[i]->mha->W_Q);
        fill_random(t->encoders[i]->mha->W_K);
        fill_random(t->encoders[i]->mha->W_V);
        fill_random(t->encoders[i]->mha->W_O);
        fill_random(t->encoders[i]->W1);
        fill_random(t->encoders[i]->W2);
        for (int j = 0; j < t->d_model; j++) {
            t->encoders[i]->gamma1[j] = 1.0f; t->encoders[i]->beta1[j] = 0.0f;
            t->encoders[i]->gamma2[j] = 1.0f; t->encoders[i]->beta2[j] = 0.0f;
        }

        // decoder weights
        fill_random(t->decoders[i]->masked_mha->W_Q);
        fill_random(t->decoders[i]->masked_mha->W_K);
        fill_random(t->decoders[i]->masked_mha->W_V);
        fill_random(t->decoders[i]->masked_mha->W_O);
        fill_random(t->decoders[i]->cross_mha->W_Q);
        fill_random(t->decoders[i]->cross_mha->W_K);
        fill_random(t->decoders[i]->cross_mha->W_V);
        fill_random(t->decoders[i]->cross_mha->W_O);
        fill_random(t->decoders[i]->W1);
        fill_random(t->decoders[i]->W2);
        for (int j = 0; j < t->d_model; j++) {
            t->decoders[i]->gamma1[j] = 1.0f; t->decoders[i]->beta1[j] = 0.0f;
            t->decoders[i]->gamma2[j] = 1.0f; t->decoders[i]->beta2[j] = 0.0f;
            t->decoders[i]->gamma3[j] = 1.0f; t->decoders[i]->beta3[j] = 0.0f;
        }
    }
}

Transformer* transformer_create(int N, int d_model, int d_ff, int h) {
    Transformer* t = malloc(sizeof(Transformer));
    t->encoders = malloc(N * sizeof(EncoderBlock*));
    for (int i = 0; i < N; i++)
        t->encoders[i] = encoder_create(d_model, d_ff, h);
    t->decoders = malloc(N * sizeof(DecoderBlock*));
    for (int i = 0; i < N; i++)
        t->decoders[i] = decoder_create(d_model, d_ff, h);
    t->N = N;
    t->d_model = d_model;
    t->d_ff = d_ff;
    t->h = h;
    return t;
}

void transformer_free(Transformer* t) {
    for (int i = 0; i < t->N; i++) encoder_free(t->encoders[i]);
    free(t->encoders);
    for (int i = 0; i < t->N; i++) decoder_free(t->decoders[i]);
    free(t->decoders);
    free(t);
}

Tensor* transformer_forward(Transformer* t, Tensor* src, Tensor* tgt) {
    Tensor* enc_out = src;
    for (int i = 0; i < t->N; i++) {
        Tensor* next = encoder_forward(t->encoders[i], enc_out);
        if (i > 0) tensor_free(enc_out);
        enc_out = next;
    }
    Tensor* dec_out = tgt;
    for (int i = 0; i < t->N; i++) {
        Tensor* next = decoder_forward(t->decoders[i], dec_out, enc_out);
        if (i > 0) tensor_free(dec_out);
        dec_out = next;
    }
    tensor_free(enc_out);
    return dec_out;
}