#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

lang="bh"

train_set="$lang"/"train"
valid_set="$lang"/"dev"
test_sets="$lang"/"test_bh_asru"
#test_sets="test_clean"

asr_config=conf/tuning/train_conformer_ctc_attn_1gpu_ctc0.yaml
inference_config=conf/decode_asr_ctc_attn-attnonlydecode.yaml
#inference_config=conf/decode_asr_ctc_attn.yaml
#inference_config=conf/decode_asr_ctc_attn-ctconlydecode.yaml
ctc_encoder_chk=exp/asr_train_conformer_ctc_raw_bh_char_sp/valid.cer_ctc.ave_3best.pth
#https://github.com/espnet/espnet/blob/master/egs2/mini_an4/asr1/transfer_learning.md

./asr.sh \
    --lang "$lang" \
    --ngpu 1 \
    --nj 20 \
    --stage 11 \
    --gpu_inference false \
    --inference_nj 20 \
    --token_type "char" \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --feats_type raw \
    --use_lm false \
    --inference_asr_model "valid.cer_ctc.best.pth" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --pretrained_model "$ctc_encoder_chk" \
    --asr_tag "ctc_attn_load-ctc-encoder-no-freeze-ctc0" \
