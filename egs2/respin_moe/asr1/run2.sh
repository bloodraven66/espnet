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
# test_sets="test_clean"
# dev_clean dev_other"
#test_sets="test_clean"

asr_config=conf/gs_moe/train_conformer_ctc_dense_moe_noMacaronMoe-1024dim_yesNormalMoe-1024dim-4experts_utt_mix-gs-moe-sum-soft.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --lang "$lang" \
    --ngpu 1 \
    --nj 48 \
    --stage 11 \
    --gpu_inference true \
    --inference_nj 32 \
    --token_type char \
    --max_wav_duration 30 \
    --audio_format "wav" \
    --feats_type raw_copy \
    --use_lm true \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --inference_asr_model "valid.cer_ctc.ave_3best.pth" \
    --inference_tag "soft_decode" \
    --lm_train_text "data/${train_set}/text" \
