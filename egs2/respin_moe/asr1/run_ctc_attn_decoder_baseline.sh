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

asr_config=conf/tuning/train_conformer_ctc_attn_1gpu.yaml
inference_config=conf/decode_asr_ctc_attn-attnonlydecode.yaml

./asr.sh \
    --lang "$lang" \
    --ngpu 1 \
    --nj 20 \
    --stage 12 \
    --gpu_inference true \
    --inference_nj 20 \
    --token_type "char" \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --feats_type raw \
    --use_lm false \
    --inference_asr_model "valid.cer_ctc.ave.pth" \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
