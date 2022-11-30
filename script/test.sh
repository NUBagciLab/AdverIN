#!/bin/bash
DATA=./DATA
DATASET=ProstateMRI
D1=BIDMC
D2=BMC
D3=HK
SEED=0
method=uncertainty
cuda_device=0

(CUDA_VISIBLE_DEVICES=$cuda_device python MedSegDGSSL/tools/train.py \
--root /home/zze3980/project/AdverHistAug/data/ProstateMRI/processed/train3D \
--trainer Vanilla \
--uncertainty 0.5 \
--source-domains ${D1} ${D1} \
--target-domains ${D3} \
--seed ${SEED} \
--dataset-config-file configs/datasets/${DATASET}.yaml \
--config-file configs/trainers/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D1})
