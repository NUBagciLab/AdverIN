#!/bin/bash
DATA=/data/datasets/DGFramework/ProstateMRI/processed/2DSlice3
DATASET=ProstateMRI
D1=BIDMC
D2=BMC
D3=HK
D4=I2CVB
D5=UCL
D6=RUNMC

SEED=0
method=intra_domain
cuda_device=0

(CUDA_VISIBLE_DEVICES=0 python MedSegDGSSL/tools/train.py \
--root  ${DATA} \
--trainer IntraTrainer \
--source-domains ${D1} ${D2} ${D3} ${D4} ${D5} ${D6} \
--seed ${SEED} \
--fold 0 \
--config-file configs/trainers/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/fold_0 &)

(CUDA_VISIBLE_DEVICES=1 python MedSegDGSSL/tools/train.py \
--root  ${DATA} \
--trainer IntraTrainer \
--source-domains ${D1} ${D2} ${D3} ${D4} ${D5} ${D6}\
--seed ${SEED} \
--fold 1 \
--config-file configs/trainers/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/fold_1 & )

(CUDA_VISIBLE_DEVICES=3 python MedSegDGSSL/tools/train.py \
--root  ${DATA} \
--trainer IntraTrainer \
--source-domains ${D1} ${D2} ${D3} ${D4} ${D5} ${D6}\
--seed ${SEED} \
--fold 2 \
--config-file configs/trainers/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/fold_2 )

wait
echo "Finished"