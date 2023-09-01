#!/bin/bash
DATA=/data/bagcilab/datasets/zheyuan/DGFramework/Fundus/processed/2DRegionPosNeg
DATASET=Fundus
D1=Domain1
D2=Domain2
D3=Domain3
D4=Domain4

SEED=0

method=intradomain
trainer=IntraTrainer
cuda_device=0

(CUDA_VISIBLE_DEVICES=0 python MedSegDGSSL/tools/train.py \
--root  ${DATA} \
--trainer ${trainer} \
--source-domains ${D1} ${D2} ${D3} ${D4} ${D5} ${D6} \
--seed ${SEED} \
--fold 0 \
--config-file configs/trainers/${DATASET}/${DATASET}_${method}.yaml \
--output-dir /home/zze3980/Projects/DGFramework/output/${DATASET}/dg/${method}/fold_0 & )

(CUDA_VISIBLE_DEVICES=0 python MedSegDGSSL/tools/train.py \
--root  ${DATA} \
--trainer ${trainer} \
--source-domains ${D1} ${D2} ${D3} ${D4} ${D5} ${D6} \
--seed ${SEED} \
--fold 1 \
--config-file configs/trainers/${DATASET}/${DATASET}_${method}.yaml \
--output-dir /home/zze3980/Projects/DGFramework/output/${DATASET}/dg/${method}/fold_1 & )

(CUDA_VISIBLE_DEVICES=0 python MedSegDGSSL/tools/train.py \
--root  ${DATA} \
--trainer ${trainer} \
--source-domains ${D1} ${D2} ${D3} ${D4} ${D5} ${D6} \
--seed ${SEED} \
--fold 2 \
--config-file configs/trainers/${DATASET}/${DATASET}_${method}.yaml \
--output-dir /home/zze3980/Projects/DGFramework/output/${DATASET}/dg/${method}/fold_2 & )

wait
echo "Finished"