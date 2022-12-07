#!/bin/bash
DATA=/data/datasets/zheyuan/DGFramework/Fundus/processed/2D
DATASET=Fundus
D1=Domain1
D2=Domain2
D3=Domain3
D4=Domain4

SEED=0
method=instance_norm
cuda_device=0

(CUDA_VISIBLE_DEVICES=0 python MedSegDGSSL/tools/train.py \
--root  ${DATA} \
--trainer Vanilla \
--source-domains ${D1} ${D2} ${D3} \
--target-domains ${D4} \
--seed ${SEED} \
--config-file configs/trainers/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D4} & )

(CUDA_VISIBLE_DEVICES=1 python MedSegDGSSL/tools/train.py \
--root  ${DATA} \
--trainer Vanilla \
--source-domains ${D1} ${D2} ${D4} \
--target-domains ${D3} \
--seed ${SEED} \
--config-file configs/trainers/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D3} & )

(CUDA_VISIBLE_DEVICES=3 python MedSegDGSSL/tools/train.py \
--root  ${DATA} \
--trainer Vanilla \
--source-domains ${D1} ${D3} ${D4} \
--target-domains ${D2} \
--seed ${SEED} \
--config-file configs/trainers/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D2} & )

(CUDA_VISIBLE_DEVICES=4 python MedSegDGSSL/tools/train.py \
--root  ${DATA} \
--trainer Vanilla \
--source-domains ${D2} ${D3} ${D4} \
--target-domains ${D1} \
--seed ${SEED} \
--config-file configs/trainers/${DATASET}.yaml \
--output-dir output/dg/${DATASET}/${method}/${D1} )


wait
echo "Finished"