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
method=intradiff
extra_name=oneout
trainer=IntraDiffusionTrainer
cuda_device=0

(CUDA_VISIBLE_DEVICES=5 python MedSegDGSSL/tools/train.py \
--root  ${DATA} \
--trainer ${trainer} \
--source-domains ${D1} ${D2} ${D3} ${D4} ${D5} ${D6} \
--seed ${SEED} \
--fold 0 \
--config-file configs/trainers/${DATASET}/${DATASET}_${method}.yaml \
--output-dir /data/datasets/DGFramework/${DATASET}/output/dg/${method}_${extra_name}/fold_0 )

:'
(CUDA_VISIBLE_DEVICES=7 python MedSegDGSSL/tools/train.py \
--root  ${DATA} \
--trainer ${trainer} \
--source-domains ${D1} ${D2} ${D3} ${D4} ${D5} ${D6} \
--seed ${SEED} \
--fold 1 \
--config-file configs/trainers/${DATASET}/${DATASET}_${method}.yaml \
--output-dir /data/datasets/DGFramework/${DATASET}/output/dg/${method}_${extra_name}/fold_1 )

(CUDA_VISIBLE_DEVICES=7 python MedSegDGSSL/tools/train.py \
--root  ${DATA} \
--trainer ${trainer} \
--source-domains ${D1} ${D2} ${D3} ${D4} ${D5} ${D6} \
--seed ${SEED} \
--fold 2 \
--config-file configs/trainers/${DATASET}/${DATASET}_${method}.yaml \
--output-dir /data/datasets/DGFramework/${DATASET}/output/dg/${method}_${extra_name}/fold_2 )
'
wait
echo "Finished"