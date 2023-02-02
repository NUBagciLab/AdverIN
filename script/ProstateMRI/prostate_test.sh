#!/bin/bash
DATA=/data/datasets/DGFramework/ProstateMRI/processed/2DSlice3Region
DATASET=ProstateMRI
D1=BIDMC
D2=BMC
D3=HK
D4=I2CVB
D5=UCL
D6=RUNMC

SEED=0
method=diff
extra_name=adamw
trainer=DiffusionTrainer
cuda_device=0

(CUDA_VISIBLE_DEVICES=0 python MedSegDGSSL/tools/train.py \
--root  ${DATA} \
--trainer ${trainer} \
--source-domains ${D1} ${D2} ${D3} ${D4} ${D5} \
--target-domains ${D6} \
--seed ${SEED} \
--config-file configs/trainers/${DATASET}/${DATASET}_${method}.yaml \
--output-dir /data/datasets/DGFramework/${DATASET}/output/dg/${method}_${extra_name}/${D6} & )

(CUDA_VISIBLE_DEVICES=1 python MedSegDGSSL/tools/train.py \
--root  ${DATA} \
--trainer ${trainer} \
--source-domains ${D1} ${D2} ${D3} ${D4} ${D6} \
--target-domains ${D5} \
--seed ${SEED} \
--config-file configs/trainers/${DATASET}/${DATASET}_${method}.yaml \
--output-dir /data/datasets/DGFramework/${DATASET}/output/dg/${method}_${extra_name}/${D5} & )

(CUDA_VISIBLE_DEVICES=3 python MedSegDGSSL/tools/train.py \
--root  ${DATA} \
--trainer ${trainer} \
--source-domains ${D1} ${D2} ${D3} ${D6} ${D5} \
--target-domains ${D4} \
--seed ${SEED} \
--config-file configs/trainers/${DATASET}/${DATASET}_${method}.yaml \
--output-dir /data/datasets/DGFramework/${DATASET}/output/dg/${method}_${extra_name}/${D4} & )

(CUDA_VISIBLE_DEVICES=4 python MedSegDGSSL/tools/train.py \
--root  ${DATA} \
--trainer ${trainer} \
--source-domains ${D1} ${D2} ${D4} ${D6} ${D5} \
--target-domains ${D3} \
--seed ${SEED} \
--config-file configs/trainers/${DATASET}/${DATASET}_${method}.yaml \
--output-dir /data/datasets/DGFramework/${DATASET}/output/dg/${method}_${extra_name}/${D3} & )

(CUDA_VISIBLE_DEVICES=5 python MedSegDGSSL/tools/train.py \
--root  ${DATA} \
--trainer ${trainer} \
--source-domains ${D1} ${D4} ${D3} ${D6} ${D5} \
--target-domains ${D2} \
--seed ${SEED} \
--config-file configs/trainers/${DATASET}/${DATASET}_${method}.yaml \
--output-dir /data/datasets/DGFramework/${DATASET}/output/dg/${method}_${extra_name}/${D2} & )

(CUDA_VISIBLE_DEVICES=6 python MedSegDGSSL/tools/train.py \
--root  ${DATA} \
--trainer ${trainer} \
--source-domains ${D4} ${D2} ${D3} ${D6} ${D5} \
--target-domains ${D1} \
--seed ${SEED} \
--config-file configs/trainers/${DATASET}/${DATASET}_${method}.yaml \
--output-dir /data/datasets/DGFramework/${DATASET}/output/dg/${method}_${extra_name}/${D1})


wait
echo "Finished"