#!/bin/bash
DATA=/data/datasets/DGFramework/Cardiac/processed/2DSlice3
DATASET=Cardiac
D1=Domain1
D2=Domain2
D3=Domain3
D4=Domain4

SEED=0
trainers_list=(Vanilla Vanilla RandConvDG MixUpDG StyleAugDG StyleAugDG StyleAugDG StyleAugDG StyleAugDG AdverTraining AdverTraining RSCDG AlignFeaturesDG AlignFeaturesDG)
methods_list=(bnorm inorm randconv mixup mixstyle csu dsu padain binorm adverbias advernoise rsc alignmmd aligncrossentropy)
# trainers_list=(StyleAugDG)
# methods_list=(csu)
cuda_device=0

int=0

while (( $int<${#methods_list[@]} ))
do
    echo $int
    trainer=${trainers_list[$int]}
    method=${methods_list[$int]}
    echo ${trainer}
    echo ${method}
    echo '*************************************************************'
    (CUDA_VISIBLE_DEVICES=0 python MedSegDGSSL/tools/train.py \
    --root  ${DATA} \
    --trainer ${trainer} \
    --source-domains ${D1} ${D2} ${D3}  \
    --target-domains ${D4} \
    --seed ${SEED} \
    --config-file configs/trainers/${DATASET}/${DATASET}_${method}.yaml \
    --output-dir /data/datasets/DGFramework/output/${DATASET}/dg/${method}/${D4} & )
    

    (CUDA_VISIBLE_DEVICES=1 python MedSegDGSSL/tools/train.py \
    --root  ${DATA} \
    --trainer ${trainer} \
    --source-domains ${D1} ${D2} ${D4}  \
    --target-domains ${D3} \
    --seed ${SEED} \
    --config-file configs/trainers/${DATASET}/${DATASET}_${method}.yaml \
    --output-dir /data/datasets/DGFramework/output/${DATASET}/dg/${method}/${D3} & )

    (CUDA_VISIBLE_DEVICES=3 python MedSegDGSSL/tools/train.py \
    --root  ${DATA} \
    --trainer ${trainer} \
    --source-domains ${D1} ${D3} ${D4}  \
    --target-domains ${D2} \
    --seed ${SEED} \
    --config-file configs/trainers/${DATASET}/${DATASET}_${method}.yaml \
    --output-dir /data/datasets/DGFramework/output/${DATASET}/dg/${method}/${D2} & )

    (CUDA_VISIBLE_DEVICES=5 python MedSegDGSSL/tools/train.py \
    --root  ${DATA} \
    --trainer ${trainer} \
    --source-domains ${D2} ${D3} ${D4}  \
    --target-domains ${D1} \
    --seed ${SEED} \
    --config-file configs/trainers/${DATASET}/${DATASET}_${method}.yaml \
    --output-dir /data/datasets/DGFramework/output/${DATASET}/dg/${method}/${D1} )


    let "int++"
    wait
done

echo 'Finished Here'