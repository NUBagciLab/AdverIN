#!/bin/bash
DATA=/data/bagcilab/datasets/zheyuan/DGFramework/Fundus/processed/2DRegionPosNeg
DATASET=Fundus
D1=Domain1
D2=Domain2
D3=Domain3
D4=Domain4

SEED=0

trainers_list=(AdverHist \
               AdverHist \
               AdverHist )

methods_list=(adverhistregion_numcontrol_6 \
              adverhistregion_numcontrol_11 \
              adverhistregion_numcontrol_31 \
              adverhistregion_numcontrol_41 \
              adverhistregion_numcontrol_51 )

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
    
    (CUDA_VISIBLE_DEVICES=2 python MedSegDGSSL/tools/train.py \
    --root  ${DATA} \
    --trainer ${trainer} \
    --source-domains ${D1} ${D2} ${D3}  \
    --target-domains ${D4} \
    --seed ${SEED} \
    --config-file configs/trainers/${DATASET}/${DATASET}_${method}.yaml \
    --output-dir /home/zze3980/Projects/DGFramework/output/${DATASET}/dg/${method}/${D4} & )

   (CUDA_VISIBLE_DEVICES=2 python MedSegDGSSL/tools/train.py \
    --root  ${DATA} \
    --trainer ${trainer} \
    --source-domains ${D1} ${D2} ${D4}  \
    --target-domains ${D3} \
    --seed ${SEED} \
    --config-file configs/trainers/${DATASET}/${DATASET}_${method}.yaml \
    --output-dir /home/zze3980/Projects/DGFramework/output/${DATASET}/dg/${method}/${D3} & )

    (CUDA_VISIBLE_DEVICES=3 python MedSegDGSSL/tools/train.py \
    --root  ${DATA} \
    --trainer ${trainer} \
    --source-domains ${D1} ${D3} ${D4}  \
    --target-domains ${D2} \
    --seed ${SEED} \
    --config-file configs/trainers/${DATASET}/${DATASET}_${method}.yaml \
    --output-dir /home/zze3980/Projects/DGFramework/output/${DATASET}/dg/${method}/${D2} & )

    let "int++"
    if [[ $(($int % 3)) -eq 0 ]]
    then
        (CUDA_VISIBLE_DEVICES=3 python MedSegDGSSL/tools/train.py \
        --root  ${DATA} \
        --trainer ${trainer} \
        --source-domains ${D3} ${D2} ${D4}  \
        --target-domains ${D1} \
        --seed ${SEED} \
        --config-file configs/trainers/${DATASET}/${DATASET}_${method}.yaml \
        --output-dir /home/zze3980/Projects/DGFramework/output/${DATASET}/dg/${method}/${D1} )
    else
         (CUDA_VISIBLE_DEVICES=3 python MedSegDGSSL/tools/train.py \
        --root  ${DATA} \
        --trainer ${trainer} \
        --source-domains ${D3} ${D2} ${D4}  \
        --target-domains ${D1} \
        --seed ${SEED} \
        --config-file configs/trainers/${DATASET}/${DATASET}_${method}.yaml \
        --output-dir /home/zze3980/Projects/DGFramework/output/${DATASET}/dg/${method}/${D1} & )
    fi
    wait
done

echo 'Finished Here'