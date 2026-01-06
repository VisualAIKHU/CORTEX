#! /bin/bash

for max_iter in 60000; do
    for lr in 8e-05; do
        for weight_decay in 0.0; do
            for bsize in 16; do
                for itda_lambda in 0.0001; do
                    exp_dir="experiments"
                    run_name="CORTEX_DIRL"
                    snapshot=$max_iter
                    gpu=0
                    args=(
                        --cfg configs/dynamic/transformer.yaml
                        --gpu                   $gpu
                        --seed                  1111
                        --exp_dir               $exp_dir
                        --lr                    $lr
                        --batch_size            $bsize
                        --weight_decay          $weight_decay
                        --run_name              $run_name
                        --max_iter              $max_iter
                        --itda_lambda           $itda_lambda
                    )

                    # Train
                    python train.py "${args[@]}"

                    for snapshot in $(seq 10000 10000 60000); do

                        # Test
                        python test.py --cfg configs/dynamic/transformer.yaml --snapshot $snapshot --gpu $gpu --exp_dir $exp_dir --run_name $run_name

                        results_dir="$exp_dir/$run_name/test_output_${snapshot}/captions"
                        anno_path="data/total_change_captions_reformat.json"
                        type_file_path="data/type_mapping.json"

                        # Evaluate
                        python evaluate.py --results_dir $results_dir --anno $anno_path --type_file $type_file_path
                    done
                done
            done
        done
    done
done