for THETA in 0.1 0.3; do 
    for TEMP in 0.03 0.15; do 
        for NA in 0 1; do 
            srun python -u lvae_train.py \
            --encode_z 10 \
            --contra_lambda 1 \
            --beta_z 6 \
            --dataset CIFAR10 \
            --epochs 100 \
            --tensorboard \
            --temperature 1 \
            --lr 0.0005 \
            --correct_split \
            --threshold 0.8 \
            --no_aug $NA \
            --mmd_loss 0 \
            --supcon_loss 1 \
            --contrastive_loss 0 \
            --rm_skips 0 \
            --theta $THETA \
            --con_temperature $TEMP \
            --exp $SLURM_ARRAY_TASK_ID > "$TMPDIR"/SupCon_abl_$SLURM_ARRAY_TASK_ID.out
        done;
    done; 
done;