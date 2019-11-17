export CUDA_VISIBLE_DEVICES=3

python task.py --data_dir /home/mayinghan/py3/CAP4770Group7/data \
               --bert_model bert-base-cased \
               --task clf \
               --output_dir /home/mayinghan/py3/CAP4770Group7/Yinghan/output \
               --max_seq_length 512 \
               --do_train \
               --do_eval \
               --train_batch_size 8 \
               --eval_batch_size 8 \
               --num_train_epochs 3 \
              --dev_step 500 \
              --warmup_ratio 0.1 \
              --overwrite_output_dir