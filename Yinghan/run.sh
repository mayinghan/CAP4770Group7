export CUDA_VISIBLE_DEVICES=3

python task.py --data_dir /home/mayinghan/py3/CAP4770Group7/data \
               --bert_model bert-base-cased \
               --task clf \
               --output_dir /home/mayinghan/py3/CAP4770Group7/Yinghan/output \
               --max_seq_length 128 \
               --do_train \
               --train_batch_size 32 \
               --num_train_epochs 2 \
              --overwrite_cache