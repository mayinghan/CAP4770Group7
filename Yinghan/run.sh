export CUDA_VISIBLE_DEVICES=2

python task.py --data_dir /home/mayinghan/py3/CAP4770Group7/data \
               --bert_model bert-base-cased \
               --task clf \
               --output_dir /home/mayinghan/py3/CAP4770Group7/Yinghan/output \
               --max_seq_length 128 \
               --do_train \
               --train_batch_size 16 \
               --num_train_epochs 1 \
            #    --overwrite_cache
