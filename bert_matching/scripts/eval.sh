python data/convert_oie2match.py

python main.py \
  --do_eval \
  --bert_model /share/model/bert/uncased_L-12_H-768_A-12 \
  --do_lower_case \
  --eval_batch_size 64 \
  --num_train_epochs 100 \
  --eval_each_step 1000 \
  --data_dir data/tmp \
  --output_dir model/snli_model_match \
  --device 0 \
  --processor matching \

python data/convert_match2rerank.py