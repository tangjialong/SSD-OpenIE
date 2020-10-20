# Train Model 
## Train supervised bert oie baseline and further RL
### Prepare data.head
```
python convert_from_head2conll.py
```
### Pretrain model.bert
```
allennlp train training_config/oie.jsonnet --include-package oie --serialization-dir saved/XXX
```
### Train model.rl
```
allennlp train training_config/oie.jsonnet.rl --include-package oie --serialization-dir saved/XXX
```

## Train model in an unsupervised way
### Prepare data
```
python convert_sen2parsing.py
```
### Pretrain model.bert
```
allennlp train training_config/oie.jsonnet --include-package oie --serialization-dir saved/XXX
```
### Train model.rl
```
allennlp train training_config/oie.jsonnet.rl --include-package oie --serialization-dir saved/XXX
```

# Extract and Evaluation 
```
python openie_extract.py --inp data/XXX --keep_one --model saved/XXX/model.tar.gz  --out saved/XXX
python benchmark.py --gold oie_corpus/XXX --predArgHeadMatch --tabbed XXX --out XXX/PR_Curve 
python pr_plot.py --in XXX --out XXX
```