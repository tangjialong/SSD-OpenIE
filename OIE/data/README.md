Datasets are saved in this folder.
Attention: in order to used syntantic constraint in RL you need to add headword informations to "train.head, dev.head, train.parsing, dev.parsing" by openly released evaluation repository "supervised-oie":
```
bash run_extract_head.sh
python convert_from_head2conll.py
```

"train.head, dev.head" are oracle OIE2016 datasets:
```
@inproceedings{stanovsky-dagan-2016-creating,
    title = "Creating a Large Benchmark for Open Information Extraction",
    author = "Stanovsky, Gabriel  and
      Dagan, Ido",
    booktitle = "Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2016",
    address = "Austin, Texas",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D16-1252",
    doi = "10.18653/v1/D16-1252",
    pages = "2300--2305",
}
```
"train.parsing, dev.parsing" are automatically labeled OIE2016 datasets useing our data labeling fuctions.

"sent, WEB, NYT" are test datasets:
```
@inproceedings{stanovsky-dagan-2016-creating,
    title = "Creating a Large Benchmark for Open Information Extraction",
    author = "Stanovsky, Gabriel  and
      Dagan, Ido",
    booktitle = "Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2016",
    address = "Austin, Texas",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D16-1252",
    doi = "10.18653/v1/D16-1252",
    pages = "2300--2305",
}

@inproceedings{mesquita-etal-2013-effectiveness,
    title = "Effectiveness and Efficiency of Open Relation Extraction",
    author = "Mesquita, Filipe  and
      Schmidek, Jordan  and
      Barbosa, Denilson",
    booktitle = "Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2013",
    address = "Seattle, Washington, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D13-1043",
    pages = "447--457",
}
```
