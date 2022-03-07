# USTC-NELSLIP-SemEval2022Task11-GAIN
Winner system (USTC-NELSLIP) of SemEval 2022 MultiCoNER shared task over 3 out of 13 tracks (Chinese, Bangla, Code-Mixed). Rankings: https://multiconer.github.io/results.

This repository containing the training and prediction code of the system developed by the USTC-NELSLIP team for SemEval-2022 Task 11 MultiCoNER.

We provide code of two gazetteer-based methods used in our final system, **GAIN** and **weighted summation integration with gazetteer method**.

[GAIN](https://github.com/Mckysse/GAIN/tree/main/GAIN): Gazetteer-Adapted Integration Network with crf classifier mentioned in Section 3.3 in [paper](https://arxiv.org/).

[weighted_fusion_crf](https://github.com/Mckysse/GAIN/tree/main/weighted_fusion_crf): Weighted 
summation integration with gazetteer method using crf classifier mentioned in Section 3.2 in [paper](https://arxiv.org/).


## Citation
[USTC-NELSLIP at SemEval-2022 Task 11: Gazetteer-Adapted Integration Network for Multilingual Complex Named Entity Recognition](https://arxiv.org/)



## Getting Started

### Setting up the code environment

```
$ pip install -r requirements.txt
```

### Arguments
Most of our arguments are the same as those in [MULTI-CONER NER Baseline System](https://github.com/amzn/multiconer-baseline).

Notice that we add argument *gazetteer* to introduce the path of gazetteer.

```
    p.add_argument('--train', type=str, help='Path to the train data.', default=None)
    p.add_argument('--test', type=str, help='Path to the test data.', default=None)
    p.add_argument('--dev', type=str, help='Path to the dev data.', default=None)
    p.add_argument('--gazetteer', type=str, help='Path to the gazetteer data.', default=None)

    p.add_argument('--out_dir', type=str, help='Output directory.', default='.')
    p.add_argument('--iob_tagging', type=str, help='IOB tagging scheme', default='wnut')

    p.add_argument('--max_instances', type=int, help='Maximum number of instances', default=-1)
    p.add_argument('--max_length', type=int, help='Maximum number of tokens per instance.', default=128)

    p.add_argument('--encoder_model', type=str, help='Pretrained encoder model to use', default='xlm-roberta-large')
    p.add_argument('--keep_training_model', type=str, help='keep Pretrained encoder model to use', default='')
    p.add_argument('--model', type=str, help='Model path.', default=None)
    p.add_argument('--model_name', type=str, help='Model name.', default=None)
    p.add_argument('--stage', type=str, help='Training stage', default='fit')
    p.add_argument('--prefix', type=str, help='Prefix for storing evaluation files.', default='test')

    p.add_argument('--batch_size', type=int, help='Batch size.', default=128)
    p.add_argument('--gpus', type=int, help='Number of GPUs.', default=1)
    p.add_argument('--epochs', type=int, help='Number of epochs for training.', default=5)
    p.add_argument('--lr', type=float, help='Learning rate', default=1e-5)
    p.add_argument('--dropout', type=float, help='Dropout rate', default=0.1)
```

### Running

**1. Move into the folder of method you chose**

`cd AGAN` or `cd weighted_fusion_crf`

#### Before you running any shell file, you need to modify the arguments to your own paths or hyper-parameters at first.

**2. Training**

Train a xlm-roberta-large based model. The pretrained xlmr model is from [HuggingFace](https://huggingface.co/xlm-roberta-large) 

`bash run_train.sh`

**3. Fine-Tuning**

Fine-tuning from a pretrained NER model.

`bash run_finetune.sh`

**4. Predicting**

Predicting the tags from a pretrained model. 

`bash run_predict.sh`

## Reference
[MULTI-CONER NER Baseline System](https://github.com/amzn/multiconer-baseline)

## License 
The code under this repository is licensed under the [Apache 2.0 License](https://github.com/Mckysse/GAIN/blob/main/LICENSE).
