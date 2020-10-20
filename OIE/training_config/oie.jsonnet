// Configuration for RnnOIE
{
  "dataset_reader": {
    "type": "my-srl",
    "token_indexers": {
      "bert": {
        "type": "bert-pretrained",
        "pretrained_model": "pretrain/bert/bert-base-uncased",
      }
    }
  },
  "train_data_path": "data/train.head",
  "validation_data_path": "data/dev.head",
  "model": {
    "type": "my-semisupervised-open-information-extraction",
    "text_field_embedder": {
      "bert": {
        "type": "bert-pretrained",
        "pretrained_model": "pretrain/bert/bert-base-uncased",
        "top_layer_only": true,
        "requires_grad": false
      },
      "embedder_to_indexer_map": {
        "tokens": ["tokens"],
        "bert": ["bert", "bert-offsets"]
      },
      "allow_unmatched_keys": true
    },
    "encoder": {
      "type": "alternating_lstm",
      "input_size": 868,
      "hidden_size": 64,
      "num_layers": 4,
      "recurrent_dropout_probability": 0.1,
      "use_input_projection_bias": false
    },
    "binary_feature_dim": 100,
    "cuda_device": 4,
    "train_mode": "nll",
    "initializer": [
      [
        "tag_projection_layer.*weight",
        {
          "type": "orthogonal"
        }
      ]
    ],
    "regularizer": [[".*scalar_parameters.*", {"type": "l2", "alpha": 0.001}]]
  },
  "iterator": {
    "type": "bucket",
    "sorting_keys": [["tokens", "num_tokens"]],
    "batch_size" : 80
  },
  "trainer": {
    "num_epochs": 200,
    "grad_clipping": 1.0,
    "patience": 10,
    "num_serialized_models_to_keep": 10,
    "validation_metric": "+f1-measure-overall",
    "cuda_device": 4,
    "optimizer": {
      "type": "adadelta",
      "rho": 0.95
    }
  },
  "vocabulary": {
    "directory_path": "pretrain/vocabulary"
  }
}