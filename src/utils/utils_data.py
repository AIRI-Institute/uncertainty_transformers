import wget
import gzip
import os
import json
import numpy as np
import collections
import shutil
from datasets import Dataset, DatasetDict, load_dataset, load_metric
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups
from sklearn.preprocessing import LabelEncoder
import pytreebank
from datasets.arrow_dataset import Dataset as ArrowDataset
from torch.utils.data import Subset

import logging

log = logging.getLogger(__name__)

glue_datasets = ["cola", "mnli", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    #new datasets not from GLUE benchmark
    "20newsgroups": ("text", None),
    "amazon": ("text", None),
    "sst5": ("text", None),
    "twitter_hso": ("text", None),
    "imdb": ("text", None),
    'trec': ("text", None),
    'wmt16': ("text", None),
}

def load_data(config):
    if config.data.task_name=='20newsgroups':
        datasets = load_20newsgroups(config)
    elif config.data.task_name=='amazon':
        datasets = load_amazon_5core(config)
    elif config.data.task_name=='sst5':
        datasets = load_sst5(config)
    elif config.data.task_name=='twitter_hso':
        datasets = load_twitter_hso(config)
    elif config.data.task_name=='ag_news':
        datasets = load_ag_news(config)
    elif config.data.task_name in glue_datasets:
        datasets = load_dataset("glue", config.data.task_name, cache_dir=config.cache_dir)
    else:
        raise ValueError(
                f"Cannot load dataset with this name: {config.data.task_name}"
            )
    return datasets



def load_ag_news(config):

    dataset = load_dataset('ag_news', cache_dir=config.cache_dir)

    train_dataset = dataset['train']
    eval_dataset = dataset['test']

    datasets = DatasetDict({'train': train_dataset,
                            'validation': eval_dataset})

    return datasets


def load_ood_dataset(dataset_path, max_seq_length, tokenizer, cache_dir=None, config=None):
    log.info("Load out-of-domain dataset.")
    datasets_ood = load_dataset(
        dataset_path, ignore_verifications=True, cache_dir=cache_dir
    )
    log.info("Done with loading the dataset.")

    log.info("Preprocessing the dataset...")
    sentence1_key, sentence2_key = ("text", None)

    f_preprocess = lambda examples: preprocess_function(
        None, sentence1_key, sentence2_key, tokenizer, max_seq_length, examples
    )

    datasets_ood = datasets_ood.map(
        f_preprocess,
        batched=True,
        load_from_cache_file=True,
    )

    ood_dataset = datasets_ood[config.ue.dropout.ood_sampling.subset].select(
        list(range(config.ue.dropout.ood_sampling.number_of_samples))
    )
    log.info("Done with preprocessing the dataset.")

    return ood_dataset

def load_ood_dataset_ner(dataset_path, data_args, tokenizer, cache_dir=None, config=None):
    log.info("Load out-of-domain dataset.")
    datasets_ood = load_dataset(
        dataset_path, ignore_verifications=True, cache_dir=cache_dir
    )
    log.info("Done with loading the dataset.")

    log.info("Preprocessing the dataset...")

    text_column_name, label_column_name = "tokens", "ner_tags"
    label_to_id = {0: 0}
    f_preprocess = lambda examples: tokenize_and_align_labels(
        tokenizer,
        examples,
        text_column_name,
        label_column_name,
        data_args=data_args,
        label_to_id=label_to_id,
    )

    datasets_ood = datasets_ood.map(
        f_preprocess,
        batched=True,
        load_from_cache_file=True,
    )

    ood_dataset = datasets_ood[config.ue.dropout.ood_sampling.subset].select(
        list(range(config.ue.dropout.ood_sampling.number_of_samples))
    )
    ood_dataset = ood_dataset.remove_columns(["text", "label"])
    log.info("Done with preprocessing the dataset.")

    # for el in ood_dataset:
    #    print(len(el['ner_tags']), len(el['tokens']))
    # Have to drop labels col, otherwise it will be used by data_collator instead of ner_tags
    # ood_dataset["label"] = ood_dataset["ner_tags"]# ood_dataset.remove_columns("label")
    # ood_dataset = ood_dataset.remove_columns("label")
    return ood_dataset

def split_dataset(dataset, train_size=0.9, shuffle=True, seed=42):
    if isinstance(dataset, ArrowDataset):
        data = dataset.train_test_split(
            train_size=train_size, shuffle=shuffle, seed=seed
        )
        train_data, eval_data = data['train'], data['test']
    else:
        train_idx, eval_idx = train_test_split(
            range(len(dataset)), shuffle=shuffle, random_state=seed
        )
        train_data = Subset(dataset, train_idx)
        eval_data = Subset(dataset, eval_idx)

    return train_data, eval_data


def tokenize_and_align_labels(
    tokenizer,
    examples,
    text_column_name,
    label_column_name,
    data_args,
    label_to_id,
    padding="max_length",
):
    if text_column_name not in examples:
        examples[text_column_name] = [exp.split(" ") for exp in examples["text"]]
        examples[label_column_name] = [
            [0] * len(exp.split(" ")) for exp in examples["text"]
        ]

    tokenized_inputs = tokenizer(
        examples[text_column_name],
        padding=padding,
        max_length=data_args.max_seq_length,
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples[label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(
                    label_to_id[label[word_idx]] if data_args.label_all_tokens else -100
                )

            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def preprocess_function(
    label_to_id, sentence1_key, sentence2_key, tokenizer, max_seq_length, examples
):
    # Tokenize the texts
    args = (
        (examples[sentence1_key],)
        if sentence2_key is None
        else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(
        *args, padding="max_length", max_length=max_seq_length, truncation=True
    )

    # Map labels to IDs (not necessary for GLUE tasks)
    if label_to_id is not None and "label" in examples:
        result["label"] = [
            (label_to_id[l] if l != -1 else -1) for l in examples["label"]
        ]
    return result


def load_amazon_5core(config):
    """Return closest version of Amazon Reviews Sports & Outdoors split from the paper
    Towards More Accurate Uncertainty Estimation In Text Classification.
    """
    texts, targets = [], []
    # get zipped dataset
    url = 'http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Sports_and_Outdoors_5.json.gz'
    save_path = os.path.join(config.cache_dir, 'amazon_5core.json.gz')
    # check if file already exists, load if not
    if not(os.path.isfile(save_path)):
        save_path = wget.download(url, out=save_path)
    # unzip it and extract data to arrays
    with gzip.open(save_path, 'rb') as f:
        for line in f.readlines():
            data = json.loads(line)
            texts.append(data['reviewText'])
            targets.append(np.int64(data['overall']))
    # to shift classes from 1-5 to 0-4
    targets = np.asarray(targets) - 1
    # split on train|val|test
    text_buf, text_eval, targ_buf, targ_eval = train_test_split(texts, targets,
                                                                test_size=0.1,
                                                                random_state=config.seed)
    text_train, text_val, targ_train, targ_val = train_test_split(text_buf, targ_buf,
                                                                  test_size=2.0/9.0,
                                                                  random_state=config.seed)
    amazon_train = {'label': targ_train, 'text': text_train}
    amazon_eval = {'label': targ_eval, 'text': text_eval}
    datasets = DatasetDict({'train': Dataset.from_dict(amazon_train),
                            'validation': Dataset.from_dict(amazon_eval)})
    return datasets


def load_20newsgroups(config):
    newsgroups_train = fetch_20newsgroups(subset='train')
    newsgroups_train = {'label': newsgroups_train['target'], 'text': newsgroups_train['data']}
    newsgroups_eval = fetch_20newsgroups(subset='test')
    newsgroups_eval = {'label': newsgroups_eval['target'], 'text': newsgroups_eval['data']}
    datasets = DatasetDict({'train': Dataset.from_dict(newsgroups_train),
                            'validation': Dataset.from_dict(newsgroups_eval)})
    return datasets

def load_sst5(config):
    dataset = pytreebank.load_sst()
    sst_datasets = {}
    for category in ['train', 'test', 'dev']:
        df = {'text':[], 'label': []}
        for item in dataset[category]:
            df['text'].append(item.to_labeled_lines()[0][1])
            df['label'].append(item.to_labeled_lines()[0][0])
        cat_name = category if category!='dev' else 'validation'
        sst_datasets[cat_name] = Dataset.from_dict(df)
    dataset = DatasetDict(sst_datasets)
    return dataset


def load_mnli(config, matched: bool=True, annotator: int=-1):
    """Return MNLI dataset in different versions - matched/mismatched,
    with val labels from one annotator.
    Input:
    matched: bool, load matched or mismatched dev part
    annotator: int, annotator index. Should be in range 0-4, -1 means
    that we load mean label by all annotators for dev part
    """
    # get zipped dataset
    assert annotator >= -1 and annotator <=4, 'Annotator index should be int from -1 to 4'
    url = 'https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip'
    save_path = os.path.join(config.cache_dir, 'mnli_1_0.zip')
    if not(os.path.isfile(save_path)):
        print("File doesn't found")
        save_path = wget.download(url, out=save_path)
    print('Loaded')
    # after unpack folder
    if not(os.path.isdir(os.path.join(config.cache_dir, 'multinli_1.0'))):
        print("Extracting archive")
        shutil.unpack_archive(save_path, config.cache_dir)
    train_path = os.path.join(config.cache_dir, 'multinli_1.0/multinli_1.0_train.jsonl')
    if matched:
        dev_path = os.path.join(config.cache_dir, 'multinli_1.0/multinli_1.0_dev_matched.jsonl')
    else:
        dev_path = os.path.join(config.cache_dir, 'multinli_1.0/multinli_1.0_dev_mismatched.jsonl')

    def read_fields(data_path, annotator):
        data_texts1, data_texts2, data_targets = [], [], []
        target_key = 'annotator_labels'
        with open(data_path, 'r') as f:
            for line in f.readlines():
                data = json.loads(line)
                data_texts1.append(data['sentence1'])
                data_texts2.append(data['sentence2'])
                if annotator == -1:
                    # get the most frequent label
                    data_targets.append(collections.Counter(data[target_key]).most_common()[0][0])
                else:
                    data_targets.append(data[target_key][annotator])
        return data_texts1, data_texts2, data_targets

    # for train part set idx to 0, because there is only one label on train
    train_texts1, train_texts2, train_targets = read_fields(train_path, 0)
    dev_texts1, dev_texts2, dev_targets = read_fields(dev_path, annotator)
    # after encode targets as int classes
    target_encoder = LabelEncoder()
    train_targets = target_encoder.fit_transform(train_targets)
    dev_targets = target_encoder.transform(dev_targets)
    # and finally build dataset
    mnli_train = {'label': train_targets, 'sentence1': train_texts1, 'sentence2': train_texts2}
    mnli_eval = {'label': dev_targets, 'sentence1': dev_texts1, 'sentence2': dev_texts2}
    datasets = DatasetDict({'train': Dataset.from_dict(mnli_train),
                            'validation': Dataset.from_dict(mnli_eval)})
    return datasets



def load_twitter_hso(config):

    dataset = load_dataset('hate_speech_offensive', cache_dir=config.cache_dir)
    df = dataset['train'].to_pandas()
    annotators_count_cols = ['hate_speech_count', 'offensive_language_count', 'neither_count']

    #split by ambiguity (for test select most ambiguous part by annotators disagreement)
    df_test = df[df['count'] != df[annotators_count_cols].max(axis=1)].reset_index(drop=True)
    df_train = df[df['count'] == df[annotators_count_cols].max(axis=1)].reset_index(drop=True)

    train_dataset = {'label': df_train['class'],
                     'text': df_train['tweet']}

    eval_dataset = {'label': df_test['class'],
                    'text': df_test['tweet']}

    datasets = DatasetDict({'train': Dataset.from_dict(train_dataset),
                            'validation': Dataset.from_dict(eval_dataset)})

    return datasets



def load_data_ood(dataset, config, data_args, dataset_type="plus", split="", tokenizer=None):
    if dataset == "clinc_oos":
        # Load CLINC dataset. Types could be 'small', 'imbalanced', 'plus'. 'plus' type stands for CLINC-150, used in paper on Mahalonobis distance.
        log.info("Load dataset.")
        datasets = load_dataset(
            dataset, dataset_type, cache_dir=config.cache_dir
        )  # load_dataset("glue", config.data.task_name, cache_dir=config.cache_dir)
        log.info("Done with loading the dataset.")

        datasets = datasets.rename_column("intent", "label")
        datasets["train"] = datasets["train"].filter(lambda x: x["label"] != 42)

        def map_classes(examples):
            examples["label"] = (
                examples["label"] if (examples["label"] < 42) else examples["label"] - 1
            )
            return examples

        datasets["train"] = datasets["train"].map(
            map_classes,
            batched=False,
            load_from_cache_file=not data_args.overwrite_cache,
        )
    elif dataset in ["rostd", "snips", "rostd_coarse"]:
        # Load ROSTD/ROSTD-Coarse/SNIPS dataset
        if not (config.data.get("data_dir", False)):
            raise ValueError(
                "For ROSTD or SNIPS dataset you need to set config.data.data_dir"
            )
        if split != "":
            if split == 0:
                config.data.data_dir += f"unsup_0.75_{split}/"
            else:
                config.data.data_dir = (
                    os.path.dirname(config.data.data_dir[:-1]) + f"/unsup_0.75_{split}/"
                )

        datasets = load_dataset(
            "csv",
            data_files={
                "train": config.data.data_dir + "OODRemovedtrain.tsv",
                "validation": config.data.data_dir + "eval.tsv",
                "test": config.data.data_dir + "test.tsv",
            },
            delimiter="\t",
            column_names=["label", "smth", "text"],
            index_col=False,
        )
        # Make labels dict with last class as OOD class
        labels = datasets["validation"].unique("label")
        labels.remove("outOfDomain")
        labels2id = {label: idx for idx, label in enumerate(labels)}
        labels2id["outOfDomain"] = len(labels2id)
        # TODO: encode labels
        def map_classes(examples):
            examples["label"] = labels2id[examples["label"]]
            return examples

        datasets = datasets.map(map_classes, batched=False)
    elif dataset in ["sst2", '20newsgroups', 'amazon']:
        log.info(f"Loading {dataset} as ID dataset and {config.data.ood_data} as OOD dataset.")
        
        if config.data.task_name=='20newsgroups':
            id_datasets = load_20newsgroups(config)
        elif config.data.task_name=='amazon':
            id_datasets = load_amazon_5core(config)
        elif config.data.task_name=='sst2':
            id_datasets = load_dataset("glue", dataset, cache_dir=config.cache_dir)
        else:
            raise ValueError(
                    f"Cannot load dataset with this name: {config.data.task_name}"
                )
            
        if 'idx' in id_datasets.column_names['train']:
            id_datasets = id_datasets.remove_columns("idx")
                
        log.info("Done with loading the ID dataset.")
        
        sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
        padding = "max_length"
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
        f_preprocess = lambda examples: preprocess_function(
            None, sentence1_key, sentence2_key, tokenizer, max_seq_length, examples
        )

        id_datasets = id_datasets.map(
            f_preprocess,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        
        train_dataset = id_datasets['train']
        if config.data.task_name=='mnli':
            id_test_data = id_datasets['validation_mismatched']
        elif config.data.task_name in glue_datasets+['20newsgroups', 'amazon']:
            id_test_data = id_datasets['validation']
        else:
            id_test_data = id_datasets['test']
                    
        log.info("Done with preprocessing the ID dataset.")
        
        if config.data.ood_data=='20newsgroups':
            ood_datasets = load_20newsgroups(config)
        elif config.data.ood_data=='amazon':
            ood_datasets = load_amazon_5core(config)
        elif config.data.ood_data in glue_datasets:
            ood_datasets = load_dataset("glue", config.data.ood_data, cache_dir=config.cache_dir)
        elif config.data.ood_data in ['imdb', 'trec']:
            ood_datasets = load_dataset(
                config.data.ood_data, ignore_verifications=True, cache_dir=config.cache_dir
            )
        elif config.data.ood_data=='wmt16':
            ood_datasets = load_dataset(config.data.ood_data, 'de-en', cache_dir=config.cache_dir)
        else:
            raise ValueError(
                    f"Cannot load dataset with this name: {config.data.ood_data}"
                )
        
        if config.data.ood_data == 'wmt16':
            ood_dataset = {'text': [example['en'] for example in ood_datasets['test']['translation']],
                           'label': [0]*len(ood_datasets['test']['translation'])}
            ood_dataset = Dataset.from_dict(ood_dataset)
            ood_datasets = DatasetDict({'test': ood_dataset})

        log.info("Done with loading the OOD dataset.")
        
        sentence1_key, sentence2_key = task_to_keys[config.data.ood_data]
        padding = "max_length"
        max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
        f_preprocess = lambda examples: preprocess_function(
            None, sentence1_key, sentence2_key, tokenizer, max_seq_length, examples
        )

        ood_datasets = ood_datasets.map(
            f_preprocess,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        
        if config.data.ood_data=='mnli':
            ood_test_data = ood_datasets['validation_mismatched']
        elif config.data.ood_data in glue_datasets+['20newsgroups', 'amazon']:
            ood_test_data = ood_datasets['validation']
        else:
            ood_test_data = ood_datasets['test']
               
        log.info("Done with preprocessing the OOD dataset.")
        
        data_keys = ['input_ids', 'token_type_ids', 'attention_mask', 'label']
        new_train_dataset = {}
        new_test_dataset = {}
        for key in data_keys:
            new_train_dataset[key] = train_dataset[key]
            if key=='label':
                new_test_dataset[key] = [0]*len(id_test_data['input_ids']) + [1]*len(ood_test_data['input_ids'])
            else:
                new_test_dataset[key] = list(id_test_data[key]) + list(ood_test_data[key])
        
        train_dataset = Dataset.from_dict(new_train_dataset)
        test_dataset = Dataset.from_dict(new_test_dataset)
                    
        datasets = DatasetDict({'train': train_dataset,
                                'test': test_dataset})

    else:
        raise ValueError(
            "Task name for OOD must be clinc_oos, rostd, rostd_coarse or snips"
        )
    return datasets