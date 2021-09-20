import argparse
import glob
import logging
import os
import random
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

from layoutlm import LayoutlmConfig, LayoutlmForSequenceClassification
from layoutlm.data.rvl_cdip import CdipProcessor, load_and_cache_examples

ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, RobertaConfig, LayoutlmConfig)
    ),
    (),
)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizerFast),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "layoutlm": (LayoutlmConfig, LayoutlmForSequenceClassification, BertTokenizerFast),
}

class_map = {
    "detail_bill": 0,
    "summary_bill": 1,
    "lab_bill": 2,
    "pharmacy_bill": 3,
    "lab_report": 4,
    "discharge_summary": 5,
    "reciepts": 6,
    "authorization_letter": 7,
    "claim_form": 8,
    "other": 9
}


model_type = 'layoutlm'
data_dir_default = 'data/'
output_dir_default = '../checkpoints-20000/'

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def evaluate(args, model, tokenizer, mode, prefix=""):
    results = {}
    eval_dataset = load_and_cache_examples(args, tokenizer, mode=mode)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
#    batch_ids = []
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        if args.model_type != "layoutlm":
            batch = batch[:4]
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[3],
            }
            if args.model_type == "layoutlm":
                inputs["bbox"] = batch[4]
            inputs["token_type_ids"] = (
                batch[2] if args.model_type in ["bert", "layoutlm"] else None
            )  # RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
            )

#        for b in batch[5]:
#            batch_ids.append(b)

#    all_mistakes = []
#    for i in range(0,len(preds)):
#        if preds[i].argmax() != out_label_ids[i]:
#            mistakes= {}
#            mistakes['image'] = args.rev_file_map[int(batch_ids[i])]
#            mistakes['true_label'] = out_label_ids[i]
#            mistakes['predicted'] = preds[i].argmax()
#            all_mistakes.append(mistakes)
#            print(args.rev_file_map[int(batch_ids[i])])
#            print("Correct ",out_label_ids[i])
#            print("Predicted ",preds[i].argmax())
    
#    import pickle
#    with open('mistakes.pkl', 'wb') as fp:
#        pickle.dump(all_mistakes, fp)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)
    result = {"acc": simple_accuracy(preds=preds, labels=out_label_ids)}
    result['precision-recall'] = classification_report(preds, out_label_ids)
    result['classwise_confusion_matrix'] = multilabel_confusion_matrix(out_label_ids, preds)
    result['confusion_matrix'] = confusion_matrix(out_label_ids, preds)
    results.update(result)

    return results


parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir",
    default=data_dir_default,
    type=str,
    help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
parser.add_argument(
    "--model_type",
    default=model_type,
    type=str,
    )
parser.add_argument(
    "--model_name_or_path",
    default='../classifier/layoutlm-base-uncased/',
    type=str,
)
parser.add_argument(
    "--output_dir",
    default=output_dir_default,
    type=str,
    help="The output directory where the model predictions and checkpoints will be written.",
    )
parser.add_argument(
    "--config_name",
    default="",
    type=str,
    help="Pretrained config name or path if not the same as model_name",
    )
parser.add_argument(
    "--tokenizer_name",
    default="",
    type=str,
    help="Pretrained tokenizer name or path if not the same as model_name",
)
parser.add_argument(
    "--cache_dir",
    default="",
    type=str,
    help="Where do you want to store the pre-trained models downloaded from s3",
)
parser.add_argument(
    "--max_seq_length",
    default=512,
    type=int,
    help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.",
)
parser.add_argument(
    "--do_eval",
    default=True,
)
parser.add_argument(
    "--do_lower_case",
    default=True,
    help="Set this flag if you are using an uncased model.",
)
parser.add_argument(
    "--per_gpu_eval_batch_size",
    default=16,
    type=int,
    help="Batch size per GPU/CPU for evaluation.",
)

parser.add_argument(
    "--eval_all_checkpoints",
    default = False,
    action="store_true",
    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
)
parser.add_argument(
    "--overwrite_cache",
    action="store_true",
    help="Overwrite the cached training and evaluation sets",
)

parser.add_argument(
    "--fp16",
    default = True,
    action="store_true",
    help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
)
parser.add_argument(
    "--local_rank",
    type=int,
    default=-1,
    help="For distributed training: local_rank",
)

args = parser.parse_args()

device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )
if torch.cuda.is_available():
   torch.cuda.set_device(device)
    
args.n_gpu = torch.cuda.device_count()

#device = "cpu"
args.device = device
print(device)

processor = CdipProcessor()
label_list = processor.get_labels()
num_labels = len(label_list)

config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

tokenizer = tokenizer_class.from_pretrained(
            args.output_dir, do_lower_case=True
        )

checkpoints = [args.output_dir]


file_map = {}
rev_file_map = {}
i = 0
with open(os.path.join(args.data_dir, "labels", "{}.txt".format("val"))) as f:
    for line in f.readlines():
        file, label = line.split()
        file_map[os.path.basename(file)] = i
        rev_file_map[i] = os.path.basename(file)
        i+=1

args.file_map = file_map
args.rev_file_map = rev_file_map

for checkpoint in checkpoints:
    global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
    prefix = (
        checkpoint.split("/")[-1]
        if checkpoint.find("checkpoint") != -1 and args.eval_all_checkpoints
        else ""
    )

    model = model_class.from_pretrained(checkpoint)
    model.to(device)
    result = evaluate(args, model, tokenizer, mode="val", prefix=prefix)

    print(result['confusion_matrix'])
    print("\n")
    print(result['precision-recall'])
    print("\n Classwise confusion matrix \n")
    print(result['classwise_confusion_matrix'])
    print(class_map_mod)
