import argparse
import glob
import logging
import os
import json
import time
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
from callback.adversarial import FGM
from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.progressbar import ProgressBar
from tools.common import seed_everything,json_to_text
from tools.common import init_logger, logger
#**********************
from transformers import AutoModel,WEIGHTS_NAME,BertConfig,AlbertConfig,RobertaConfig
from transformers import AutoTokenizer,AdamW
from transformers import XLMRobertaConfig,AutoTokenizer,XLMRobertaTokenizer
from models.bert_for_ner import BertSpanForNer
from models.xlmrobert_for_ner_cat import XLMRoBertSpanForNer
from models.robert_for_ner import RoBertSpanForNer
from processors.utils_ner import CNerTokenizer
from processors.ner_span import convert_examples_to_features
from processors.ner_span import ner_processors as processors
from processors.ner_span import collate_fn
from metrics.ner_metrics import SpanEntityScore,extract_spans,SpanF1
from processors.utils_ner import bert_extract_item
import numpy
import torch.nn.functional as F
#ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig,AlbertConfig)), ())
#******************
MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertSpanForNer, AutoTokenizer),
    'roberta':(RobertaConfig,RoBertSpanForNer,AutoTokenizer),
    'xlm-roberta':(XLMRobertaConfig,XLMRoBertSpanForNer,AutoTokenizer)
}

def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                  collate_fn=collate_fn)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    lr=args.learning_rate
    optimizer_grouped_parameters = [{'params': model.bert.parameters(), 'lr':lr}, 
                                    {'params': model.feature_embd.parameters(), 'lr': lr*5},
                                    {'params': model.birnn.parameters(), 'lr': lr *5},
                                    {'params': model.start_fc.parameters(), 'lr': lr},
                                    {'params': model.end_fc.parameters(), 'lr': lr},]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,weight_decay=0.01)
    #optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)
    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model_name_or_path, "scheduler.pt")):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)
    # multi-gpu training (should be after apex fp16 initialization)
    
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path) and "checkpoint" in args.model_name_or_path:
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split("-")[-1].split("/")[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)
        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
    tr_loss, logging_loss = 0.0, 0.0
    #**************adversial
    if args.do_adv:
        fgm = FGM(model, emb_name=args.adv_name, epsilon=args.adv_epsilon)
    model.zero_grad()
    seed_everything(args.seed)  # Added here for reproductibility (even between python 2 and 3)
    #************************
    f1=0
    for _ in range(int(args.num_train_epochs)):
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {"input_ids": batch[0], "attention_mask": batch[1],
                      "start_positions": batch[3],"end_positions": batch[4],"start_end_matrix":batch[8]}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (batch[2] if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            #adversial training
            if args.do_adv:
                fgm.attack()
                loss_adv = model(**inputs)[0]
                if args.n_gpu > 1:
                    loss_adv = loss_adv.mean()
                loss_adv.backward()
                fgm.restore()
            #***********
            pbar(step, {'loss': loss.item()})
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

        if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
            # Log metrics
            print(" ")
            if args.local_rank == -1:
                # Only evaluate when single GPU otherwise metrics may not average well
                results=evaluate(args, model, tokenizer)

        if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
            # Save model checkpoint
            if results['macro@F1']>f1:
                output_dir = os.path.join(args.output_dir, "checkpoint"+str(args.seed))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                tokenizer.save_vocabulary(output_dir)
                tokenizer.save_pretrained(output_dir)
                logger.info("Saving model checkpoint to %s", output_dir)
                #torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                #torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)
                f1=results['macro@F1']
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
    return global_step, tr_loss / global_step

def evaluate(args, model,model1,model2,model3,model4,tokenizer, prefix=""):
    metric = SpanEntityScore(args.id2label)
    eval_output_dir = args.output_dir
    cached_features_file1 = os.path.join(args.data_dir+'/'+args.model_type, 'cached_span-{}'.format("dev"))
    print(cached_features_file1)
    if os.path.exists(cached_features_file1) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file1)
        eval_features1 = torch.load(cached_features_file1)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    eval_features = load_and_cache_examples(args, args.task_name,tokenizer, data_type='dev')
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_features), desc="Evaluating")
    num_entity=0
    word_label=[]
    BIO_PRED=[]
    BIO_TRUE=[]
    for step, f in enumerate(zip(eval_features,eval_features1)):
        input_lens = f[0].input_len
        input_ids = torch.tensor([f[0].input_ids[:input_lens]], dtype=torch.long).to(args.device)
        input_mask = torch.tensor([f[0].input_mask[:input_lens]], dtype=torch.long).to(args.device)
        segment_ids = torch.tensor([f[0].segment_ids[:input_lens]], dtype=torch.long).to(args.device)
        start_ids = torch.tensor([f[0].start_ids[:input_lens]], dtype=torch.long).to(args.device)
        end_ids = torch.tensor([f[0].end_ids[:input_lens]], dtype=torch.long).to(args.device)
        is_start = torch.tensor([f[0].is_start], dtype=torch.long).to(args.device)
        is_end = torch.tensor([f[0].is_end], dtype=torch.long).to(args.device)
        subjects = f[0].subjects
        start_matrix=torch.tensor([f[1].start[:input_lens]], dtype=torch.float).to(args.device)
        end_matrix=torch.tensor([f[1].end[:input_lens]], dtype=torch.float).to(args.device)
        start_end_matrix = torch.tensor([numpy.concatenate((f[1].start[:input_lens],f[1].end[:input_lens,1:]),axis=1)], dtype=torch.float).to(args.device)
        model.eval()
        with torch.no_grad():
            inputs = {"input_ids": input_ids, "attention_mask": input_mask,
                      "start_positions": start_ids,"end_positions": end_ids,"start_end_matrix" :start_end_matrix}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (segment_ids if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            outputs1=model1(**inputs)
            outputs2=model2(**inputs)
            outputs3=model3(**inputs)
            outputs4=model4(**inputs)
            tmp_eval_loss, start_logits, end_logits = outputs[:3]
            tmp_eval_loss, start_logits1, end_logits1 = outputs1[:3]
            tmp_eval_loss, start_logits2, end_logits2 = outputs2[:3]
            tmp_eval_loss, start_logits3, end_logits3 = outputs3[:3]
            tmp_eval_loss, start_logits4, end_logits4 = outputs4[:3]
            '''
            end_logits = F.softmax(end_logits, -1)
            end_logits1 = F.softmax(end_logits1, -1)
            end_logits2 = F.softmax(end_logits2, -1)
            end_logits3 = F.softmax(end_logits3, -1)
            end_logits4 = F.softmax(end_logits4, -1)
            start_logits=F.softmax(start_logits,-1)
            start_logits1=F.softmax(start_logits1,-1)
            start_logits2=F.softmax(start_logits2,-1)
            start_logits3=F.softmax(start_logits3,-1)
            start_logits4=F.softmax(start_logits4,-1)


            end_logits=torch.sigmoid(end_logits)
            end_logits1=torch.sigmoid(end_logits1)
            end_logits2=torch.sigmoid(end_logits2)
            end_logits3=torch.sigmoid(end_logits3)
            end_logits4=torch.sigmoid(end_logits4)
            
            start_logits=torch.sigmoid(start_logits)
            start_logits1=torch.sigmoid(start_logits1)
            start_logits2=torch.sigmoid(start_logits2)
            start_logits3=torch.sigmoid(start_logits3)
            start_logits4=torch.sigmoid(start_logits4)
            '''
            
            start_logits=(start_logits+start_logits1+start_logits2+start_logits3+start_logits4)/5
            end_logits=(end_logits+end_logits1+end_logits2+end_logits3+end_logits4)/5
            #print(start_logits,'\n',start_logits1,'\n',start_logits2,'\n',start_logits3,'\n',start_logits4)
            R = bert_extract_item(start_logits, end_logits,is_start,is_end)
            if R==[]:
                word_label.append([])
                word_R=[]
            else:
                word_R=convert_token2word(R,is_start,is_end)
                word_label.append(word_R)
            #R = bert_extract_item(start_logits, end_logits)
            T = subjects
            word_T=convert_token2word(T,is_start,is_end)
            metric.update(true_subject=word_T, pred_subject=word_R)
            #print(word_T, word_R)
            #*****************span2label
            is_start=is_start.cpu().numpy()[0]
            len_sentence=0
            for i in is_start:
                len_sentence+=i
            bio_pred=["O" for i in range(len_sentence)]
            for R in word_R:
                if R[1]==R[2]:
                    bio_pred[R[1]]="B-"+args.id2label[R[0]]
                else:
                    bio_pred[R[1]]="B-"+args.id2label[R[0]]
                    i=R[1]+1
                    while(i>R[1] and i<=R[2]):
                        bio_pred[i]="I-"+args.id2label[R[0]]
                        i+=1
            bio_true=["O" for i in range(len_sentence)]
            for T in word_T:
                if T[1]==T[2]:
                    bio_true[T[1]]="B-"+args.id2label[T[0]]
                else:
                    bio_true[T[1]]="B-"+args.id2label[T[0]]
                    i=T[1]+1
                    while(i>T[1] and i<=T[2]):
                        bio_true[i]="I-"+args.id2label[T[0]]
                        i+=1
            #print(bio_pred,bio_true)
            BIO_PRED.append(bio_pred)
            BIO_TRUE.append(bio_true)
            num_entity+=len(T)
            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
            eval_loss += tmp_eval_loss.item()
        
        nb_eval_steps += 1
        pbar(step)
    #matroF1
    pred_result=[]
    label_result=[]
    eval_loss = eval_loss / nb_eval_steps


    for pred,label in zip(BIO_PRED,BIO_TRUE):
        label_result.append(extract_spans(label))
        pred_result.append(extract_spans(pred))    
    span_f1 = SpanF1()
    span_f1(pred_result, label_result)
    word_result=span_f1.get_metric()
    keys=list(word_result.keys())
    keys.sort()
    logger.info("***** Eval results %s *****", prefix)
    logger.info("***** Eval loss %s *****", str(eval_loss))
    for key in keys:
        #print('{} : {}'.format(key,word_result[key]))
        logger.info('{} : {}'.format(key,word_result[key]))

    save_path=args.save_dir+"predict/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path+args.lang+"-predict.json", 'w') as fr:
        for bio in BIO_PRED:
            for i in range(len(bio)):
                fr.write(bio[i]+'\n')
            fr.write('\n')
    return word_result


def convert_token2word(token_R,is_start,is_end):
    word_R=[]
    is_start=is_start.cpu().numpy()[0]
    is_end=is_end.cpu().numpy()[0]
    for unit in token_R:
        m=0
        n=0
        for j in range(unit[1]):
            m+=is_start[j]
        for j in range(unit[2]):
            n+=is_end[j]
        word_R.append((unit[0],m,n))
    return word_R

def predict(args, model, tokenizer, prefix=""):
    metric = SpanEntityScore(args.id2label)
    eval_output_dir = args.output_dir
    cached_features_file1 = os.path.join(args.data_dir+'/'+args.model_type, 'cached_span-{}'.format("dev"))
    print(cached_features_file1)
    if os.path.exists(cached_features_file1) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file1)
        eval_features1 = torch.load(cached_features_file1)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)
    eval_features = load_and_cache_examples(args, args.task_name,tokenizer, data_type='dev')
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_features))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    pbar = ProgressBar(n_total=len(eval_features), desc="Evaluating")
    num_entity=0
    word_label=[]
    BIO_PRED=[]
    BIO_TRUE=[]
    for step, f in enumerate(zip(eval_features,eval_features1)):
        input_lens = f[0].input_len
        input_ids = torch.tensor([f[0].input_ids[:input_lens]], dtype=torch.long).to(args.device)
        input_mask = torch.tensor([f[0].input_mask[:input_lens]], dtype=torch.long).to(args.device)
        segment_ids = torch.tensor([f[0].segment_ids[:input_lens]], dtype=torch.long).to(args.device)
        start_ids = torch.tensor([f[0].start_ids[:input_lens]], dtype=torch.long).to(args.device)
        end_ids = torch.tensor([f[0].end_ids[:input_lens]], dtype=torch.long).to(args.device)
        is_start = torch.tensor([f[0].is_start], dtype=torch.long).to(args.device)
        is_end = torch.tensor([f[0].is_end], dtype=torch.long).to(args.device)
        subjects = f[0].subjects
        start_matrix=torch.tensor([f[1].start[:input_lens]], dtype=torch.float).to(args.device)
        end_matrix=torch.tensor([f[1].end[:input_lens]], dtype=torch.float).to(args.device)
        start_end_matrix = torch.tensor([numpy.concatenate((f[1].start[:input_lens],f[1].end[:input_lens,1:]),axis=1)], dtype=torch.float).to(args.device)
        model.eval()
        with torch.no_grad():
            inputs = {"input_ids": input_ids, "attention_mask": input_mask,
                      "start_positions": start_ids,"end_positions": end_ids,"start_end_matrix" :start_end_matrix}
            if args.model_type != "distilbert":
                # XLM and RoBERTa don"t use segment_ids
                inputs["token_type_ids"] = (segment_ids if args.model_type in ["bert", "xlnet"] else None)
            outputs = model(**inputs)
            tmp_eval_loss, start_logits, end_logits = outputs[:3]
            R = bert_extract_item(start_logits, end_logits,is_start,is_end)
            if R==[]:
                word_label.append([])
                word_R=[]
            else:
                word_R=convert_token2word(R,is_start,is_end)
                word_label.append(word_R)
            #R = bert_extract_item(start_logits, end_logits)
            T = subjects
            word_T=convert_token2word(T,is_start,is_end)
            metric.update(true_subject=word_T, pred_subject=word_R)
            #print(word_T, word_R)
            #*****************span2label
            is_start=is_start.cpu().numpy()[0]
            len_sentence=0
            for i in is_start:
                len_sentence+=i
            bio_pred=["O" for i in range(len_sentence)]
            for R in word_R:
                if R[1]==R[2]:
                    bio_pred[R[1]]="B-"+args.id2label[R[0]]
                else:
                    bio_pred[R[1]]="B-"+args.id2label[R[0]]
                    i=R[1]+1
                    while(i>R[1] and i<=R[2]):
                        bio_pred[i]="I-"+args.id2label[R[0]]
                        i+=1
            bio_true=["O" for i in range(len_sentence)]
            for T in word_T:
                if T[1]==T[2]:
                    bio_true[T[1]]="B-"+args.id2label[T[0]]
                else:
                    bio_true[T[1]]="B-"+args.id2label[T[0]]
                    i=T[1]+1
                    while(i>T[1] and i<=T[2]):
                        bio_true[i]="I-"+args.id2label[T[0]]
                        i+=1
            #print(bio_pred,bio_true)
            BIO_PRED.append(bio_pred)
            BIO_TRUE.append(bio_true)
            num_entity+=len(T)
            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating
            eval_loss += tmp_eval_loss.item()
        
        nb_eval_steps += 1
        pbar(step)
    #matroF1
    pred_result=[]
    label_result=[]


    for pred,label in zip(BIO_PRED,BIO_TRUE):
        label_result.append(extract_spans(label))
        pred_result.append(extract_spans(pred))    
    span_f1 = SpanF1()
    span_f1(pred_result, label_result)
    word_result=span_f1.get_metric()
    keys=list(word_result.keys())
    keys.sort()
    logger.info("***** Eval results %s *****", prefix)
    for key in keys:
        #print('{} : {}'.format(key,word_result[key]))
        logger.info('{} : {}'.format(key,word_result[key]))
    #write_json
    with open(eval_output_dir+"/predict.json", 'w') as fr:
        for bio in BIO_PRED:
            for i in range(len(bio)):
                fr.write(bio[i]+'\n')
            fr.write('\n')



def load_and_cache_examples(args, task, tokenizer, data_type='train'):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    processor = processors[task]()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_span-{}_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.train_max_seq_length if data_type=='train' else args.eval_max_seq_length),
        str(task)))

    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if data_type == 'train':
            examples = processor.get_train_examples(args.data_dir)
        elif data_type == 'dev':
            examples = processor.get_dev_examples(args.data_dir)
        else:
            examples = processor.get_test_examples(args.data_dir)
        features = convert_examples_to_features(examples=examples,
                                                tokenizer=tokenizer,
                                                label_list=label_list,
                                                max_seq_length=args.train_max_seq_length if data_type=='train' \
                                                               else args.eval_max_seq_length,
                                                cls_token_at_end=bool(args.model_type in ["xlnet"]),
                                                pad_on_left=bool(args.model_type in ['xlnet']),
                                                cls_token = tokenizer.cls_token,
                                                cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
                                                sep_token=tokenizer.sep_token,
                                                # pad on the left for xlnet
                                                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                                pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                                )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)
    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    # Convert to Tensors and build dataset
    if data_type =='dev' or data_type=="test":
        return features
     #添加start和end矩阵
    cached_features_file1 = os.path.join(args.data_dir+'/'+args.model_type, 'cached_span-{}'.format("train"))
    if os.path.exists(cached_features_file1) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file1)
        features1 = torch.load(cached_features_file1)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_start_ids = torch.tensor([f.start_ids for f in features], dtype=torch.long)
    all_end_ids = torch.tensor([f.end_ids for f in features], dtype=torch.long)
    all_input_lens = torch.tensor([f.input_len for f in features], dtype=torch.long)
    all_start = torch.tensor([f.start for f in features1], dtype=torch.float)
    all_end = torch.tensor([f.end for f in features1], dtype=torch.float)
    start_end = torch.tensor([numpy.concatenate((f.start,f.end[:,1:]),axis=1) for f in features1],dtype=torch.float)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_ids,all_end_ids,all_input_lens,all_start,all_end,start_end)
    return dataset

def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--data_dir",default=None,type=str,required=True,
                        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",)
    parser.add_argument("--model_type",default=None,type=str,required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),)
    parser.add_argument("--model_name_or_path",default=None,type=str,required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--output_dir",default=None,type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.", )
    parser.add_argument('--lang', default='en', type=str)
    # Other parameters
    parser.add_argument('--markup', default='bio', type=str, choices=['bios', 'bio'])
    parser.add_argument('--loss_type', default='ce', type=str, choices=['lsr', 'focal', 'ce'])
    parser.add_argument( "--labels",default="",type=str,
                        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",)
    parser.add_argument( "--config_name", default="", type=str,
                         help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name",default="",type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name",)
    parser.add_argument("--cache_dir",default="",type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3", )
    parser.add_argument("--train_max_seq_length", default=128,type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.",)
    parser.add_argument("--eval_max_seq_length",default=512,type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.", )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    #**************************************************8
    parser.add_argument("--do_adv", action="store_true", help="Whether to run adversial training.")
    parser.add_argument('--adv_epsilon', default=1.0, type=float,
                        help="Epsilon for adversarial.")
    parser.add_argument('--adv_name', default='word_embeddings', type=str,
                        help="name for adversarial layer.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
    parser.add_argument("--evaluate_during_training",action="store_true",
                        help="Whether to run evaluation during training at each logging step.", )
    parser.add_argument("--do_lower_case", action="store_true",
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--gradient_accumulation_steps",type=int,default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument( "--max_steps", default=-1,type=int,
                         help="If > 0: set total number of training steps to perform. Override num_train_epochs.",)

    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints",action="store_true",
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",)
    parser.add_argument("--predict_checkpoints", type=int, default=0,
                        help="predict checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--fp16",action="store_true",
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",)
    parser.add_argument("--fp16_opt_level",type=str,default="O1",
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html",)
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
    parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
    parser.add_argument("--cuda", type=str, default="cuda:3", help="For distant debugging.")
    #parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="For distant debugging.")
    parser.add_argument("--save_dir",default="1",type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    #parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="For distant debugging.")
    
    args = parser.parse_args()
    args.save_dir=args.output_dir
    #*******************output_dir
    print(args.output_dir)
    args.data_dir=args.data_dir+'{}'.format(args.lang)
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir +'{}'.format(args.lang)+'{}'.format('/')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir = args.output_dir+'{}'.format(args.model_type)+'{}'.format('/')
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    time_ = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    init_logger(log_file=args.output_dir + '/{}-{}-{}-{}-{}-{}.log'.format(args.model_type, args.lang,args.per_gpu_train_batch_size,args.num_train_epochs,args.seed,time_))
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))
    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        #device=torch.device(args.cuda)
        #args.n_gpu = torch.cuda.device_count()
        #****************singel gpu
        args.n_gpu = 1
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                args.local_rank,device,args.n_gpu, bool(args.local_rank != -1),args.fp16,)
    # Set seed
    seed_everything(args.seed)
    # Prepare NER task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    args.id2label = {i: label for i, label in enumerate(label_list)}
    args.label2id = {label: i for i, label in enumerate(label_list)}
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab
    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels)
    config.soft_label=True
    config.loss_type=args.loss_type
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)

    #************************
    model = model_class(config=config,encoder_model=args.model_name_or_path)
    
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    logger.info("Training/evaluation parameters %s", args)
    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name,tokenizer, data_type='train')
        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)
        logger.info("Saving model checkpoint to %s", args.output_dir)

    # Evaluation
    results = {}
    output_dir=args.output_dir
    if args.do_eval and args.local_rank in [-1, 0]:
        args.output_dir="/home/jyma/gemnet/outputs/4_1/"+'fold1/'+args.lang+"/xlm-roberta/checkpoint20/"
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model = model_class.from_pretrained(args.output_dir,config=config,encoder_model=args.output_dir)
        model.to(args.device)
        #predict(args, model, tokenizer)
        
        args.output_dir="/home/jyma/gemnet/outputs/4_1/"+'fold2/'+args.lang+"/xlm-roberta/checkpoint20/"
        tokenizer1 = tokenizer_class.from_pretrained(args.output_dir)
        model1 = model_class.from_pretrained(args.output_dir,config=config,encoder_model=args.output_dir)
        model1.to(args.device)
        #predict(args, model1, tokenizer)
        
        args.output_dir="/home/jyma/gemnet/outputs/4_1/"+'fold3/'+args.lang+"/xlm-roberta/checkpoint20/"
        tokenizer2 = tokenizer_class.from_pretrained(args.output_dir)
        model2 = model_class.from_pretrained(args.output_dir,config=config,encoder_model=args.output_dir)
        model2.to(args.device)
        #predict(args, model2, tokenizer)
        
        args.output_dir="/home/jyma/gemnet/outputs/4_1/"+'fold4/'+args.lang+"/xlm-roberta/checkpoint20/"
        tokenizer3 = tokenizer_class.from_pretrained(args.output_dir)
        model3 = model_class.from_pretrained(args.output_dir,config=config,encoder_model=args.output_dir)
        model3.to(args.device)
        #predict(args, model3, tokenizer)
        
        args.output_dir="/home/jyma/gemnet/outputs/4_1/"+'fold5/'+args.lang+"/xlm-roberta/checkpoint20/"
        tokenizer4 = tokenizer_class.from_pretrained(args.output_dir)
        model4 = model_class.from_pretrained(args.output_dir,config=config,encoder_model=args.output_dir)
        model4.to(args.device)
        #predict(args, model4, tokenizer)
        '''
        predict(args, model, tokenizer)
        predict(args, model1, tokenizer1)
        predict(args, model2, tokenizer2)
        predict(args, model3, tokenizer3)
        '''
        #predict(args, model4, tokenizer4)

        result = evaluate(args, model,model1,model2,model3,model4,tokenizer)


    if args.do_predict and args.local_rank in [-1, 0]:
        args.output_dir=output_dir+'checkpoint'+str(args.seed)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model = model_class.from_pretrained(args.output_dir,config=config,encoder_model=args.output_dir)
        model.to(args.device)
        predict(args, model, tokenizer)

if __name__ == "__main__":
    main()