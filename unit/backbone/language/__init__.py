# import os
# import torch

# # tokenizers
# from unit.backbone.language.tokenizer import *
# # language models
# from unit.backbone.language.bert import BertConfig, BertModel, BertLMHeadModel
# from unit.backbone.language.xlm_roberta import XLMRobertaConfig, XLMRobertaModel, XLMRobertaLMHeadModel
# from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM

# from unit.utils import download_cached_file, XHUB


# __all__ = ['init_tokenizer', 'init_language_model', 'init_llm']


# SUPPORTED_TOKENIZERS = {
#     'bert-base-uncased': 'BertTokenizer',
#     'chinese-macbert-base': 'BertTokenizer',
#     'chinese-roberta-wwm-ext': 'BertTokenizer',
#     'bert-base-multilingual-cased': 'BertTokenizer',
#     'xlm-roberta-base': 'autoTokenizerBlip',
#     'google/flan-t5-xl': 'T5TokenizerFast',
#     'google/mt5-xl': 'autoTokenizerLLM',
#     'bigscience/bloom-3b': 'autoTokenizerLLM',
#     'bigscience/bloom-7b1': 'autoTokenizerLLM',
#     'bigscience/bloomz-3b': 'autoTokenizerLLM',
#     'bigscience/bloomz-7b1': 'autoTokenizerLLM',
#     'LinkSoul/Chinese-Llama-2-7b': 'autoTokenizerLLM',
#     'baichuan-inc/Baichuan-7B': 'autoTokenizerLLM',
#     'Baichuan2-7B-Base': 'autoTokenizerLLM',
# }


# def init_tokenizer(name, do_lower_case=True, use_jieba=True, root="./", **kwargs):
#     assert os.path.basename(name) in SUPPORTED_TOKENIZERS.keys()
#     if SUPPORTED_TOKENIZERS[name] == 'BertTokenizer':
#         use_jieba = kwargs.get('use_jieba', False)
#         return bertTokenizer(bert_type=name, do_lower_case=do_lower_case, use_jieba=use_jieba, root=root, **kwargs)
#     elif SUPPORTED_TOKENIZERS[name] =='autoTokenizerBlip':
#         return autoTokenizerBlip(bert_type=name, do_lower_case=do_lower_case, **kwargs)
#     elif SUPPORTED_TOKENIZERS[name] == 'T5TokenizerFast':
#         return t5TokenizerFast(t5_model=name, **kwargs)
#     # elif SUPPORTED_TOKENIZERS[name] == 'mt5Tokenizer':
#     #     return mt5Tokenizer(mt5_model=name)
#     elif SUPPORTED_TOKENIZERS[os.path.basename(name)] == 'autoTokenizerLLM':
#         return autoTokenizerLLM(model=name, **kwargs)
#     else:
#         raise NotImplementedError


# def init_config(name, med_config):
#     if 'xlm-roberta' in name:
#         bert_config = XLMRobertaConfig.from_json_file(med_config)
#     else:
#         bert_config = BertConfig.from_json_file(med_config)
#     return bert_config


# def init_bert_encoder(bert_name, bert_config, pretrained_bert, **kwargs):
#     if pretrained_bert:
#         if 'nlvr' in bert_name:
#             if 'xlm' in bert_name:
#                 language_model = NLVRXLMRobertaModel.from_pretrained(bert_name.split('nlvr-')[-1], config=bert_config, **kwargs)
#             else:
#                 language_model = NLVRBertModel.from_pretrained(bert_name.split('nlvr-')[-1], config=bert_config, **kwargs)
#         elif 'xlm-roberta' in bert_name:
#             language_model = XLMRobertaModel.from_pretrained(bert_name, config=bert_config, **kwargs)
#         else:
#             language_model = BertModel.from_pretrained(bert_name, config=bert_config, **kwargs)
#     else:
#         if 'nlvr' in bert_name:
#             if 'xlm' in bert_name:
#                 language_model = NLVRXLMRobertaModel(config=bert_config, **kwargs)
#             else:
#                 language_model = NLVRBertModel(config=bert_config, **kwargs)
#         elif 'xlm-roberta' in bert_name:
#             language_model = XLMRobertaModel(bert_config, **kwargs)
#         else:
#             language_model = BertModel(bert_config, **kwargs)
#     return language_model


# def init_bert_decoder(bert_name, bert_config, pretrained_bert, **kwargs):
#     if pretrained_bert:
#         if 'xlm-roberta' in bert_name:
#             language_model = XLMRobertaLMHeadModel.from_pretrained(bert_name, config=bert_config, **kwargs)
#         else:
#             language_model = BertLMHeadModel.from_pretrained(bert_name, config=bert_config, **kwargs)
#     else:
#         if 'xlm-roberta' in bert_name:
#             language_model = XLMRobertaLMHeadModel(bert_config, **kwargs)
#         else:
#             language_model = BertLMHeadModel(bert_config)
#     return language_model


# def init_language_model(model:str, name:str, config=None, pretrained=True, root="./", **kwargs):
#     """
#     Create language model, either from existing or custom configs.
#     """
#     # using preloaded weights
#     if pretrained:
#         # root = kwargs.get('root', '')
#         cached_file = os.path.join(root, name)  # using a directory as the path
#         print(f'[loading] using pretrained {cached_file} for {name}')
#         # cached_file = download_cached_file(SUPPORTED_VISION_MODELS[name], progress=False)
    
#     # Bert series
#     if model in ['bert_encoder', 'bert_decoder']:
#         name = cached_file if pretrained else name
#         if config is None:
#             med_config = os.path.join(name, 'config.json')
#         else:
#             med_config = config
#             # simple checking, in case someone inputs the path under pretrained files
#             if os.path.basename(config).startswith('med_config_'):
#                 pretrained = False
#         print(f'build {model} with pretrained={pretrained}')
        
#         bert_config = init_config(name, med_config)
#         if 'encoder_width' in kwargs:
#             bert_config.encoder_width = kwargs.pop('encoder_width')
        
#         if model == 'bert_encoder':
#             language_model = init_bert_encoder(name, bert_config, pretrained, **kwargs)
#         elif model == 'bert_decoder':
#             language_model = init_bert_decoder(name, bert_config, pretrained, **kwargs)
#         # NOTE: resize token embeddings should be added manually.
    
#     # flan-t5 series
#     elif model == 't5':
#         name = cached_file if pretrained else name
#         t5_config = T5Config.from_pretrained(name)
#         if 'dense_act_fn' in kwargs:
#             t5_config.dense_act_fn = kwargs.pop('dense_act_fn')
#         language_model = T5ForConditionalGeneration.from_pretrained(name, config=t5_config, **kwargs)
    
#     # mt5 series
#     elif model == 'mt5':
#         name = cached_file if pretrained else name
#         language_model = AutoModelForSeq2SeqLM.from_pretrained(name)
#     else:
#         raise NotImplementedError
    
#     return language_model


# # def init_llm(model:str, name:str, config=None, pretrained=True, **kwargs):
# def init_llm(model, name, dtype='fp16', **kwargs):
#     root = kwargs.get('root', '')
#     cached_file = os.path.join(root, name)  # using a directory as the path
#     print(f'[loading] using pretrained {cached_file} for {name}')
#     name = cached_file or name
    
#     # if model in ['bloom', 'bloomz']:
#     dtype = torch.float16 if dtype == 'fp16' else torch.float32
#     llm = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, trust_remote_code=True)
#     return llm
