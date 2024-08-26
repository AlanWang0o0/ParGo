"""
    The hub list for models preloading.
"""

import os

__all__ = ['XHUB', 'getCachedBasename']


# Load files from HDFS if needed.
XHUB = {
    # ========== Vision Models ==========
    'vision': {
        'vit_base': 'hdfs://haruna/home/byte_lab_ocr_cn/user/feixiang.77/hub/vit_base.pth', 
        'vit_large': 'hdfs://haruna/home/byte_lab_ocr_cn/user/feixiang.77/hub/vit_large.npz', 
        'eva_vit_g': 'hdfs://haruna/home/byte_lab_ocr_cn/user/feixiang.77/hub/eva_vit_g.pth',
        'clip_vit_L': 'hdfs://haruna/home/byte_lab_ocr_cn/user/feixiang.77/hub/clip_vit_L.pth',
        # 'vit_patchdrop': '',
        # 'swin_base': '',
    },
    
    # ========== Tokenizers ==========
    'tokenizer': {
        'bert-base-uncased': 'hdfs://haruna/home/byte_lab_ocr_cn/user/shiwei.11/shiwei.11/pretrained_models/bert/bert-base-uncased',
        'bert-base-chinese': 'hdfs://haruna/home/byte_lab_ocr_cn/user/shiwei.11/shiwei.11/pretrained_models/bert/bert-base-chinese',
        'chinese-macbert-base': 'hdfs://haruna/home/byte_lab_ocr_cn/user/shiwei.11/shiwei.11/pretrained_models/bert/hfl/chinese-macbert-base',
        'chinese-roberta-wwm-ext': 'hdfs://haruna/home/byte_lab_ocr_cn/user/shiwei.11/shiwei.11/pretrained_models/bert/hfl/chinese-macbert-base',
        'albert-base-chinese-cluecorpussmall': 'hdfs://haruna/home/byte_lab_ocr_cn/user/shiwei.11/shiwei.11/pretrained_models/bert/uer/albert-base-chinese-cluecorpussmall',
        'bert-base-multilingual-cased': 'hdfs://haruna/home/byte_lab_ocr_cn/user/shiwei.11/shiwei.11/pretrained_models/bert/bert-base-multilingual-cased',
        'xlm-roberta-base': 'hdfs://haruna/home/byte_lab_ocr_cn/user/shiwei.11/shiwei.11/pretrained_models/bert/xlm-roberta-base',
    },

    # ========== Language Models ==========
    'language': {
        'google/mt5-xl': 'hdfs://haruna/home/byte_lab_ocr_cn/user/feixiang.77/hub/google/mt5-xl',
        'bigscience/bloom-3b': 'hdfs://haruna/home/byte_lab_ocr_cn/user/feixiang.77/hub/bigscience/bloom-3b',
        'bigscience/bloom-7b1': 'hdfs://haruna/home/byte_lab_ocr_cn/user/feixiang.77/hub/bigscience/bloom-7b1',
        'bigscience/bloomz-3b': 'hdfs://haruna/home/byte_lab_ocr_cn/user/feixiang.77/hub/bigscience/bloomz-3b',
        'bigscience/bloomz-7b1': 'hdfs://haruna/home/byte_lab_ocr_cn/user/feixiang.77/hub/bigscience/bloomz-7b1',
        'LinkSoul/Chinese-Llama-2-7b': 'hdfs://haruna/home/byte_lab_ocr_cn/user/feixiang.77/hub/LinkSoul/Chinese-Llama-2-7b',
        'baichuan-inc/Baichuan-7B': 'hdfs://haruna/home/byte_lab_ocr_cn/user/feixiang.77/hub/baichuan-inc/Baichuan-7B',
    }
}


def getCachedBasename(model_type, raw_name):
    return os.path.basename(XHUB[model_type][raw_name])
    # if is_dir(download_path):
    #     model_path = os.path.join(root, filename)
    # else:
    #     model_path = os.path.join(root, os.path.basename(download_path))
