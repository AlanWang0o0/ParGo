import os
import json
import time
import datetime
from collections import defaultdict, deque

import torch
import torch.distributed as dist
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from torchvision.datasets.utils import download_url
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

from unit.utils.disttools import is_dist_avail_and_initialized


__all__ = ['SmoothedValue', 'MetricLogger', 'coco_caption_eval', 'bleu_eval', 'rouge_eval']


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def global_avg(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {:.7f}".format(name, meter.global_avg)
            )
        return self.delimiter.join(loss_str)    
    
    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def coco_caption_eval(coco_gt_root, results_file, split):
    urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json',
            'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json'}
    filenames = {'val':'coco_karpathy_val_gt.json','test':'coco_karpathy_test_gt.json'}    
    
    download_url(urls[split],coco_gt_root)
    annotation_file = os.path.join(coco_gt_root,filenames[split])
    
    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    # coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
    
    return coco_eval


def bleu_eval(res, mode='zh'):
    '''
    res is a list, every element is a dict. eg:
    [{'dt':'新年快乐', 'gt':'新年吉祥'},
    {'dt':'新年快乐', 'gt':'新年快乐'},
    {'dt':'新年快乐', 'gt':'新年快乐吉祥'}]
    '''
    def compute_bleu(reference, candidate):
        '''
        single reference: gt, list[list] eg: [['this', 'is', 'a', 'test'], ['this', 'is', 'test']]
        single candidate: dt, list, eg: ['this', 'is', 'a', 'test']
        '''
        score = sentence_bleu(reference, candidate)
        return score

    total_bleu = 0
    count = 0
    for i in range(len(res)):
        try:
            reference, candidate = res[i]['gt'], res[i]['dt']
            mode = 'zh'
            len_zh = len([char for char in candidate if u'\u4e00' <= char <= u'\u9fa5'])
            if len_zh == 0:
                mode = 'en'
            if reference == candidate:
                total_bleu += 1
            else:
                if mode == 'zh':
                    # TODO: chinese tokenizer split tokens
                    candidate = list(candidate)
                    reference = [list(ref) for ref in reference] if isinstance(reference, list) else [list(reference)]
                else:
                    candidate = candidate.split()
                    reference = [ref.split() for ref in reference] if isinstance(reference, list) else [reference.split()]
                total_bleu += compute_bleu(reference, candidate)
            count += 1
        except Exception as e:
            print("caculate bleu score error:", f"bleu:{e}")
    return total_bleu/count


def rouge_eval(res, mode='zh'):
    '''
    res is a list, every element is a dict. eg:
    [{'dt':'新年快乐', 'gt':'新年吉祥'},
    {'dt':'新年快乐', 'gt':'新年快乐'},
    {'dt':'新年快乐', 'gt':'新年快乐吉祥'}]
    '''
    rouge = Rouge()
    rouge_results = {}
    for k in ["rouge-1", "rouge-2", "rouge-l"]:
        rouge_results[k] = {"f":0, "p":0, "r":0}
    
    for i in range(len(res)):
        candidate, reference = res[i]['dt'], res[i]['gt']

        if candidate == reference:
            for k in ["rouge-1", "rouge-2", "rouge-l"]:
                for v in ["f", "p", "r"]:
                    rouge_results[k][v] += 1
        else:
            if candidate and reference:
                mode = 'zh'
                len_zh = len([char for char in candidate if u'\u4e00' <= char <= u'\u9fa5'])
                if len_zh == 0:
                    mode = 'en'
                try:
                    if mode == 'zh':
                        # TODO: chinese tokenizer split tokens
                        reference = [' '.join(reference)]
                        candidate = [' '.join(candidate)]
                    else:
                        reference = [reference]
                        candidate = [candidate]
                    tmp_rouge = rouge.get_scores(candidate, reference, avg=True)
                    for k in ["rouge-1", "rouge-2", "rouge-l"]:
                        for v in ["f", "p", "r"]:
                            rouge_results[k][v] += tmp_rouge[k][v]
                except Exception as e:
                    print((f"rouge error:{str(e)}"))
    for k in ["rouge-1", "rouge-2", "rouge-l"]:
        for v in ["f", "p", "r"]:
            rouge_results[k][v] /= max(len(res), 1)
    return rouge_results


from pycocoevalcap.bleu.bleu import Bleu
#from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge as MSRouge
from pycocoevalcap.cider.cider import Cider
#from pycocoevalcap.spice.spice import Spice

def coco_caption_eval_self(coco_gt_root, result_file, split):
    filenames = {'val':'coco_karpathy_val_gt.json','test':'coco_karpathy_test_gt.json'} 
    #download_url(urls[split],coco_gt_root)
    annotation_file = os.path.join(coco_gt_root,filenames[split])
    gt_annos = json.load(open(annotation_file, "r"))

    gts = {}
    for item in gt_annos["annotations"]:
        cap = gts.get(item['image_id'], [])
        cap.append(pre_caption(item["caption"]))
        gts[item["image_id"]] = cap
    
    test = {}
    res = json.load(open(result_file, "r"))
    for i in range(len(res)):
        if res[i]["image_id"] in gts.keys():
            test[res[i]["image_id"]] = [res[i]["dt"]]
        #gts[res[i]["image_id"]] = [ref.split() for ref in res[i]["gt"]] if isinstance(res[i]["gt"], list) else [res[i]["gt"].split()]
    assert(gts.keys() == test.keys())
    scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (MSRouge(), "ROUGE_L"),
            (Cider(), "CIDEr")
            #(Spice(), "SPICE")
        ]
    test_results = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(gts, test)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                test_results[m] = sc
                #print("%s: %0.3f"%(m, sc))
        else:
            test_results[method] = score
    
    return test_results

import numpy as np

def scores_to_ranks(scores: torch.Tensor):
    """Convert model output scores into ranks."""
    batch_size, num_rounds, num_options = scores.size()
    scores = scores.view(-1, num_options)

    # sort in descending order - largest score gets highest rank
    sorted_ranks, ranked_idx = scores.sort(1, descending=True)

    # i-th position in ranked_idx specifies which score shall take this
    # position but we want i-th position to have rank of score at that
    # position, do this conversion
    ranks = ranked_idx.clone().fill_(0)
    for i in range(ranked_idx.size(0)):
        for j in range(num_options):
            ranks[i][ranked_idx[i][j]] = j
    # convert from 0-99 ranks to 1-100 ranks
    ranks += 1
    ranks = ranks.view(batch_size, num_rounds, num_options)
    return ranks

class SparseGTMetrics(object):
    """
    A class to accumulate all metrics with sparse ground truth annotations.
    These include Recall (@ 1, 5, 10), Mean Rank and Mean Reciprocal Rank.
    """

    def __init__(self):
        self._rank_list = []
        self._rank_list_rnd = []
        self.num_rounds = None

    def observe(
        self, predicted_scores: torch.Tensor, target_ranks: torch.Tensor
    ):
        predicted_scores = predicted_scores.detach()

        # shape: (batch_size, num_rounds, num_options)
        predicted_ranks = scores_to_ranks(predicted_scores)
        batch_size, num_rounds, num_options = predicted_ranks.size()
        self.num_rounds = num_rounds
        # collapse batch dimension
        predicted_ranks = predicted_ranks.view(
            batch_size * num_rounds, num_options
        )

        # shape: (batch_size * num_rounds, )
        target_ranks = target_ranks.view(batch_size * num_rounds).long()

        # shape: (batch_size * num_rounds, )
        predicted_gt_ranks = predicted_ranks[
            torch.arange(batch_size * num_rounds), target_ranks
        ]
        self._rank_list.extend(list(predicted_gt_ranks.cpu().numpy()))
        
        predicted_gt_ranks_rnd = predicted_gt_ranks.view(batch_size, num_rounds)
        #  predicted gt ranks
        self._rank_list_rnd.append(predicted_gt_ranks_rnd.cpu().numpy())

    def retrieve(self, reset: bool = True):
        num_examples = len(self._rank_list)
        if num_examples > 0:
            # convert to numpy array for easy calculation.
            __rank_list = torch.tensor(self._rank_list).float()
            metrics = {
                "r@1": torch.mean((__rank_list <= 1).float()).item(),
                "r@5": torch.mean((__rank_list <= 5).float()).item(),
                "r@10": torch.mean((__rank_list <= 10).float()).item(),
                "mean": torch.mean(__rank_list).item(),
                "mrr": torch.mean(__rank_list.reciprocal()).item()
            }
            # add round metrics
            _rank_list_rnd = np.concatenate(self._rank_list_rnd)
            _rank_list_rnd = _rank_list_rnd.astype(float)
            r_1_rnd = np.mean(_rank_list_rnd <= 1, axis=0)
            r_5_rnd = np.mean(_rank_list_rnd <= 5, axis=0)
            r_10_rnd = np.mean(_rank_list_rnd <= 10, axis=0)
            mean_rnd = np.mean(_rank_list_rnd, axis=0)
            mrr_rnd = np.mean(np.reciprocal(_rank_list_rnd), axis=0)

            for rnd in range(1, self.num_rounds + 1):
                metrics["r_1" + "_round_" + str(rnd)] = r_1_rnd[rnd-1]
                metrics["r_5" + "_round_" + str(rnd)] = r_5_rnd[rnd-1]
                metrics["r_10" + "_round_" + str(rnd)] = r_10_rnd[rnd-1]
                metrics["mean" + "_round_" + str(rnd)] = mean_rnd[rnd-1]
                metrics["mrr" + "_round_" + str(rnd)] = mrr_rnd[rnd-1]
        else:
            metrics = {}

        if reset:
            self.reset()
        return metrics

    def reset(self):
        self._rank_list = []
        self._rank_list_rnd = []

class NDCG(object):
    def __init__(self):
        self._ndcg_numerator = 0.0
        self._ndcg_denominator = 0.0

    def observe(
        self, predicted_scores: torch.Tensor, target_relevance: torch.Tensor
    ):
        """
        Observe model output scores and target ground truth relevance and
        accumulate NDCG metric.
        Parameters
        ----------
        predicted_scores: torch.Tensor
            A tensor of shape (batch_size, num_options), because dense
            annotations are available for 1 randomly picked round out of 10.
        target_relevance: torch.Tensor
            A tensor of shape same as predicted scores, indicating ground truth
            relevance of each answer option for a particular round.
        """
        predicted_scores = predicted_scores.detach()

        # shape: (batch_size, 1, num_options)
        predicted_scores = predicted_scores.unsqueeze(1)
        predicted_ranks = scores_to_ranks(predicted_scores)

        # shape: (batch_size, num_options)
        predicted_ranks = predicted_ranks.squeeze()
        batch_size, num_options = predicted_ranks.size()
        
        k = torch.sum(target_relevance != 0, dim=-1)

        # shape: (batch_size, num_options)
        _, rankings = torch.sort(predicted_ranks, dim=-1)
        # Sort relevance in descending order so highest relevance gets top rnk.
        _, best_rankings = torch.sort(
            target_relevance, dim=-1, descending=True
        )

        # shape: (batch_size, )
        batch_ndcg = []
        for batch_index in range(batch_size):
            num_relevant = k[batch_index]
            dcg = self._dcg(
                rankings[batch_index][:num_relevant],
                target_relevance[batch_index],
            )
            best_dcg = self._dcg(
                best_rankings[batch_index][:num_relevant],
                target_relevance[batch_index],
            )
            batch_ndcg.append(dcg / best_dcg)

        self._ndcg_denominator += batch_size
        self._ndcg_numerator += sum(batch_ndcg)

    def _dcg(self, rankings: torch.Tensor, relevance: torch.Tensor):
        sorted_relevance = relevance[rankings].cpu().float()
        discounts = torch.log2(torch.arange(len(rankings)).float() + 2)
        return torch.sum(sorted_relevance / discounts, dim=-1)

    def retrieve(self, reset: bool = True):
        if self._ndcg_denominator > 0:
            metrics = {
                "ndcg": float(self._ndcg_numerator / self._ndcg_denominator)
            }
        else:
            metrics = {}

        if reset:
            self.reset()
        return metrics

    def reset(self):
        self._ndcg_numerator = 0.0
        self._ndcg_denominator = 0.0