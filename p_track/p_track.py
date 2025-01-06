import matplotlib.pyplot as plt
from typing import List
from dataclasses import dataclass, asdict
from editor.base import BaseConfig
import torch, os
from utils.nethook import TraceDict
from datetime import datetime
from PIL.Image import Image as PILImage
import numpy as np
from editor.vllm_editors.base import BaseVLLMForEdit

@dataclass
class PTrackConfig(BaseConfig):
    model_name: str
    num_layers: int
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    norm_path: str
    voc_path: str

class PTrack():
    def __init__(self, vllm:BaseVLLMForEdit, config:PTrackConfig) -> None:
        self.cfg = config
        self.vllm = vllm
        self.tokenizer = vllm.get_llm_tokenizer()
        self.norm = get_module(self.vllm.model, config.norm_path)
        self.voc = get_module(self.vllm.model, config.voc_path)

    def forward_and_trace(self, prompt:str, img:PILImage):
        layer_n = self.cfg.num_layers
        layers = [self.cfg.layer_module_tmp.format(i) for i in range(layer_n)]
        atts = [self.cfg.attn_module_tmp.format(i) for i in range(layer_n)]
        mlps = [self.cfg.mlp_module_tmp.format(i) for i in range(layer_n)]
        trace_layers = layers + atts + mlps
        with torch.no_grad(), TraceDict(self.vllm.model, trace_layers, 
                                        retain_output=True) as td:
            input_embeds, vt_range = self.vllm.get_llm_input_embeds([prompt], [img])
            self.outpt = self.vllm.get_llm_outpt(input_embeds, vt_range).logits
        self.td = td
        self.prompt = prompt
        self.img = img  
    
    def p_tracking(self, track_tok_i = -1, track_module = ['layer', 'att', 'mlp'], 
            top_k = 5, save_results = True, predict_word = None, verbose = False):
        if not hasattr(self, 'td'): raise
        input_ids = self.tokenizer(self.prompt).input_ids
        prompt = self.prompt
        trace = self.tokenizer.decode(input_ids[track_tok_i])
        if predict_word == None:
            predict_id = torch.softmax(self.outpt[0, track_tok_i], 0).argmax()
            predict_word = self.tokenizer.decode(predict_id)
        else:
            predict_id = self.tokenizer(predict_word, add_special_tokens=False).input_ids[0]
            predict_word = self.tokenizer.decode(predict_id)
        if verbose:
            print('Input:"%s"'%prompt)
            print('Trace:"%s"'%trace)
            print('Predict:"%s"'%predict_word)
        tmps = {'layer': self.cfg.layer_module_tmp, 'att': self.cfg.attn_module_tmp,
                   'mlp': self.cfg.mlp_module_tmp}
        p_dist_lists, word_lists, total_p, total_v = {}, {}, {}, {}
        for tm in track_module:
            tmp = tmps[tm]
            p_dist_lists[tm], word_lists[tm], total_p[tm], total_v[tm] = [], [], [], []
            for i in range(self.cfg.num_layers):
                h = self.td[tmp.format(i)].output
                # h = h[0][0, track_tok_i] if type(h) == tuple else h[0, track_tok_i]
                h = h[0] if isinstance(h, (list, tuple)) else h
                h = h[track_tok_i] if h.dim() == 2 else h[0, track_tok_i]
                ps, ids, words = self.rep_to_voc_p(h, top_k = top_k)
                total_p[tm].append(float(torch.softmax(self.reps_to_word_predict(h), 0)[predict_id]))
                total_v[tm].append(float(self.reps_to_word_predict(h)[predict_id]))
                p_dist_lists[tm].append(ps.tolist())
                word_lists[tm].append(['%s (%s)'%(int(i), w) for i, w in zip(ids, words)])
        if save_results:
            self.save_results(track_module, prompt, trace, predict_word, p_dist_lists, 
                     word_lists, total_p, total_v)
        return p_dist_lists, word_lists, total_p, total_v

    def save_results(self, track_module, prompt, trace, predict_word, p_dist_lists, 
                     word_lists, total_p, total_v):
        img_postfix = 'png'
        t = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
        dir_path = os.path.join('records', 'p_track', self.cfg.model_name, prompt[:10] + '-' + t) 
        os.makedirs(dir_path, exist_ok=True)
        log = r'%s'%prompt
        log += '\n%s->%s'%(trace, predict_word)
        with open(os.path.join(dir_path, 'logs'), 'w') as f:
            f.write(log)
        if self.img != None:
            plt.imsave(os.path.join(dir_path, 'img.png'), np.array(self.img))
        for tm in track_module:
            img_path = os.path.join(dir_path, tm, '%s->%s.%s'%(trace, predict_word, img_postfix))
            title = prompt + '\n%s->%s'%(trace, predict_word)
            plot_multiple_bar_charts_with_ticks(p_dist_lists[tm], word_lists[tm], title, img_path)
        c =  {'layer': 'purple', 'att': 'red', 'mlp': 'green'}  
        # p
        p_list = [total_p[tm] for tm in total_p.keys()]
        ticks_list = [list(range(len(total_p[tm]))) for tm in total_p.keys()]
        colors = [c[tm] for tm in total_p.keys()]
        titles = ["'%s' %s rep predict '%s'"%(trace, tm, predict_word) for tm in total_p.keys()]
        img_path = os.path.join(dir_path, "p '%s'->'%s'.%s"%(trace, predict_word, img_postfix))
        plot_multiple_histograms(p_list, ticks_list, colors, titles,
                ['layers']*len(p_list), ['p']*len(p_list), img_path)
        # v
        values_list = [total_v[tm] for tm in total_v.keys()]
        ticks_list = [list(range(len(total_v[tm]))) for tm in total_v.keys()]
        colors = [c[tm] for tm in total_v.keys()]
        titles = ["'%s' %s rep predict '%s'"%(trace, tm, predict_word) for tm in total_v.keys()]
        img_path = os.path.join(dir_path, "v '%s'->'%s'.%s"%(trace, predict_word, img_postfix))
        plot_multiple_histograms(values_list, ticks_list, colors, titles,
                ['layers']*len(values_list), ['v']*len(values_list), img_path)
        # p * v
        pv_list = [[p * v for p, v in zip(total_p[tm], total_v[tm])] 
                       for tm in total_v.keys()]
        img_path = os.path.join(dir_path, "pxv '%s'->'%s'.%s"%(trace, predict_word, img_postfix))
        plot_multiple_histograms(pv_list, ticks_list, colors, titles,
                ['layers']*len(pv_list), ['v']*len(pv_list), img_path)
        

    def reps_to_word_predict(self, reps:torch.Tensor):
        return self.voc(self.norm(reps))

    def rep_to_voc_p(self, rep, top_k = 20):
        v = self.reps_to_word_predict(rep)
        ps, ids = torch.sort(torch.softmax(v, -1), descending = True)
        ps, ids = ps[:top_k], ids[:top_k]
        words = [self.tokenizer.decode(i) for i in ids]
        return ps, ids, words
    

def plot_multiple_bar_charts_with_ticks(data_list:List, x_ticks_list, title:str, 
                                        img_path:str):
    num_charts = len(data_list)
    fig, axes = plt.subplots(num_charts, 1, figsize=(10, 2 * num_charts), sharex=False)
    fig.suptitle(title, fontsize=16)  
    if num_charts == 1:
        axes = [axes]
    for i, (data, x_ticks) in enumerate(zip(data_list, x_ticks_list)):
        axes[i].bar(range(len(data)), data, color='blue', width=1.0)
        axes[i].set_ylabel('Layer %s'%i)
        axes[i].set_xticks(range(len(x_ticks)))
        axes[i].set_xticklabels(x_ticks, ha='center')
        axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        axes[i].set_ylim(0, 1)  
    axes[-1].set_xlabel('Index')
    plt.tight_layout()
    for ax in axes:
        for tick in ax.get_xticklabels():
            tick.set_visible(True)
    if os.path.dirname(img_path) != '':
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
    plt.savefig(img_path, bbox_inches="tight")
    plt.close()


def plot_multiple_histograms(values_list, ticks_list, colors, titles, xlabels, ylabels, img_path):
    num_plots = len(values_list)
    plt.figure(figsize=(10, 6 * num_plots))
    
    for i, (values, ticks, color, title, xlabel, ylabel) in enumerate(
            zip(values_list, ticks_list, colors, titles, xlabels, ylabels)):
        plt.subplot(num_plots, 1, i + 1)
        plt.bar(ticks, values, color=color)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xticks(ticks=range(len(ticks)), labels=ticks)
    
    if os.path.dirname(img_path) != '':
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
    plt.savefig(img_path, bbox_inches="tight")
    plt.close()


def get_module(model, name):
    """
    Finds the named module within the given model.
    """
    for n, m in model.named_modules():
        if n == name:
            return m
    raise LookupError(name)
