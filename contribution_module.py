#%%
from p_track.p_track import PTrack, PTrackConfig
from editor.vllms_for_edit import LlavaForEdit
from utils.GLOBAL import model_path_map
from matplotlib import pyplot as plt
from dataset.vllm import EVQA
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch, os

def plot_influence(vs, ps, prone2p = 0.5, filter_p = 0.0, width = 0.3, layer_be = [0, 32], save_dir = 'save_imgs'):
    import matplotlib
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams.update({'font.size': 12}) 
    mlp_infls, att_infls = [], []
    layer_ps = []
    for i in range(len(vs['mlp'])):
        if ps['layer'][i][-1] < filter_p:
            continue
        # get Values and Probs
        M = max(max(np.abs(vs['mlp'][i])), max(np.abs(vs['att'][i]))) 
        mlp_vs = vs['mlp'][i] / M
        att_vs = vs['att'][i] / M
        mlp_ps = ps['mlp'][i]
        att_ps = ps['att'][i]
        # calculate contribution
        mlp_infl = (mlp_vs / (np.abs(mlp_vs) ** prone2p)) * (mlp_ps ** prone2p)
        att_infl = (att_vs / (np.abs(att_vs) ** prone2p)) * (att_ps ** prone2p)
        mlp_infls.append(mlp_infl)
        att_infls.append(att_infl)
        layer_ps.append(ps['layer'][i])
    mean_mlp_infl = np.mean(np.stack(mlp_infls, 0), 0)
    mean_att_infl = np.mean(np.stack(att_infls, 0), 0)
    # plot 1
    fig, ax1 = plt.subplots(figsize=(6.4, 4))
    ax1.bar(np.arange(*layer_be)-width/2, mean_att_infl, width=width, color='green', label = 'Attn')
    ax1.bar(np.arange(*layer_be)+width/2, mean_mlp_infl, width=width, color='red', label = 'MLP')
    ax1.set_xlim(-1, 32)
    ax1.set_ylabel('Module Output Contribution', labelpad=1)
    ax1.set_xlabel('Layer', labelpad=0)
    ax1.legend(loc='lower left', frameon=False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.annotate('', xy=(1.02, 0), xytext=(0, 0),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->", color='black'))
    ax1.annotate('', xy=(0, 1.02), xytext=(0, 0),
                xycoords='axes fraction', textcoords='axes fraction',
                arrowprops=dict(arrowstyle="->", color='black'))
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'p_track.svg'))

#%% load VLLM and prob track class
device = 'cuda:0'
model_name = 'llava-v1.5-7b'
vllm = LlavaForEdit(model_path_map[model_name], device)
config = PTrackConfig.from_yaml('configs/p_track/%s.yaml'%model_name)
pt = PTrack(vllm, config)
#%% prepare data
evqa = EVQA() 
requests = [d['request'] for d in evqa.data]
#%% p tracking
ks = ['layer', 'mlp', 'att']
ps, vs = {k:[] for k in ks}, {k:[] for k in ks}
for r in tqdm(requests):
    pt.forward_and_trace(r['prompt'], r['image'])
    p_dist_lists, word_lists, total_p, total_v = pt.p_tracking(save_results=False)
    for k in ks:
        ps[k].append(total_p[k])
        vs[k].append(total_v[k])
ps = {k:np.array(ps[k]) for k in ks}
vs = {k:np.array(vs[k]) for k in ks}
#%% plot
plot_influence(vs, ps)
