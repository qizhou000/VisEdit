#%%
from p_track.p_track import PTrack, PTrackConfig, get_module
from utils import load_vllm_for_edit
from utils.nethook import TraceDict
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from copy import deepcopy
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch, os

def get_surrounding_pixels(i, j, n, max_height, max_width, return_i = True):
    '''In a [max_height, max_width] matrix, get surrounding pixels centered 
    around [i, j] with radius `n`. '''
    row_start = max(0, i - n)
    row_end = min(max_height, i + n + 1)
    col_start = max(0, j - n)
    col_end = min(max_width, j + n + 1)
    surrounding_pixels = []
    for row in range(row_start, row_end):
        for col in range(col_start, col_end):
            if return_i:
                surrounding_pixels.append(row * max_width + col)
            else:
                surrounding_pixels.append((row, col))
    return surrounding_pixels
def attn_layer_rep_attribution(att_layer, att_inpt, attr_toks:list[int], 
        sim_tok = -1, noise_level = 0.3, test_n = 10, seed = 123):
    '''
    `att_layer`: an attention module.
    `att_inpt`: the input of the attention module (args, kargs).
    '''
    mod_paras = ['hidden_states', 'attention_mask', 'position_ids']
    rng = np.random.default_rng(seed)
    args, kargs = att_inpt
    if len(args) != 0: raise
    if kargs['hidden_states'].shape[0] != 1: raise
    kargs['past_key_value'] = None
    kargs = {k: v.clone() if hasattr(v, 'clone') else deepcopy(v) for k, v in kargs.items()}
    kargs = {k: torch.repeat_interleave(v, 1+test_n, 0) if k in mod_paras else v 
             for k, v in kargs.items()}
    # computation
    hs = kargs['hidden_states']
    r = rng.normal(0, noise_level, [test_n, len(attr_toks), hs.shape[-1]]) 
    hs[1:, attr_toks] += torch.from_numpy(r).to(hs)
    kargs['hidden_states'] = hs
    outpt = att_layer(**kargs)[0] 
    influence = 1 - torch.cosine_similarity(outpt[1:, sim_tok], outpt[:1, sim_tok], -1)
    return influence
def img_reps_attribution(att_layer, att_inpt, vt_range, pixel_r = 3, 
                         noise_level = 0.7, test_n = 10, seed = 123):
    inf_list = []
    with torch.no_grad():
        for i in tqdm(range(24)):
            for j in range(24):
                attr_toks = [k + vt_range[0] for k in get_surrounding_pixels(i, j, pixel_r, 24, 24)]
                influence = attn_layer_rep_attribution(att_layer, att_inpt, attr_toks, 
                                        -1, noise_level, test_n, seed).mean()
                inf_list.append(influence)
    return inf_list
def trace_get_attention_reps(vllm, hook_atts, prompt, img):
    llm_inpt, vt_range = vllm.get_llm_input_embeds([prompt], [img])
    with torch.no_grad(), TraceDict(vllm.model, hook_atts, clone = True, 
                        retain_input = True, with_kwargs = True) as td:
        o = vllm.get_llm_outpt(llm_inpt, vt_range)
    return td, vt_range
def heatmap_to_colormap(heatmap:torch.Tensor, scale_factor = [16, 16], mode = 'bilinear'):
    import matplotlib.cm as cm
    h, w = heatmap.shape
    heatmap = heatmap.reshape(1, 1, h, w)
    heatmap_resized = F.interpolate(heatmap, scale_factor=scale_factor, mode=mode)
    heatmap_min = heatmap.min()
    heatmap_max = heatmap.max()
    heatmap_normalized = (heatmap_resized - heatmap_min) / (heatmap_max - heatmap_min)
    heatmap_color = cm.jet(heatmap_normalized[0, 0])  
    heatmap_color = heatmap_color[:, :, :3]  
    return heatmap_color
def img_infs_show(img, infs, save_path, alpha = 0.55, scale_factor = 16):
    heatmap = torch.tensor(infs[:576]).reshape(24,24)
    heatmap = heatmap_to_colormap(heatmap, scale_factor = [scale_factor, scale_factor])
    heatmap_img = np.array(img.resize([24*scale_factor,24*scale_factor]))/255 * alpha + heatmap *(1-alpha)
    plt.imshow(heatmap_img)
    plt.show()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.imsave(save_path, heatmap_img)
def test_and_plot(layer = 19, pixel_r = 1, noise_level = 0.7, seed = None):
    hook_att_name = hook_atts[layer] 
    td, vt_range = trace_get_attention_reps(vllm, hook_atts, prompt, img)
    att_layer = get_module(vllm.model, hook_att_name)
    att_inpt = td[hook_att_name].input
    img_infs = img_reps_attribution(att_layer, att_inpt, vt_range, pixel_r = pixel_r, 
                            noise_level = noise_level, test_n = 10, seed = seed)
    plt.plot([float(i) for i in img_infs[:-1]])
    plt.show()
    save_dir = os.path.join('save_imgs/vllm_atr_llava', os.path.basename(rs[i]['img']).split('.')[0]) 
    t = datetime.now().strftime('%Y.%m.%d-%H.%M.%S')
    img_name = 'l-%s-r-%s-n-%s-t-%s.png'%(layer, pixel_r, noise_level, t)
    save_path = os.path.join(save_dir, img_name)
    img_infs_show(img, img_infs, save_path, 0.5)


#%% load VLLM and prob track class
device = 'cuda:0'
model_name = 'llava-v1.5-7b'
vllm = load_vllm_for_edit(model_name, device)
config = PTrackConfig.from_yaml('configs/p_track/%s.yaml'%model_name)
hook_atts = [config.attn_module_tmp.format(i) for i in range(32)]
#%% select image
rs = [
{
    'img': 'data/easy-edit-mm/images/val2014/COCO_val2014_000000000285.jpg',
    "prompt": "The animal in the picture is called a", # key token: bear
}, 
{
    'img': 'data/easy-edit-mm/images/val2014/COCO_val2014_000000080517.jpg',
    "prompt": 'What is the fruit in the picture? The answer is', # key token: ban(ana)
}, 
{
    'img': 'data/easy-edit-mm/images/val2014/COCO_val2014_000000080517.jpg',
    "prompt": 'The color of the hat the man is wearing is', # key token: p(ink)
}, 
{
    'img': 'data/easy-edit-mm/images/val2014/COCO_val2014_000000526675.jpg',
    "prompt": 'What color are the flowers in the vase? It is', # key token: yellow
}, 
]
i = 0 # select test 
prompt, img = rs[i]['prompt'], Image.open(rs[i]['img'])
img.show()
inpt = vllm.get_llm_input_embeds([prompt], [img])
pred_tok_id = torch.argmax(torch.softmax(vllm.get_llm_outpt(*inpt).logits, -1)[0, -1])
pred_tok = vllm.get_llm_tokenizer().decode(pred_tok_id)
print('%s \n %s'%(prompt, pred_tok))
#%% visualization
test_and_plot(layer = 21, pixel_r = 3) 

