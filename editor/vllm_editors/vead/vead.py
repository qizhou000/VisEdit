from ...vllms_for_edit.base import BaseVLLMForEdit
from ..base import VLLMBaseEditorWithTraining
from utils import find_module, move_to_device
from torch.utils.hooks import RemovableHandle
from torch.nn.utils.rnn import pad_sequence
from utils.nethook import TraceDict, Trace
from dataclasses import dataclass, asdict
from .adpt_model import VisionEditAdaptor
from dataset.vllm import BaseVLLMEditData
from PIL.Image import Image as PILImage
from typing import Dict, List, Tuple
from ...base import BaseConfig
from torch.optim import Adam
import torch, os, yaml
from tqdm import tqdm
from torch import nn
import numpy as np

@dataclass
class VEADConfig(BaseConfig):
    @dataclass
    class TrainConfig():
        lr:float
        rel_lambda:float
        gen_lambda:float
        loc_lambda:float
        inf_mapper_lambda:float
    @dataclass
    class InfluenceTrace():
        add_it: bool
        layers: List[int] # Influence Trace layers
        test_n: int
        noise_level: float
        window: int
        vt_sample_n: int
        mid_dim: int
    edit_model_name: str
    llm_layer_tmp: str
    llm_att_tmp: str
    edit_layers: List[int]
    llm_hidden_size: int
    adaptor_mid_dim: int
    adaptor_cross_att_head_n: int
    train_cfg: TrainConfig
    IT: InfluenceTrace

    @classmethod
    def from_yaml(self, fpath):
        with open(fpath, "r") as f:
            data = yaml.safe_load(f)
        data['train_cfg'] = self.TrainConfig(**data['train_cfg'])
        data['IT'] = self.InfluenceTrace(**data['IT'])
        return self(**data)
    @classmethod
    def from_json(self, fpath):
        raise

class VEAD(VLLMBaseEditorWithTraining):
    def __init__(self, vllm: BaseVLLMForEdit, config:VEADConfig, device='cuda:0', 
                 vllm_data_proc: BaseVLLMForEdit = None, data_proc_device = None, 
                 train_data_cache_root = 'data'):
        super().__init__(vllm, config, device)
        self.cfg = config
        if vllm_data_proc != None:
            self.vllm_data_proc = vllm_data_proc
            self.vllm_data_proc.model.eval()
            self.vllm_data_proc.model.requires_grad_(False)
            self.data_proc_device = data_proc_device
            self.vllm_data_proc.set_device(data_proc_device)
        self.adaptors, self.adaptors_hooks = self.init_hook_adaptors(config, vllm, device)
        self.train_data_cache_dir = os.path.join(train_data_cache_root, 'vead_train_cache')
        self.init_wrap_get_llm_outpt() 
        self.open_adaptors(True) 
        self.set_train(False)
        
    ############################################################################
    ############################# Initialize ###################################
    ############################################################################
    def init_hook_adaptors(self, config:VEADConfig, vllm:BaseVLLMForEdit, device
            )->Tuple[Dict[str, VisionEditAdaptor], Dict[str, RemovableHandle]]:
        def adapter_hook_wrap(adpt):
            def adapter_hook(m, args, outpt):
                if isinstance(outpt, tuple):
                    outpt = list(outpt)
                    outpt[0] = adpt(outpt[0])
                else:
                    outpt = adpt(outpt)
                return outpt
            return adapter_hook
        adaptors, adaptors_hooks = {}, {}
        for i in config.edit_layers:
            edit_layer_name = self.cfg.llm_layer_tmp.format(i)
            edit_layer = find_module(vllm.model, edit_layer_name)
            if hasattr(edit_layer, 'adaptor_hooked'): 
                edit_layer._forward_hooks.clear()
                print('Clear all hooks of %s and re-hook adaptor.'%edit_layer_name)
            edit_layer.adaptor_hooked = True
            adaptor = VisionEditAdaptor(config.llm_hidden_size, config.adaptor_mid_dim, 
                config.adaptor_cross_att_head_n, self.vllm.get_img_token_n(), 
                config.IT.add_it, self.cfg.IT.mid_dim).to(device)
            hook = edit_layer.register_forward_hook(adapter_hook_wrap(adaptor))
            adaptors[edit_layer_name] = adaptor
            adaptors_hooks[edit_layer_name] = hook
        return adaptors, adaptors_hooks

    def init_wrap_get_llm_outpt(self):
        def get_llm_outpt_wrap(get_llm_outpt_func):
            def wrapped_get_llm_outpt(llm_inpt, vt_range):
                if vt_range == None:
                    has_img = False
                    vt_begin = vt_end = None
                else:
                    has_img = True
                    vt_begin, vt_end = vt_range
                for adaptor in self.adaptors.values():
                    adaptor.set_input_info(has_img, vt_begin, vt_end)
                return get_llm_outpt_func(llm_inpt, vt_range)
            wrapped_get_llm_outpt.vead_wrapped = True
            return wrapped_get_llm_outpt
        if not hasattr(self.vllm.get_llm_outpt, 'vead_wrapped'):
            self.vllm.get_llm_outpt = get_llm_outpt_wrap(self.vllm.get_llm_outpt)

    ############################################################################
    ########################### Basic Editor Functions #########################
    ############################################################################
    def name_of_editor_and_model(self)->Tuple[str, str]:
        return 'vead', self.cfg.edit_model_name

    def restore_to_original_model(self):
        for adaptor in self.adaptors.values():
            adaptor.open_adaptor(False)
            adaptor.set_edit_signal(None, None, None)

    def edit_one_piece(self, request:Dict):
        ''' Assume: `request` = {'image': PILImage, 'prompt': str, 'target_new': str, ...} '''
        self.edit_batch([request])

    def edit_batch(self, requests:List[Dict]):
        '''Assume `requests` = [
          {'image': PILImage, 'prompt': str, 'target_new': str, ...},
          {'image': PILImage, 'prompt': str, 'target_new': str, ...}, ...] '''
        self.open_adaptors(False)
        edit_reps_list = {k:[] for k in self.adaptors.keys()}
        edit_reps_att_mask_list = {k:[] for k in self.adaptors.keys()}
        prompt_end_list = {k:[] for k in self.adaptors.keys()}
        for r in requests:
            edit_reps, prompt_end = self.get_edit_signal_for_one_request(r['prompt'], r['image'], r['target_new'])
            for k in self.adaptors.keys():
                edit_reps_list[k].append(edit_reps[k][0]) # [l, d]
                edit_reps_att_mask = torch.ones([len(edit_reps[k][0])], device=self.device) # [l]
                edit_reps_att_mask_list[k].append(edit_reps_att_mask)
                prompt_end_list[k].append(prompt_end[k]) # [1]
        for k in self.adaptors.keys():
            edit_reps_list[k] = pad_sequence(edit_reps_list[k], True) # [b, l, d]
            edit_reps_att_mask_list[k] = pad_sequence(edit_reps_att_mask_list[k], True) # [b, l]
            prompt_end_list[k] = torch.tensor(prompt_end_list[k], device=self.device) # [b]
        self.set_edit_signal_for_adaptors(edit_reps_list, edit_reps_att_mask_list, prompt_end_list)
        self.open_adaptors(True)

    def if_can_batch_edit(self)->bool:
        return True
    
    ############################################################################
    ######################### Basic VEAD Functions #############################
    ############################################################################
    def open_adaptors(self, if_open:bool):
        for adaptor in self.adaptors.values():
            adaptor.open_adaptor(if_open)
    
    def get_edit_signal_for_one_request(self, prompt:str, image:PILImage, target_new:str):
        self.open_adaptors(False)
        (input_embeds, vt_range), label_ids, label_masks = self.vllm.prompts_imgs_target_to_xym([prompt], [image], [target_new])
        with TraceDict(self.vllm.model, [i for i in self.adaptors.keys()], 
                            retain_output=True, stop = True) as td:
            self.vllm.get_llm_outpt(input_embeds, vt_range)
        edit_reps = {k:td[k].output[0] if not isinstance(td[k].output, torch.Tensor) else td[k].output
                     for k in self.adaptors.keys()} # edit_reps[k]: [1,l,d]
        assert all([isinstance(r, torch.Tensor) for r in edit_reps.values()])
        # prompt_end is length of input_embeds minus length of label_ids
        prompt_end = {k:input_embeds['inputs_embeds'].shape[1] - label_ids.shape[1] 
                      for k in self.adaptors.keys()} # prompt_end[k]: int
        return edit_reps, prompt_end

    def set_edit_signal_for_adaptors(self, edit_reps:Dict[str, torch.Tensor], 
            edit_reps_att_mask:Dict[str, torch.Tensor], prompt_end:Dict[str, torch.Tensor]):
        for k in self.adaptors.keys():
            self.adaptors[k].set_edit_signal(edit_reps[k], edit_reps_att_mask[k], prompt_end[k])

    def infer_from_mid_layer(self, infer_vllm:BaseVLLMForEdit, llm_inpt, vt_range, 
                             mid_inpt_layer_i:int, mid_inpt_reps:torch.Tensor):
        def skip_layer(*args, **kargs):
            return [None]*2
        def mid_inpt_embeds(inpt, layer):
            args, kargs = td[self.cfg.llm_layer_tmp.format(0)].input
            args = (mid_inpt_reps,)
            return args, kargs
        skip_layers = [self.cfg.llm_layer_tmp.format(i) for i in range(mid_inpt_layer_i)]
        inpt_layer = [self.cfg.llm_layer_tmp.format(mid_inpt_layer_i)]
        with TraceDict(infer_vllm.model, [self.cfg.llm_layer_tmp.format(0)], retain_input=True
            ) as td, TraceDict(infer_vllm.model, skip_layers, layer_func_replace=skip_layer
            ), TraceDict(infer_vllm.model, inpt_layer, edit_input=mid_inpt_embeds): 
            outpt = infer_vllm.get_llm_outpt(llm_inpt, vt_range)
        return outpt
    
    ############################################################################
    ####################### Training Functions ###########################
    ############################################################################
    def other_train_init_final(self):
        pass

    def get_modules_for_training(self)->Dict[str, nn.Module]:
        return self.adaptors
    
    def reinit_train_parameters(self):
        for v in self.adaptors.values():
            v.reset_parameters()

    def preprocess_train_data(self, raw_data: BaseVLLMEditData, start_i = 0, end_i = None) -> List:
        ''' 
        start_i/end_i: start and end index of raw data to be processed. 
        This function save all middle variables used for training. 
        Directory structure:
            -vead_train_cache_dir
                -edit_model_name
                    -dataset_name
                        -edit_signal
                            -0
                                -language_model.model.layers.16 (edit reps)
                                -language_model.model.layers.17 (edit reps)
                                -...
                            ...
                            -data_n
                        -xym
                            -0
                                -language_model.model.layers.15 (rel/gen/loc inpt reps)
                                -...
                            ...
                            -data_n
        `edit_signal` is the VEAD editing representation of each request, 
        `xym` is the input embeddings and predict label_ids/label_masks of the 
            layer before the first VEAD, including rel/gen/loc for each sample.
        return the paths of `edit_signal` and `xym` for samples.
        '''
        # set random number generator
        if not hasattr(self, 'data_proc_device'):
            raise BaseException("Not set data processing model.")
        self.np_rng = np.random.default_rng(self.random_seed)
        self.pt_rng = torch.Generator(device=self.data_proc_device)
        self.pt_rng.manual_seed(self.random_seed)
        # processing
        def get_llm_layer_inpt_embeds(input_embeds, vt_range):
            with torch.no_grad(), Trace(self.vllm.model, self.mid_inpt_start_layer, 
                    retain_input=True, with_kwargs=False, stop=True) as t:
                self.vllm.get_llm_outpt(input_embeds, vt_range)
            return t.input[0] # args[0] is inputs_embeds
        training_data_paths = []
        data_dir = os.path.join(self.train_data_cache_dir, 
            self.name_of_editor_and_model()[1], raw_data.dataset_name())
        edit_signal_dir = os.path.join(data_dir, 'edit_signal')
        xym_dir = os.path.join(data_dir, 'xym')
        self.mid_inpt_start_layer_i = min(self.cfg.edit_layers + self.cfg.IT.layers)
        self.mid_inpt_start_layer = self.cfg.llm_layer_tmp.format(self.mid_inpt_start_layer_i)
        end_i = len(raw_data.data) if end_i == None else min(len(raw_data.data), end_i)
        self.open_adaptors(False)
        for i in tqdm(range(start_i, end_i), 'Pre-processing train data'):
            d = raw_data.data[i]
            # save edit_signal
            edit_signal_dir_i = os.path.join(edit_signal_dir, str(i))
            flg = False
            for k in self.adaptors.keys():
                save_path = os.path.join(edit_signal_dir_i, k)
                if not os.path.exists(save_path):
                    flg = True
                    break
            if flg:
                r = d['request']
                edit_reps, prompt_end = self.get_edit_signal_for_one_request(r['prompt'], r['image'], r['target_new'])
                os.makedirs(edit_signal_dir_i, exist_ok=True)
                for k in edit_reps.keys():
                    save_path = os.path.join(edit_signal_dir_i, k)
                    if not os.path.exists(save_path):
                        save_data = {'edit_reps':edit_reps[k], 'prompt_end':prompt_end[k]}
                        torch.save(save_data, save_path)
            # save evaluation embeddings
            xym_dir_i = os.path.join(xym_dir, str(i))
            os.makedirs(xym_dir_i, exist_ok = True)
            save_path = os.path.join(xym_dir_i, self.mid_inpt_start_layer)
            if not os.path.exists(save_path):
                # Reliability
                prompt = [d['request']['prompt']]
                img = [d['request']['image']]
                target = [d['request']['target_new']]
                (input_embeds, vt_range), label_ids, label_masks = self.vllm.prompts_imgs_target_to_xym(prompt, img, target)
                input_embeds = get_llm_layer_inpt_embeds(input_embeds, vt_range)
                rel_data = (input_embeds, vt_range), label_ids, label_masks
                # Generality
                gen_data = {}
                for gen_name in d['generality'].keys():
                    prompt = [d['generality'][gen_name][0]['prompt']]
                    img = [d['generality'][gen_name][0]['image']]
                    img = None if img[0] == None else img
                    target = [d['generality'][gen_name][0]['target']]
                    (input_embeds, vt_range), label_ids, label_masks = self.vllm.prompts_imgs_target_to_xym(prompt, img, target)
                    input_embeds = get_llm_layer_inpt_embeds(input_embeds, vt_range)
                    gen_data[gen_name] = (input_embeds, vt_range), label_ids, label_masks
                # Locality
                loc_data = {}
                for loc_name in d['locality'].keys():
                    img = [d['locality'][loc_name][0]['image']]
                    if img[0] == None:
                        continue # text-only queries does not need train
                    prompt = [d['locality'][loc_name][0]['prompt']]
                    target = [d['locality'][loc_name][0]['target']]
                    (input_embeds, vt_range), label_ids, label_masks = self.vllm.prompts_imgs_target_to_xym(prompt, img, target)
                    input_embeds = get_llm_layer_inpt_embeds(input_embeds, vt_range)
                    loc_data[loc_name] = (input_embeds, vt_range), label_ids, label_masks
                torch.save((rel_data, gen_data, loc_data), save_path)
            training_data_paths.append((edit_signal_dir_i, xym_dir_i))
        return training_data_paths

    def __get_xy_for_influence_mapper__(self, rel_xym, gen_xym, loc_xym
                                    )->Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        # x and y for computing influence mapper loss
        trace_vllm = self.vllm_data_proc
        # random choose rel/gen xym
        sps = [rel_xym, *[v for v in gen_xym.values()]] 
        rg_xym = sps[self.np_rng.integers(0, len(sps))]
        sps = [v for v in loc_xym.values()]
        loc_xym = sps[self.np_rng.integers(0, len(sps))]
        # trace: 
        trace_atts = [self.cfg.llm_att_tmp.format(i) for i in self.cfg.IT.layers]
        edit_layers = [self.cfg.llm_layer_tmp.format(i) for i in self.cfg.edit_layers]
        (mid_inpt_r, vt_range_r), _, label_masks = rel_xym 
        prompt_ends = [mid_inpt_r['inputs_embeds'].shape[1] - label_masks.shape[1] + i for i in 
                       (label_masks.cumsum(dim=1) == 1).int().argmax(dim=1)] # [b]
        (mid_inpt_rg, vt_range_rg), _, _ = rg_xym 
        (mid_inpt_l, vt_range_l), _, _ = loc_xym 
        with torch.no_grad():
            # 1. inputs of traced attention layers for rel. 2. outputs of edit layers for rel
            with (TraceDict(trace_vllm.model, trace_atts, retain_input=True) as att_trace_r, 
                  TraceDict(trace_vllm.model, edit_layers, retain_output=True) as edit_trace_r):
                self.infer_from_mid_layer(trace_vllm, mid_inpt_r, vt_range_r, 
                        self.mid_inpt_start_layer_i, mid_inpt_r['inputs_embeds'])
            # 1. inputs of traced attention layers for rel/gen. 2. outputs of edit layers for rel&gen
            with (TraceDict(trace_vllm.model, trace_atts, retain_input=True) as att_trace_rg,
                  TraceDict(trace_vllm.model, edit_layers, retain_output=True) as edit_trace_rg):
                self.infer_from_mid_layer(trace_vllm, mid_inpt_rg, vt_range_rg, 
                        self.mid_inpt_start_layer_i, mid_inpt_rg['inputs_embeds'])
            # 1. outputs of edit layers for loc
            with TraceDict(trace_vllm.model, edit_layers, retain_output=True) as edit_trace_l: 
                self.infer_from_mid_layer(trace_vllm, mid_inpt_l, vt_range_l, 
                        self.mid_inpt_start_layer_i, mid_inpt_l['inputs_embeds'])
        # random choose pollute image patches representations and compute their
        # influences to the end of prompt token in attention layers
        test_n = self.cfg.IT.test_n
        noise_level = self.cfg.IT.noise_level
        vtsn = self.cfg.IT.vt_sample_n
        img_token_n = self.vllm.get_img_token_n()
        l = np.array(range(img_token_n))
        self.np_rng.shuffle(l)
        l = l[:vtsn]
        if self.vllm.is_q_former_based(): # if Q-former-based, random sample image tokens to be polluted
            attr_toks = [np.array(get_surrounding_pixels(0, i, self.cfg.IT.window, 
                                1, img_token_n)) + vt_range_r[0] for i in l]
        else: # if not Q-former-based, random sample image patches tokens (with window) to be polluted
            wh = int(img_token_n**0.5)
            assert wh * wh == img_token_n
            trace_is, trace_js = l//wh, l%wh
            attr_toks = [np.array(get_surrounding_pixels(ti, tj, self.cfg.IT.window, 
                wh, wh)) + vt_range_r[0] for ti, tj in zip(trace_is, trace_js)] # vtsn * [<=(2*window+1)**2]
        rg_influences = [] 
        with torch.no_grad():
            batch_size, tok_len, hidden_dim = mid_inpt_r['inputs_embeds'].shape
            for k in att_trace_r.keys():
                args_r, kargs_r = att_trace_r[k].input
                args_rg, kargs_rg = att_trace_rg[k].input
                assert len(args_r) == 0 and len(args_rg) == 0
                kargs_r['hidden_states'][:, vt_range_r[0], vt_range_r[1]] = kargs_rg['hidden_states'][:, vt_range_rg[0], vt_range_rg[1]]
                kargs = {k: torch.repeat_interleave(v.unsqueeze(0), 1 + vtsn * test_n, 0) # [1+48*10, 4, 576+, 4096]
                    if k in ['hidden_states', 'attention_mask'] else v for k, v in kargs_r.items()}
                kargs['attention_mask'] = kargs['attention_mask'].flatten(0, 1) # [(1+48*10)*4, 576+, 4096]
                hs = kargs['hidden_states'] # [1+48*10, 4, 576+, 4096]
                # pollute
                for i in range(vtsn): # 48
                    ats = attr_toks[i] 
                    pollute_size = [test_n, batch_size, len(ats), hidden_dim]
                    bs, be = 1 + i * test_n, 1 + (i + 1) * test_n
                    hs[bs:be, :, ats] += torch.normal(0, noise_level, pollute_size, 
                        generator=self.pt_rng, device = self.data_proc_device) 
                kargs['hidden_states'] = hs.reshape((1+vtsn*test_n)*batch_size, 
                        tok_len, hidden_dim) # [(1+48*10)*4, 576+, 4096]
                att_layer = find_module(trace_vllm.model, k)
                outpt = att_layer(**kargs)[0].reshape(1+vtsn*test_n, batch_size, 
                        tok_len, hidden_dim) # [1+48*10, 4, 576+, 4096]
                clean_out = [outpt[0, i, pe] for i, pe in enumerate(prompt_ends)] # b * [4096]
                dirty_out = outpt[1:].reshape(vtsn, test_n, batch_size, tok_len, hidden_dim) # [48, 10, 4, 576+, 4096]
                dirty_out = [dirty_out[:, :, i, prompt_ends[i]] for i in range(batch_size)] # b * [48, 10, 4096]
                batch_infl = [1 - torch.cosine_similarity(do, co.reshape(1, 1, hidden_dim), -1) # ~ [0, 2]
                             for do, co in zip(dirty_out, clean_out)] # b * [48, 10]
                batch_infl = torch.stack([i.mean(1) for i in batch_infl], 0) # [b, 48]
                rg_influences.append(batch_infl)
        rg_influences = torch.stack(rg_influences, 0).mean(0) # [b, 48]
        rg_influences = rg_influences / rg_influences.sum(1, keepdim = True) # [b, 48]
        # get influence inputs
        rg_img_tok_i = np.array(l) + vt_range_r[0] # [48]
        assert all(rg_img_tok_i == np.array(l) + vt_range_r[0])
        l_img_tok_i = np.array(l) + vt_range_l[0]
        influence_mapper_inpts = {} 
        for k in edit_trace_r.keys():
            # rel/gen img_reps input to influence mapper
            rg_otpt = edit_trace_rg[k].output 
            rg_otpt = rg_otpt[0] if not isinstance(rg_otpt, torch.Tensor) else rg_otpt # [b, l, d]
            assert isinstance(rg_otpt, torch.Tensor)
            rg_img_reps = rg_otpt[:, rg_img_tok_i] # [b, 48, d]
            # loc img_reps input to influence mapper
            l_otpt = edit_trace_l[k].output 
            l_otpt = l_otpt[0] if not isinstance(l_otpt, torch.Tensor) else l_otpt # [b, l, d]
            assert isinstance(l_otpt, torch.Tensor)
            l_img_reps = l_otpt[:, l_img_tok_i] # [b, 48, d]
            # rel prompt end token input to influence mapper
            r_otpt = edit_trace_r[k].output 
            r_otpt = r_otpt[0] if not isinstance(r_otpt, torch.Tensor) else r_otpt # [b, l, d]
            assert isinstance(r_otpt, torch.Tensor)
            prompt_end_reps = r_otpt[range(len(prompt_ends)), prompt_ends] # [b, d]
            influence_mapper_inpts[k] = (rg_img_reps, l_img_reps, prompt_end_reps)
        return influence_mapper_inpts, rg_influences

    def organize_batch_data(self, a_batch_of_training_data_paths:List):
        '''`a_batch_of_training_data_paths`: selected from output of `self.process_raw_data`. '''
        # load data
        edit_signal = {k:[] for k in self.adaptors.keys()}
        rel_data, gen_data, loc_data = [], [], []
        for edit_signal_dir_i, xym_dir_i in a_batch_of_training_data_paths:
            # load edit reps
            for k in edit_signal.keys():
                path = os.path.join(edit_signal_dir_i, k)
                d = torch.load(path, map_location = self.data_proc_device)
                edit_signal[k].append(d)
            # load middle reps
            path = os.path.join(xym_dir_i, self.mid_inpt_start_layer)
            rd, gd, ld = torch.load(path, map_location = self.data_proc_device)
            rel_data.append(rd)
            gen_data.append(gd)
            loc_data.append(ld)
        # organize edit signal
        batch_edit_reps, batch_edit_reps_att_mask, batch_prompt_end = {}, {}, {}
        for k, v in edit_signal.items():
            edit_reps = [signal['edit_reps'][0] for signal in v] # edit_reps[i]: [l,d]
            att_mask = [torch.ones([len(r)], device=self.data_proc_device) for r in edit_reps]
            prompt_end = [signal['prompt_end'] for signal in v] # prompt_end[i]: int
            edit_reps = pad_sequence(edit_reps, True) # [b, l_max, d]
            att_mask = pad_sequence(att_mask, True) # [b, l_max]
            prompt_end = torch.tensor(prompt_end, device=self.data_proc_device)
            batch_edit_reps[k] = edit_reps
            batch_edit_reps_att_mask[k] = att_mask
            batch_prompt_end[k] = prompt_end
        # organize training xym for rel/gen/loc
        def organize_middle_xym(embed_list):
            # input_embeds[i]: [1, l1, d]. label_ids[i]/label_masks[i]: [1, l2]
            input_embeds_vt_range, label_ids, label_masks = zip(*embed_list)
            input_embeds, vt_range = zip(*input_embeds_vt_range) 
            assert all(v == vt_range[0] for v in vt_range)
            vt_range = vt_range[0]
            max_inpt_len = max(i.shape[1] for i in input_embeds) 
            min_prompt_len = min(i.shape[1] - l.shape[1] for i, l in zip(input_embeds, label_ids)) 
            label_ids = [torch.cat([torch.zeros(i.shape[1]-l.shape[1]-min_prompt_len, device=self.data_proc_device), 
                l[0], torch.zeros(max_inpt_len - i.shape[1], device=self.data_proc_device)]).to(torch.long)
                for i, l in zip(input_embeds, label_ids)]
            label_ids = torch.stack(label_ids, 0) 
            label_masks = [torch.cat([torch.zeros(i.shape[1]-m.shape[1]-min_prompt_len, device=self.data_proc_device), 
                m[0], torch.zeros(max_inpt_len - i.shape[1], device=self.data_proc_device)]).to(torch.long)
                for i, m in zip(input_embeds, label_masks)]
            label_masks = torch.stack(label_masks, 0)
            att_masks = [torch.ones(i.shape[1], device=self.data_proc_device) for i in input_embeds]
            att_masks = pad_sequence(att_masks, True) 
            input_embeds = pad_sequence([e[0] for e in input_embeds], True) 
            mid_inpt = {'attention_mask': att_masks, 'inputs_embeds': input_embeds} 
            return (mid_inpt, vt_range), label_ids, label_masks
        rel_xym = organize_middle_xym(rel_data)
        gen_xym = {}
        loc_xym = {}
        for gen_name in gen_data[0].keys():
            gl = [d[gen_name] for d in gen_data]
            gen_xym[gen_name] = organize_middle_xym(gl)
        for loc_name in loc_data[0].keys():
            ll = [d[loc_name] for d in loc_data]
            loc_xym[loc_name] = organize_middle_xym(ll)
        # Reliability & Generality x and y for computing influence mapper loss
        infm_xy = self.__get_xy_for_influence_mapper__(rel_xym, gen_xym, loc_xym)
        # a batch of training data
        a_batch_of_training_data = move_to_device(((batch_edit_reps, batch_edit_reps_att_mask, 
                batch_prompt_end), rel_xym, gen_xym, loc_xym, infm_xy), self.device)
        return a_batch_of_training_data

    def train_a_batch(self, a_batch_of_training_data:Tuple):
        ((batch_edit_reps, batch_edit_reps_att_mask, batch_prompt_end), 
            rel_xym, gen_xym, loc_xym, infm_xy) = a_batch_of_training_data
        infer_vllm = self.vllm
        # prediction before edit for locality loss
        with torch.no_grad():
            self.open_adaptors(False)
            for loc_name in loc_xym.keys():
                (mid_inpt, vt_range), label_ids, label_masks = loc_xym[loc_name]
                pre_logits = self.infer_from_mid_layer(infer_vllm, mid_inpt, vt_range, 
                    self.mid_inpt_start_layer_i, mid_inpt['inputs_embeds']).logits
                loc_xym[loc_name] = (mid_inpt, vt_range), pre_logits, label_masks
        self.set_edit_signal_for_adaptors(batch_edit_reps, batch_edit_reps_att_mask, batch_prompt_end) # edit
        loss, log_dict = 0, {}
        self.open_adaptors(True)
        # Reliability loss
        (mid_inpt, vt_range), label_ids, label_masks = rel_xym
        logits = self.infer_from_mid_layer(infer_vllm, mid_inpt, vt_range, 
                    self.mid_inpt_start_layer_i, mid_inpt['inputs_embeds']).logits
        rel_loss = label_loss(logits, label_ids, label_masks) * self.cfg.train_cfg.rel_lambda
        log_dict['Reliability loss'] = float(rel_loss)
        loss += rel_loss
        # Generality loss
        log_dict['Generality loss'] = {}
        for loss_name, sp in gen_xym.items(): 
            (mid_inpt, vt_range), label_ids, label_masks = sp
            logits = self.infer_from_mid_layer(infer_vllm, mid_inpt, vt_range, 
                    self.mid_inpt_start_layer_i, mid_inpt['inputs_embeds']).logits
            gen_loss = label_loss(logits, label_ids, label_masks) * self.cfg.train_cfg.gen_lambda
            log_dict['Generality loss'][loss_name] = float(gen_loss)
            loss += gen_loss
        # Locality loss
        log_dict['Locality loss'] = {}
        for loss_name, sp in loc_xym.items():
            (mid_inpt, vt_range), pre_logits, label_masks = sp
            post_logits = self.infer_from_mid_layer(infer_vllm, mid_inpt, vt_range, 
                    self.mid_inpt_start_layer_i, mid_inpt['inputs_embeds']).logits
            loc_loss = logit_KL_loss(pre_logits, post_logits, label_masks) * self.cfg.train_cfg.loc_lambda
            log_dict['Locality loss'][loss_name] = float(loc_loss)
            loss += loc_loss
        # influence mapper loss for rel/gen
        if self.cfg.IT.add_it:
            influence_mapper_inpts, rg_influences = infm_xy # [b, 48], rg_influences[i].sum() = 1, 
            infm_lambda = self.cfg.train_cfg.inf_mapper_lambda
            for k in self.adaptors.keys():
                rg_img_reps, l_img_reps, prompt_end_reps = influence_mapper_inpts[k]
                infm_rg_pre = self.adaptors[k].influence_mapper(rg_img_reps, prompt_end_reps) # [b, 48]
                infm_l_pre = self.adaptors[k].influence_mapper(l_img_reps, prompt_end_reps) # [b, 48]
                log_dict['Influence predict sigmoid rel-gen mean-%s'%k] = float(torch.sigmoid(infm_rg_pre).mean())
                log_dict['Influence predict sigmoid loc mean-%s'%k] = float(torch.sigmoid(infm_l_pre).mean())
                # losses
                rg_relative_inf_loss = -(rg_influences * torch.log(torch.softmax(infm_rg_pre, 1) + 1e-8)).sum(1).mean(0) * infm_lambda
                # y = 0.95
                # rg_up_loss = - (y * torch.log(torch.sigmoid(infm_rg_pre) + 1e-8) 
                #     + (1 - y) * torch.log(1 - torch.sigmoid(infm_rg_pre) + 1e-8)).mean() * infm_lambda
                rg_up_loss = -torch.log(torch.sigmoid(infm_rg_pre) + 1e-8).mean() * infm_lambda
                l_down_loss = -torch.log(1 - torch.sigmoid(infm_l_pre) + 1e-8).mean() * infm_lambda
                log_dict['Influence mapper rel-gen relative loss-%s'%k] = float(rg_relative_inf_loss)
                log_dict['Influence mapper rel-gen up loss-%s'%k] = float(rg_up_loss)
                log_dict['Influence mapper loc down loss-%s'%k] = float(l_down_loss)
                infm_loss = rg_relative_inf_loss + rg_up_loss + l_down_loss
                loss += infm_loss
        # update
        loss.backward()
        self.opt.step()
        self.opt.zero_grad()
        return float(loss), log_dict

    def get_a_new_optimizer(self)->torch.optim.Optimizer:
        para_lr = [{'params':adaptor.parameters(), 'lr': self.cfg.train_cfg.lr} 
                   for adaptor in self.adaptors.values()]
        return Adam(para_lr)
    
    def set_train(self, if_train:bool):
        '''
        Set training state for editor.
        '''
        self.vllm.model.eval()
        self.vllm.model.requires_grad_(False)
        for k in self.adaptors.keys():
            self.adaptors[k].requires_grad_(if_train)
            self.adaptors[k].train(if_train)

def label_loss(logits, label_ids, masks, average = True):
    # logits: [batch_size, total_l, d], label_ids/masks: [batch_size, short_l]
    logits = logits[:, -label_ids.shape[1]:]
    log_pre_p = torch.log_softmax(logits, -1)
    log_pre_p = log_pre_p.gather(-1, label_ids.unsqueeze(-1)).squeeze(-1) # [batch, short_l]
    loss = -(log_pre_p * masks).sum()
    if average:
        loss = loss / masks.sum() 
    return loss

def logit_KL_loss(logits1, logits2, masks, average = True):
    # logits1/logits2: [batch, total_l, voc_size], masks: [batch, short_l]
    logits1 = logits1[:, -masks.shape[1]:]
    logits2 = logits2[:, -masks.shape[1]:]
    log_p1 = torch.log_softmax(logits1, -1)
    log_p2 = torch.log_softmax(logits2, -1)
    p1 = torch.softmax(logits1, 2)
    kl_loss = (p1 * (log_p1 - log_p2)).sum(2) # [batch, max_len]
    loss = (kl_loss * masks).sum()
    if average:
        loss = loss / masks.sum() 
    return loss

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
