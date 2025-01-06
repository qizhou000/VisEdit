from editor.vllm_editors.base import VLLMBaseEditorWithTraining
from editor.vllms_for_edit.base import BaseVLLMForEdit
from utils.GLOBAL import model_path_map, ROOT_PATH
from typing import Union, List, Dict, Optional
from transformers import AutoModelForCausalLM
from torch import nn
import torch, os

def find_module(module, module_path:str)->Union[torch.Tensor, nn.Module]:
    for comp in module_path.split('.'):
        if hasattr(module, comp):
            module = getattr(module, comp)
        elif comp.isdigit():
            module = module[int(comp)]
        else:
            raise RuntimeError(f"Couldn't find child module {comp}")
    return module
 
def move_to_device(data, device):
    '''Move list and dictionary nested PyTorch tensors to a specific device.'''
    if isinstance(data, (torch.Tensor, nn.Module)):
        return data.to(device)
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple([move_to_device(item, device) for item in data])
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, (int, float, str, bool, type(None))):
        return data
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

def get_full_model_name(model_name_part:str)->str:
    model_name_part = model_name_part.lower()
    if 'blip2' in model_name_part:
        return 'blip2-opt-2.7b'
    elif 'llava' in model_name_part:
        return 'llava-v1.5-7b'
    elif 'mini' in model_name_part:
        if '4' in model_name_part and 'gpt' in model_name_part:
            return 'minigpt-4-vicuna-7b'
        else:
            raise
    raise

def get_editor_config_path(editor_name:str, edit_model_name:str):
    path = os.path.join(ROOT_PATH, 'configs', editor_name.lower(), '%s.yaml'%get_full_model_name(edit_model_name))
    return path

def get_model_path(model_name:str)->str:
    return model_path_map[get_full_model_name(model_name)]


def load_vllm_for_edit(model_name:str, device:str)->BaseVLLMForEdit:
    model_name = get_full_model_name(model_name)
    model_path = get_model_path(model_name)
    print('Loading %s from "%s".'%(model_name, model_path))
    if 'llava' in model_name:
        from editor.vllms_for_edit import LlavaForEdit
        return LlavaForEdit(model_path, device, True)
    elif 'blip2' in model_name:
        from editor.vllms_for_edit import BLIP2OPTForEdit
        return BLIP2OPTForEdit(model_path, device)
    elif 'mini' in model_name and 'gpt' in model_name and '4' in model_name:
        from editor.vllms_for_edit import MiniGPT4ForEdit
        return MiniGPT4ForEdit(model_path, device, True)
    raise BaseException('Have not write `BaseVLLMForEdit` for `%s`.'%model_name)

def load_vllm_editor(editor_name:str, edit_model_name:str, device:int, 
        extra_devices:List[int] = [1], editor_ckpt_path = None, for_train = False):
    editor_name = editor_name.lower()
    config_path = get_editor_config_path(editor_name, edit_model_name)
    vllm = load_vllm_for_edit(edit_model_name, device)
    # load editor
    if editor_name == 'vead':
        from editor.vllm_editors.vead.vead import VEAD, VEADConfig
        data_proc_device = 'cuda:%s'%extra_devices[0] if for_train else None
        vllm_data_proc = load_vllm_for_edit(edit_model_name, data_proc_device) if for_train else None
        config = VEADConfig.from_yaml(config_path)
        train_data_cache_root = os.path.join(ROOT_PATH, 'data')
        editor = VEAD(vllm, config, device, vllm_data_proc, data_proc_device, train_data_cache_root)
    else:
        raise 
    if editor_ckpt_path != None and isinstance(editor, VLLMBaseEditorWithTraining):
        editor.load_ckpt(editor_ckpt_path, True, False)
    return editor

