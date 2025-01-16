import copy
import re, ast
from transformers import AutoConfig, LlamaConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from easydict import EasyDict as EasyDict
from importlib import import_module
import os.path as osp
import argparse
import json
from copy import deepcopy
import sys
import yaml


BASE_KEY = "_base_"
BASE_CONFIG = {}

class VideoChat2Config(PretrainedConfig):
    model_type = "InternVideo2_VideoChat2"

    def __init__(
            self,
            model_config=None,
            **kwargs):
        super().__init__(**kwargs)
        
        # model_config가 문자열(파일 경로)인 경우 파일에서 읽기
        if isinstance(model_config, str):
            with open(model_config, 'r') as f:
                model_config = json.load(f)
                
        self.model_config = EasyDict(model_config)
        
        # config.json의 주요 설정들을 클래스 속성으로 설정
        for key, value in self.model_config.items():
            setattr(self, key, value)

    @classmethod
    def pretty_text(cls, cfg: dict, indent=2) -> str:
        """format dict to a string

        Args:
            cfg (EasyDict): the params.

        Returns: The string to display.

        """
        msg = "{\n"
        for i, (k, v) in enumerate(cfg.items()):
            if isinstance(v, dict):
                v = cls.pretty_text(v, indent + 4)
            spaces = " " * indent
            msg += spaces + "{}: {}".format(k, v)
            if i == len(cfg) - 1:
                msg += " }"
            else:
                msg += "\n"
        return msg

    @classmethod
    def dump(cls, cfg, savepath=None):
        """dump cfg to `json` file.

        Args:
            cfg (dict): The dict to dump.
            savepath (str): The filepath to save the dumped dict.

        Returns: TODO

        """
        if savepath is None:
            savepath = osp.join(cfg.WORKSPACE, "config.json")
        json.dump(cfg, open(savepath, "w"), indent=2)

    @classmethod
    def get_config(cls, default_config: dict = None):
        """get a `Config` instance.

        Args:
            default_config (dict): The default config. `default_config` will be overrided
                by config file `--cfg`, `--cfg` will be overrided by commandline args.

        Returns: an EasyDict.
        """
        global cfg
        if cfg is not None:
            return cfg

        # define arg parser.
        parser = argparse.ArgumentParser()
        # parser.add_argument("--cfg", help="load configs from yaml file", default="", type=str)
        parser.add_argument(
            "config_file", help="the configuration file to load. support: .yaml, .json, .py"
        )
        parser.add_argument(
            "opts",
            default=None,
            nargs="*",
            help="overrided configs. List. Format: 'key1 name1 key2 name2'",
        )
        args = parser.parse_args()

        cfg = EasyDict(BASE_CONFIG)
        if osp.isfile(args.config_file):
            cfg_from_file = cls.from_file(args.config_file)
            cfg = merge_a_into_b(cfg_from_file, cfg)
        cfg = cls.merge_list(cfg, args.opts)
        cfg = eval_dict_leaf(cfg)

        # update some keys to make them show at the last
        for k in BASE_CONFIG:
            cfg[k] = cfg.pop(k)
        return cfg

    @classmethod
    def from_file(cls, filepath: str) -> EasyDict:
        """Build config from file. Supported filetypes: `.py`,`.yaml`,`.json`.

        Args:
            filepath (str): The config file path.

        Returns: TODO

        """
        filepath = osp.abspath(osp.expanduser(filepath))
        if not osp.isfile(filepath):
            raise IOError(f"File does not exist: {filepath}")
        if filepath.endswith(".py"):
            sys.path.insert(0, osp.dirname(filepath))
            mod = import_module(osp.splitext(osp.basename(filepath))[0])
            cfg_dict = {
                name: value
                for name, value in mod.__dict__.items()
                if not name.startswith("__")
            }

        elif filepath.endswith((".yml", ".yaml")):
            cfg_dict = VideoChat2Config.load(open(filepath, "r"), Loader=yaml.Loader)
        elif filepath.endswith(".json"):
            cfg_dict = json.load(open(filepath, "r"))
        else:
            raise IOError("Only py/yml/yaml/json type are supported now!")

        cfg_text = filepath + "\n"
        with open(filepath, "r") as f:
            cfg_text += f.read()

        if BASE_KEY in cfg_dict:  # load configs in `BASE_KEY`
            cfg_dir = osp.dirname(filepath)
            base_filename = cfg_dict.pop(BASE_KEY)
            base_filename = (
                base_filename if isinstance(base_filename, list) else [base_filename]
            )

            cfg_dict_list = list()
            for f in base_filename:
                _cfg_dict = VideoChat2Config.from_file(osp.join(cfg_dir, f))
                cfg_dict_list.append(_cfg_dict)

            base_cfg_dict = dict()
            for c in cfg_dict_list:
                if len(base_cfg_dict.keys() & c.keys()) > 0:
                    raise KeyError("Duplicate key is not allowed among bases")
                base_cfg_dict.update(c)

            cfg_dict = merge_a_into_b(cfg_dict, base_cfg_dict)

        return EasyDict(cfg_dict)
    
def merge_a_into_b(a, b, inplace=False):
    """The values in a will override values in b.

    Args:
        a (dict): source dict.
        b (dict): target dict.

    Returns: dict. recursively merge dict a into dict b.

    """
    if not inplace:
        b = deepcopy(b)
    for key in a:
        if key in b:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                b[key] = merge_a_into_b(a[key], b[key], inplace=True)
            else:
                b[key] = a[key]
        else:
            b[key] = a[key]
    return b


def eval_dict_leaf(d, orig_dict=None):
    """eval values of dict leaf.

    Args:
        d (dict): The dict to eval.

    Returns: dict.

    """
    if orig_dict is None:
        orig_dict = d
    for k, v in d.items():
        if not isinstance(v, dict):
            d[k] = eval_string(v, orig_dict)
        else:
            eval_dict_leaf(v, orig_dict)
    return d


def eval_string(string, d):
    """automatically evaluate string to corresponding types.

    For example:
        not a string  -> return the original input
        '0'  -> 0
        '0.2' -> 0.2
        '[0, 1, 2]' -> [0,1,2]
        'eval(1+2)' -> 3
        'eval(range(5))' -> [0,1,2,3,4]
        '${a}' -> d.a



    Args:
        string (str): The value to evaluate.
        d (dict): The

    Returns: the corresponding type

    """
    if not isinstance(string, str):
        return string
    # if len(string) > 1 and string[0] == "[" and string[-1] == "]":
    #     return eval(string)
    if string[0:5] == "eval(":
        return eval(string[5:-1])

    s0 = string
    s1 = re.sub(r"\${(.*)}", r"d.\1", s0)
    if s1 != s0:
        while s1 != s0:
            s0 = s1
            s1 = re.sub(r"\${(.*)}", r"d.\1", s0)
        return eval(s1)

    try:
        v = ast.literal_eval(string)
    except:
        v = string
    return v
