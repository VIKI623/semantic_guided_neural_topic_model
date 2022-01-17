import joblib
import torch
import json
from typing import Iterable, Dict, Any


def save_object_to_file(obj, full_path):
    with open(full_path, 'wb') as f:
        joblib.dump(obj, f)


def restore_object_from_file(full_path):
    try:
        with open(full_path, 'rb') as f:
            obj = joblib.load(f)
    except FileNotFoundError:
        return None
    return obj


def save_json(full_path, data, indent=4):
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def read_json(full_path):
    with open(full_path, 'r', encoding='utf-8') as f:
        load_dict = json.load(f)
    return load_dict


def save_model(full_path, torch_model):
    torch.save(torch_model.state_dict(), full_path)


def load_model(full_path, torch_model):
    torch_model.load_state_dict(torch.load(full_path))


def save_txt(full_path, lines, delimiter=' '):
    with open(full_path, 'w', encoding='utf-8') as f:
        for line in lines:
            if isinstance(line, list):
                line = delimiter.join((str(ele) for ele in line))
            else:
                line = str(line)
            f.write(line + '\n')


def read_txt(full_path, delimiter=' ', convert_func=None):
    with open(full_path, 'r', encoding='utf-8') as f:
        if convert_func is None:
            load_txt = [ele for line in f for ele in line.split(delimiter)]
        else:
            load_txt = [convert_func(ele) for line in f for ele in line.split(delimiter)]
    return load_txt


def save_jsons(full_path: str, jsons: Iterable[Dict[str, Any]]):
    with open(full_path, 'w', encoding='utf-8') as f:
        for obj in jsons:
            f.write(json.dumps(obj) + '\n')


def read_jsons(full_path: str):
    objs = []
    with open(full_path, 'r', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line.strip())
            objs.append(obj)
    return objs
