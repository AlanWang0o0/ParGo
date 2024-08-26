"""Utilities."""

import os
from PIL import Image
from torch.hub import download_url_to_file, urlparse, get_dir
from unit.utils.hublist import XHUB, getCachedBasename


# helper functions
def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")

def is_hdfs(path):
    return path.startswith('hdfs')

def is_dir(path:str):
    return (path.endswith('/')) or ('.' not in os.path.basename(path))


def import_from(module, name):
    module = __import__(module, fromlist=[name])
    return getattr(module, name)


def get_cache_dir(child_dir=''):
    hub_dir = get_dir()
    child_dir = () if not child_dir else (child_dir,)
    model_dir = os.path.join(hub_dir, 'checkpoints', *child_dir)
    os.makedirs(model_dir, exist_ok=True)
    return model_dir


def download_cached_file(path, root='./', progress=False):
    if is_url(path):
        parts = urlparse(path)
        filename = os.path.basename(parts.path)
        cached_file = os.path.join(get_cache_dir(), filename)
        if not os.path.exists(cached_file):
            download_url_to_file(path, cached_file, None, progress=progress)
    elif is_hdfs(path):
        filename = os.path.basename(path)
        os.makedirs(root, exist_ok=True)
        cached_file = os.path.join(root, filename)
        if not os.path.exists(cached_file):
            os.system(f'hdfs dfs -get {path} {cached_file}')
    return cached_file


def remove_exif(image_name):
    image = Image.open(image_name)
    if not image.getexif():
        return
    data = list(image.getdata())
    image_without_exif = Image.new(image.mode, image.size)
    image_without_exif.putdata(data)
    return image_without_exif
