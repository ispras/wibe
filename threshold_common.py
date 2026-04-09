import os
import sys
import argparse
import yaml
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import torch
import time
from wibench.config_loader import load_pipeline_config_yaml

cache_base = "/fast-drive/alisaustinova/img-watermarking-test/hf_datasets_cache"
os.makedirs(cache_base, exist_ok=True)

os.environ["HF_HOME"] = cache_base
os.environ["HF_DATASETS_CACHE"] = os.path.join(cache_base, "datasets")
os.environ["HF_HUB_CACHE"] = os.path.join(cache_base, "hub")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(cache_base, "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(cache_base, "transformers")
os.environ["XDG_CACHE_HOME"] = cache_base

print("="*60)
print(f"Кэш перенаправлен на: {cache_base}")

import shutil
total, used, free = shutil.disk_usage(cache_base)
print(f"Свободно на диске: {free // (2**30)} GB")
print("="*60)

from wibench.datasets.diffusiondb import DiffusionDB
from wibench.algorithms.base import BaseAlgorithmWrapper

from wibench.algorithms.maxsive.wrapper import MaXsiveWrapper
from wibench.algorithms.ringid.wrapper import RingIDWrapper
from wibench.algorithms.dft_circle.wrapper import DFTMarkerWrapper

def get_percentile_and_reverse(method_name: str, target_fpr: float, method_wrapper=None) -> tuple:
    """
    Возвращает (percentile, reverse) для заданного метода.
    
    Для DFT Circle: extract возвращает корреляцию (0-1), маркированные имеют БОЛЬШУЮ корреляцию
    Для MaXsive: extract возвращает score (корреляция или L1)
    Для RingID: extract возвращает расстояние, маркированные имеют МЕНЬШЕЕ расстояние
    """
    if method_name == "ringid":
        #меньшее расстояние = маркированное
        return target_fpr * 100, True
    
    elif method_name == "maxsive":
        if method_wrapper and hasattr(method_wrapper, 'params'):
            distant_func = method_wrapper.params.distant_func
        else:
            distant_func = "corr"
        
        if distant_func == "l1":
            # L1: меньшее значение = маркированное
            return target_fpr * 100, True
        else:
            # Корреляция: большее значение = маркированное
            return (1 - target_fpr) * 100, False
    
    elif method_name == "dft_circle":
        # DFT Circle: корреляция большее = маркированное
        return (1 - target_fpr) * 100, False
    
    else:
        return (1 - target_fpr) * 100, False


def main():
    parser = argparse.ArgumentParser(description="Расчет порога для методов водяных знаков")
    
    parser.add_argument("--fpr", "-f", type=float, default=0.01,
                        help="Целевой FPR (например, 0.01 для 1%%)")
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="Путь к YML конфигу метода")
    parser.add_argument("--samples", "-s", type=int, default=None,
                        help="Количество образцов")
    parser.add_argument("--dataset", "-d", type=str, default="diffusiondb",
                        help="Количество образцов")
    parser.add_argument("--subset", type=str, default="2m_first_5k",
                        help="Subset датасета DiffusionDB")
    
    args = parser.parse_args()
    
    
    if args.config:
        full_config = load_pipeline_config_yaml(args.config)
       
    method, method_params = full_config['algorithms'][0]
    method_params = method_params if method_params is not None else {}

    # Вычисляем количество образцов
    if args.samples:
        required_samples = args.samples
    else:
        required_samples = int(10 / args.fpr)
    
    print("="*60)
    print(f"РАСЧЕТ ПОРОГА ДЛЯ МЕТОДА: {method.upper()}")
    print("="*60)
    print(f"Целевой FPR: {args.fpr:.2%}")
    print(f"Необходимо образцов: {required_samples}")
    
    method_class = BaseAlgorithmWrapper._registry.get(method)
    if method_class is None:
        raise ValueError(f"Метод не найден в реестре")
    
    method_wrapper = method_class(method_params)
    
    percentile, reverse = get_percentile_and_reverse(method, args.fpr, method_wrapper)
    
    print(f"Метрика: {'расстояние (меньше = маркированное)' if reverse else 'корреляция (больше = маркированное)'}")
    print(f"Перцентиль: {percentile:.2f}%")
   
    print("\nЗагрузка датасета DiffusionDB...")
    
    dataset = DiffusionDB(
        subset=args.subset,
        sample_range=(0, required_samples),
        return_prompt=False,
        skip_nsfw=True,
        cache_dir=os.path.join(cache_base, "diffusiondb")
    )
    
    print(f"Размер датасета: {len(dataset)}")
   
    print("\nСбор статистики немаркированных...")
    scores = []
    generator = dataset.generator()
    processed = 0
    start_time = time.time()
    
    for obj in generator:
        if processed >= required_samples:
            break
        
        try:
            img_tensor = obj.image
            watermark_data = method_wrapper.watermark_data_gen()
            score = method_wrapper.extract(img_tensor, watermark_data)
            
            scores.append(float(score))
            processed += 1
            
            if processed % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  Обработано {processed} / {required_samples} (скорость: {processed/elapsed:.2f} изоб/сек)")
                
        except Exception as e:
            print(f"  Ошибка: {e}")
            processed += 1
            continue
    
    print(f"\n Собрано {len(scores)} валидных немаркированных изображений")
    
    if len(scores) == 0:
        print("\n Нет валидных данных для вычисления порога")
        sys.exit(1)
   
    df_scores = pd.DataFrame(scores, columns=['score'])
    df_scores.to_csv(f"{method}_scores.csv", index=False)
    print(f" Сырые значения сохранены в {method}_scores.csv")
    
    threshold = float(np.percentile(scores, percentile))
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТ")
    print("="*60)
    print(f"Метод: {method}")
    print(f"Целевой FPR: {args.fpr:.2%}")
    print(f"Перцентиль: {percentile:.2f}%")
    print(f"Порог: {threshold:.6f}")
    print(f"Reverse: {reverse} ({'меньше = маркированные' if reverse else 'больше = маркированные'})")
    print(f"Валидных образцов: {len(scores)}")
    
    # Сохраняем результат
    result = pd.DataFrame([{
        "method": method,
        "target_fpr": args.fpr,
        "percentile": percentile,
        "threshold": threshold,
        "reverse": reverse,
        "valid_samples": len(scores)
    }])
    result.to_csv(f"{method}_Threshold.csv", index=False)
    print(f"\n Результат сохранен в {method}_Threshold.csv")


if __name__ == "__main__":
    main()