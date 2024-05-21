# Итеративная подборка коллекции релевантных документов

### 1. Пример запуска обучения на датасете **MS MARCO**:
```python
python train.py \
    --save_dir logs/ \
    --data_dirs ./datasets/training-data/msmarco-data \
    --weights 100 \
    --batch_size 32 \
    --num_workers 16 \
    --steps 200000 \
    --val_check_interval 10000 \
    --pooling mean \
    --loss mnrl \
    --sampling mixed \
    --fp16 \
    --gpus 3
```
### 2. Пример запуска дообучения на датасете синтетических данных:
```python
python train.py \
    --save_dir logs/ \
    --data_dirs ./datasets/training-data/synthetic \
    --weights 100 \
    --batch_size 32 \
    --num_workers 16 \
    --steps 100000 \
    --val_check_interval 5000 \
    --pooling mean \
    --loss mnrl \
    --sampling mixed \
    --fp16 \
    --checkpoint_path "your start checkpoint path"\
    --gpus 1
```

### 3. Пример запуска обучения на нескольких датасетах c использованием смешивания внутри одного пакета:
```python
python train.py \
    --save_dir logs/ \
    --data_dirs ./datasets/training-data/unarxiv-d2d ./datasets/training-data/unarxiv-q2d ./datasets/training-data/msmarco-data \
    --weights 33 33 34 \
    --batch_size 32 \
    --num_workers 16 \
    --steps 100000 \
    --val_check_interval 5000 \
    --pooling mean \
    --loss mnrl \
    --sampling mixed \
    --fp16 \
    --gpus 0
```

### 4. Пример запуска обучения на нескольких датасетах:
```python
python train.py \
    --save_dir logs/ \
    --data_dirs ./datasets/training-data/independent-cropping ./datasets/training-data/unarxiv-d2d ./datasets/training-data/unarxiv-q2d ./datasets/training-data/msmarco-data \
    --batch_size 32 \
    --num_workers 16 \
    --steps 100000 \
    --val_check_interval 5000 \
    --pooling mean \
    --loss mnrl \
    --sampling alternate \
    --fp16 \
    --gpus 1
```

### 5. Пример запуска валидации на поднаборах scidocs, scifact, arguana:
```python
python evaluate.py \
    --models "your checkpoint path"\
    --datasets scidocs scifact arguana \
    --batch_size 32
```
----
### Данные:
**training-data:**
1. Синтетические данные: [InPars-v1](https://github.com/zetaalphavector/InPars?tab=readme-ov-file)
2. Неразмеченные данные: [independent-cropping](https://drive.google.com/file/d/1_eCs_LhvvYm0lSaXR6byd3VJHq1kfUzG/view?usp=sharing)
3. unarXive: [doc-doc](https://drive.google.com/file/d/17eBHeFj0Vbs4sFdwjdods7kZJSL-hG9p/view?usp=sharing), [query-doc](https://drive.google.com/file/d/1rXWG-m1ZQc-GcUOlGCLMFLN4dP3NE3nN/view?usp=sharing)
**evaluation-data:**
1. [SciDocs](https://drive.google.com/file/d/1kyP7ytWFIxQ7d0H0sldvR9yIPU9fkVFN/view?usp=sharing)
2. ArguAna в папке datasets
3. SciFact в папке datasets
