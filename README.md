# hlht_Function
 

#### 文件结构

| 目录名           | 作用                   |
|:--------------|:---------------------|
| performance   | 性能评测工具               |
| function      | 功能评测工具               |


## Usage

### Environment setup

``````shell
conda activate hlht
``````
安装环境依赖

``````shell
pip install -r required.txt
``````

### Add model
将模型权重文件放入model_weights目录下，将模型结构定义在`model.py`文件中，并修改``ptcv_get_model``函数定义。
### Run

测试模型功能：
``````shell
python funtest.py
``````
测试模型性能：
``````shell
python pertest.py
``````
### 工具结构
```angular2html
|-- README.md
|-- data
|   |-- dataset
|   |   |-- imagenet
|   |   |-- meta.txt
|   |-- SST-2
|   |   |-- test.tsv
|   |   |-- train.tsv
|   |   |-- dev.tsv
|-- model_bert
|   |-- config.json
|   |-- pytorch_model.bin
|   |-- tokenizer_config.json
|   |-- train_args.json
|   |-- vocab.txt
|-- function
|   |-- function.py
|   |-- nlp_function.py
|-- performance
|   |-- performance.py
|   |-- nlp_performance.py
|-- model_weights
|   |-- resnet50-0633-b00d1c8e.pth
|   |-- vgg16-0865-5ca155da.pth.pth
|-- dataset.py
|-- funtest.py
|-- pertest.py
|-- model.py
|-- README.md
|-- required.txt
|-- select_imagenetmodel.py
```

### Demo
提供测试例子vgg16和resnet50，模型参数在model_weights目录下。
