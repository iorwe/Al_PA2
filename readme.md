# 情感分类模型

## 运行环境

- Python 3.11.8
- PyTorch 2.2.2
- tqdm
- gensim

请确保安装了以上库，以保证程序正常运行。

## 使用方法

- 在文件根目录下使用命令

```bash
	python main.py
```

- 默认使用隐藏层单元为**100**的**MLP**模型。
- 训练前需要确认模型参数信息，如果已经存在训练好的模型会询问是否要重新训练模型。
- 可以通过修改 `main.py` 中的参数来选择不同的模型和配置。

## 可选参数

| 命令        | 参数            |                    用途                    | 默认值  |
| ----------- | :-------------- | :----------------------------------------: | ------- |
| -m <str>    | model           | 选择模型，可选参数: cnn, rnn,lstm, gru,mlp | mlp     |
| -e <int>    | epoch           |                最大迭代次数                | 100     |
| -b <int>    | batch_size      |                  批量大小                  | 64      |
| -lr <float> | learning_rate   |                   学习率                   | 0.01    |
| -sl <int>   | sequence_length |                  句子词数                  | 64      |
| -t <str>    | if_train        |     是否训练新模型，可选参数: Y/N(y/n)     | Y       |
| -nf <int>   | num_filters     |             CNN模型卷积核个数              | 100     |
| -fz <int>   | filter_size     |            CNN卷积核大小与数量             | [2,3,4] |
| -hd <int>   | hidden_dim      |                   隐藏层                   |         |
|             |                 |                                            |         |