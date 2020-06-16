通过调用DataLoader.py里的get_dataloader(mode)函数，获得train_dataloader和valid_dataloader，其中mode参数可选为"price"或"diff"，表示为股价净值（经过正态归一化），或者为当天股价与前天股价的差值。

选用哪个mode自己挑。

数据集规模：batch_size=16，train_dataloader的每个batch为：[batch_size=16, seq_len=10, stock_num=22]
对CNN来说，可以看成是batch=16, input_channel=1, height=10, width=22的一个数据集，这个数据集的卷积核对height可以自选，对width为1。
（从上证50中挑出了上市10年的22支股票，因为上证50中有部分股票数据上市较晚，训练数据规模相差较大，难以一起训练）
（将时间窗口大小设置为10，即从上两周交易日的信息预测接下来一天的股价）