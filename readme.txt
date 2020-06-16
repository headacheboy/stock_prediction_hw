通过调用DataLoader.py里的get_dataloader(mode)函数，获得train_dataloader和valid_dataloader，其中mode参数可选为"price"或"diff"，表示为股价净值（经过正态归一化），或者为当天股价与前天股价的差值。

统一选用price mode（即mode="price"）。

数据集规模：batch_size=16，train_dataloader的每个batch为：[batch_size=16, seq_len=10, stock_num=22]
对CNN来说，可以看成是batch=16, input_channel=1, height=10, width=22的一个数据集，这个数据集的卷积核对height可以自选，对width为1。
（从上证50中挑出了上市10年的22支股票，因为上证50中有部分股票数据上市较晚，训练数据规模相差较大，难以一起训练）
（将时间窗口大小设置为10，即从上两周交易日的信息预测接下来一天的股价）

最后的评价指标中，使用inverse_transform_MSE，即对预测出来的[458, 22]个测试数据，用scaler.inverse_transform()变换之后，求MSE。
具体代码如下：

y_tot_all和y_pred_all为维数=[458, 22]的np.array，其中y_tot_all表示测试数据的经过scaler变换之后的ground truth，即直接将valid_dataloader里的数据读出来即可。y_pred_all为预测的数据。

y_tot_all = scaler.inverse_transform(y_tot_all)
y_pred_all = scaler.inverse_transform(y_pred_all)
val_loss = math.sqrt(np.sum((y_pred_all - y_tot_all)**2) / (y_pred_all.shape[0] * y_pred_all.shape[1]))
