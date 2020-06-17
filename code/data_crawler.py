import tushare as ts

ts.set_token('hhhhhhhhh')
pro = ts.pro_api()
#print(ts.get_sz50s().code[0])

f = open("data", "w", encoding='utf-8')
c = ts.get_sz50s().code
length_ls = []
for i in range(50):
    df = pro.daily(ts_code=c[i]+".SH", start_date='20110101', end_date='20200614')
    print(i)
    if len(df.ts_code.tolist()) < 2284:
        continue


    ls = []
    ls.append(df.ts_code.tolist()[0])   # 代码
    ls.append(df.trade_date.tolist()[::-1])    # 日期
    ls.append(df.close.tolist()[::-1])     #  收盘价
    ls.append(df.pre_close.tolist()[::-1]) #   昨日收盘价
    ls.append(df.vol.tolist()[::-1])       #   成交量
    ls.append(df.amount.tolist()[::-1])    #   成交额

    length_ls.append(len(df.ts_code.tolist()))

    f.write(str(ls)+"\n")

print(length_ls)




#66b40e5fd57c190c72fbf1d755598934d2fce80ee978e4c5610d66be
