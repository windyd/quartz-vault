---
aliases:
  - "{ VALUE:title }": 
cssclass: 
ReviewedDate: "[[Daily_Notes/15-10-24]]"
tags: my/summary,state/process
child:
---

## DuckDB
duckdb 是一个轻量的数据库, 同时对不同格式的数据有较好的适配。
### Install
> ref::   [install guide](https://duckdb.org/docs/installation/index?version=stable&environment=cli&platform=macos&download_method=package_manager)

1. [[美美的数据 PlayGround#^822521|打开命令行]]
2. 用工具 brew 安装 duckdb
	```bash
	brew install duckdb
	```
3. 打开 duckdb 交互式界面
	```bash
	duckdb
	```
4. (Optional) 我们的硬盘小的可怜，有必要有一个 httpfs 的功能
	```sql
	INSTALL httpfs;
	LOAD httpfs;
	```
### SQL 基本结构
sql 的基本结构如下

```sql
SELECT ... AS -- 要哪些列
FROM ... AS -- 从哪里拿（duckdb 可以直接从本地/远程文件进行读取）
WHERE ... -- optional: 条件筛选
GROUP BY ... -- optional: 按 x 分组聚合
ORDER BY ... -- optional: 按 x 排序
LIMIT ... -- optional: 只显示 x 行
```

有了如上的结构，我们就能得到一张表。


如果需要对两个表格进行拼接操作
- 假设对 p1, p2 两列为用于拼接的列
- 我们通常需要重命名表格来区分 *不同表格中的同一列*

```sql
SELECT ...
FROM ... AS T
JOIN OtherTable -- 另外一个表
ON T.p1 = OtherTable.p1 AND T.p2 = OtherTable.p2
```


知道这个以后我们就可以开始练手了！
### Mobile Vehicle Dataset
[kaggle link](https://www.kaggle.com/datasets/arnavsmayan/vehicle-manufacturing-dataset?resource=download) 下可以下载数据

#### 习题
1. 求最大的 Price
2. 求 2019 年，每个品牌 Brand 卖的最便宜的价格
#### Challenge

最新三年 (2018,2019,2020)， 按城市分组, 最便宜的
- 品牌
- 车 Brand
- ...

我们逐渐问题拆分开来
1. 找到最新的三年 <- 一个查询可以搞定
2. 按照最新的三年过滤: 然后按照城市分组求最便宜的价格对应的价格，以及相应的品牌，Year，颜色等信息 <- 一个查询 ==搞不定==，需要进一步拆分. [[美美玩 SQL#^217627|为什么一个 groupby 搞不定？]]


因此我们的问题进一步演化为
1. （查询1）找到最新的三年
2. （查询2）按照最新的三年过滤:
	1. （子查询2.1）按照 Location 进行分组，求小的 Price, 然后把表格命名为 MinPrices
	2. （子查询2.2）用 2.1 的查到的数据 Location, MinPrice 反查原表

 [[美美玩 SQL#^solution-1326|答案]]




## Misc


> [!NOTE] How Group By Works
> 当我们进行 group by 的时候，每一个 group 里面:
> - Group By 的 key 会输出一行
> - 其他的列会被汇聚到一起，waiting to be 进行聚合成一行
> 
> ```sql
> SELECT 
>     MIN(Price) -- 多行的 Price, 通过 MIN 汇聚成一行
> 	Location -- Group By 的 KEY
> FROM '/Users/dorawong/Downloads/Car Data.csv'
> GROUP BY Location 
> LIMIT 3
> ```
> 
> 以下是一个不 work 的例子
> ```sql
> SELECT 
>     Price -- ⚠️ 多行的 Price, 没有被汇聚成一行
> 	Location -- Group By 的 KEY
> FROM '/Users/dorawong/Downloads/Car Data.csv'
> GROUP BY Location 
> LIMIT 3
> ```
> 

^217627



> [!NOTE] Solution: 最近三年，按城市分组最小价格的各数据
> 因此我们的问题进一步演化为
> 1. （查询1）找到最新的三年
> 	```SQL
> 	SELECT DISTINCT(Year)
> 	FROM '/Users/dorawong/Downloads/Car Data.csv'
> 	ORDER BY Year DESC -- DESC: descending
> 	LIMIT 3
> 	```
> 	得知是 2018, 2019, 2020
> 2. （查询2）按照最新的三年过滤:
> 	1. （子查询2.1）按照 Location 进行分组，求小的 Price, 然后把表格命名为 MinPrices
> 		```sql
> 		WITH MinPrices AS (
> 		    SELECT Location, MIN(Price) AS MinPrice
> 		    FROM '/Users/dorawong/Downloads/Car Data.csv'
> 		    WHERE Year IN (2018, 2019, 2020)
> 		    GROUP BY Location
> 		)
> 		```
> 	2. （子查询2.2）用 2.1 的查到的数据 Location, MinPrice 反查原表
> 		```sql
> 		SELECT cars.Location, cars.Brand, cars.Year, cars.Price
> 		FROM '/Users/dorawong/Downloads/Car Data.csv' AS cars
> 		JOIN MinPrices
> 		ON cars.Location = MinPrices.Location AND cars.Price = MinPrices.MinPrice
> 		WHERE cars.Year IN (2018, 2019, 2020);
> 		```

^solution-1326
## Reference

