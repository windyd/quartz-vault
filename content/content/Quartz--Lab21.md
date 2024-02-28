


## [[01-12-23]]
今天主要做 embedding 分析

打算使用的 code
`D:\windyd\Projects\lab-32\LAB-22\embedding_LLM\`

- 10:12 started embedding
	```log
	kevin in 🌐 2GPU3090Ti in lab-32/LAB-22/embedding_LLM on  develop [!?] via 🐍 v3.8.13 via 🅒 LLM
	❯ python test_langchain_retriver.py --model text2vec-large-chinese --data_path /home/kevin/Data/Hue/ymd=2023-04-13/h=00/part-00000-9bac1344-0b02-4c2c-9226-a26e5f796142-c000.gz.parquet
	...
	...
	...
	...
	2023-12-01 02:10:47,827 - langchain_retrieve - INFO - Getting or creating vec store...
	2023-12-01 02:10:47,827 - langchain_retrieve - INFO - No existing faiss store found, building faiss store...
	2023-12-01 02:10:47,827 - langchain_retrieve - INFO - documents_df has size 211145
	Batches:   0%|▏                                             | 22/6599 [00:49<3:24:08,  1.86s/it]
	```
	
	211,145 的文档要 embed 3.5 小时

- 11:29 想到有必要做一个类似 [zejunwang1/CSTS: 中文自然语言推理与语义相似度数据集 (github.com)](https://github.com/zejunwang1/CSTS#chinese-sts-b-%E6%95%B0%E6%8D%AE%E9%9B%86) 的数据集
	- [中文Sentence Embeddings text2vec-base-chinese VS OpenAIEmbedding - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/623912895)
- 13:35 在 12:50:47 秒完成，共耗时 2:39:39 比预期要快
	```log
	2023-12-01 02:10:47,827 - langchain_retrieve - INFO - documents_df has size 211145
	Batches: 100%|████████████████████████████████████████████| 6599/6599 [2:39:39<00:00,  1.45s/it]
	2023-12-01 04:50:36,171 - faiss.loader - INFO - Loading faiss with AVX2 support.
	2023-12-01 04:50:36,219 - faiss.loader - INFO - Successfully loaded faiss with AVX2 support.
	2023-12-01 04:50:47,031 - langchain_retrieve - DEBUG - --> Building FAISS store <--: start_time 02:10:47 elapsed_time 9599.204355239868 s
	```
	
	211,145 封邮件 从 143M (full) 变成了 825M （提取 subject + content + attach）
	```log
	kevin in 🌐 2GPU3090Ti in LAB-22/embedding_LLM/text2vec-large-chinese_faiss_store on  develop [!?] via 🅒 LLM
	❯ ll -h
	total 941M
	-rw-rw-r-- 1 kevin kevin 825M Dec  1 04:50 index.faiss
	-rw-rw-r-- 1 kevin kevin 116M Dec  1 04:50 index.pkl
	```
	
	```log
	>>> import pandas as pd
	>>> df = pd.read_parquet('/home/kevin/Data/Hue/ymd=2023-04-13/h=00/part-00000-9bac1344-0b02-4c2c-9226-a26e5f796142-c000.gz.parquet', columns=['subject', 'content', 'attach'])
	>>> df.info()
	<class 'pandas.core.frame.DataFrame'>
	RangeIndex: 211145 entries, 0 to 211144
	Data columns (total 3 columns):
	 #   Column   Non-Null Count   Dtype
	---  ------   --------------   -----
	 0   subject  211145 non-null  object
	 1   content  211145 non-null  object
	 2   attach   211145 non-null  object
	dtypes: object(3)
	memory usage: 4.8+ MB
	>>> df.to_parquet('checkout_size.parquet')
	>>> exit()
	
	kevin in 🌐 2GPU3090Ti in lab-32/LAB-22/embedding_LLM on  develop [!?] via 🐍 v3.8.13 via 🅒 LLM
	❯ ll -h checkout_size.parquet
	-rw-rw-r-- 1 kevin kevin 46M Dec  1 05:50 checkout_size.parquet
	```
	如果只看 subject + content + attach 的话，原本的数据只占内存 4.8+MB
	pandas 写到硬盘后占 46M (不知道 pandas 的处理有没有膨胀)

- 15:40 Read code `langchain` does this internally
	```python
	# save index separately since it is not picklable
	faiss = dependable_faiss_import()
	faiss.write_index(
		self.index, str(path / "{index_name}.faiss".format(index_name=index_name))
	)
	
	# save docstore and index_to_docstore_id
	with open(path / "{index_name}.pkl".format(index_name=index_name), "wb") as f:
		pickle.dump((self.docstore, self.index_to_docstore_id), f)
	```
	`self.index_to_docstore_id` is the mapping to docstore:
	`Dict[int, str]`
	
	```log
	0: 5b1d34fc-b406-4406-ab50-0436ebb98a82
	1: 23221a7c-1f63-4633-b74b-fb53614aae11
	2: 5915bf29-ec3f-4a98-b1b8-6373e039054a
	3: e520c1f9-0aa7-4026-b033-2f7cdec9e4de
	4: da8105c1-164a-4645-b9d5-e00211ab81d9
	5: 11f00be6-ffb1-4bb3-82d5-61d33cb6591b
	6: 8d2d2804-7fea-4f80-9c67-b9c46729185a
	7: 4f931f78-0d5e-4486-a5b0-e4a7e1900fb5
	8: 3b904e22-1f8a-41f1-87f6-0674007ecfbc
	9: 41cd72a3-efb1-4f46-9377-5dd80125cd61
	10: f91a01ed-9acc-4ec4-85c2-0485cb5ca4dc
	11: fbe4dc78-e2e6-401d-840d-172d01638a3d
	```
	how do we access the doc's index using this id?
	```
	docstore.search(index_to_docstore[i])
	```
	
	I don't understand why `self.index_to_docstore` has to be such a mapping
- 16:46 `Docstore` is a component of `vecstore`
	```python
	vecstore = cls(
		embedding,
		index,
		InMemoryDocstore(),
		{},
		normalize_L2=normalize_L2,
		distance_strategy=distance_strategy,
	)
	```
	
- > I don't understand why `self.index_to_docstore` has to be such a mapping
	
	因为要支持增删查改, 顺序的数字不方便操作，用 uuid 比较方便 
- 16:59 *可见 int number 是用来访问向量的，uuid 是用来访问 docs 的*。要 access 向量，simply use 第 i 个 index 就好了 by looking at the delete function
	```python
	def delete(self, ids, **kwargs):
		...
		## Reversed mapping: ids --> i-th item
		reversed_index = {id_: idx for idx, id_ in self.index_to_docstore_id.items()}
		index_to_delete = [reversed_index[id_] for id_ in ids]
		## delete by i-th item
		self.index.remove_ids(np.array(index_to_delete, dtype=np.int64))
		self.docstore.delete(ids)
		
		remaining_ids = [
			id_
			for i, id_ in sorted(self.index_to_docstore_id.items())
			if i not in index_to_delete
		]
		self.index_to_docstore_id = {i: id_ for i, id_ in enumerate(remaining_ids)}
	
		return True
	```


## [[04-12-23]]
MAIN work files:
- Try_Phoenix.ipynb
- faiss_try.py
- test_langchain_retriever.py
- [I] 可以用编辑距离之类的快速去掉重复的邮件？
 - [I] 直接 把 paquet -> embedded parquet 的 function 之后再写到 `test_langchain_retriver.py` 之中
 
- 10:48 implement langchain `faiss_store` to parquet
- 11:19 弄成 data.parquet 看起来更省空间
	```log
	kevin in 🌐 2GPU3090Ti in lab-32/LAB-22/embedding_LLM on  develop [!?] via 🐍 v3.8.13 via 🅒 LLM
	❯ ll -h text2vec-large-chinese_faiss_store/
	total 941M
	-rw-rw-r-- 1 kevin kevin 825M Dec  1 04:50 index.faiss
	-rw-rw-r-- 1 kevin kevin 116M Dec  1 04:50 index.pkl
	
	kevin in 🌐 2GPU3090Ti in lab-32/LAB-22/embedding_LLM on  develop [!?] via 🐍 v3.8.13 via 🅒 LLM
	❯ ll -h data.parquet
	-rw-rw-r-- 1 kevin kevin 880M Dec  4 03:10 data.parquet
	```
- 11:22 准备测试 phoenix
- 12:28 暂时无法显示文字，想办法 get it to work
- 13:39 done above, just need to pass `raw_data_column_name` to schema
- 13:56 export in phoenix --> output a parquet of *selected data*
- 14:10 HDBSCAN: 当采用较大的 `min_cluster_size` 时，会明显的把中英文分出来
	- [!] 注意到 attach 因为通常英文、乱七八糟的东西比较多，有可能会影响结果（什么结果？）
- 14:21 数据要重新拿一下, 只拿中文的试试
- 15:09 直接 把 paquet -> embedded parquet 的 function 之后再写到 `test_langchain_retriver.py` 之中
- 15:15 正在 embed `/home/kevin/Data/Hue/ymd=2023-04-13/h=16/part-00003-e094963c-fecf-4aa5-887d-2c59a5fa2628-c000.gz.parquet` --> aborted
- 15:22 2 things
	1. add correct hf calling according to github
	2. make correct preprocessing
- 16:45 concerns: 降维比较伤 info
	- [?] 目前的 exact process 是怎样的？
	- SOM 降维、autoencoder 降维
- 17:14 重新从 langchain faiss store 提取 parquet
	- 17:33 先重新 embed (因为之前的 filter columns 不包含想要的 metadata)
		```log
		2023-12-04 09:32:22,906 - sentence_transformers.SentenceTransformer - INFO - Use pytorch device: cuda
		2023-12-04 09:32:22,906 - langchain_retrieve - INFO - Getting or creating vec store...
		2023-12-04 09:32:22,906 - langchain_retrieve - INFO - No existing faiss store found, building faiss store...
		2023-12-04 09:32:22,906 - langchain_retrieve - INFO - documents_df has size 408703
		Batches:   2%|▋                                           | 195/12772 [05:06<4:46:11,  1.37s/it]
		```
- 17:34 看看 [[UMAP--Uniform Manifold Approximation and Projection for Dimension Reduction.pdf]]

## [[05-12-23]]
1. embed`test_langchain_retrieve.py` -> text2vec-large-chinese_faiss_store
1. transform_parquet(`faiss_try.py`) -> data.parquet
- 09:49 connection issue? jupyter to 3090Ti seems laggy
- 10:40 找到一组不错的参数
![[LAB21 2023-12-05 10.51.03.excalidraw]]
- 12:53 有可能 phoenix 只能处理小部分数据
	- 怎么用在大流量上？
	- 方案一：小范围聚好类
		- option a：dist to centroids 来推断
		- option b：密度可达推断
	- 方案二：相同的参数放到 spark 跑
- 12:59 faiss -> k-means res 
- 15:07 [Benchmarking Performance and Scaling of Python Clustering Algorithms — hdbscan 0.8.1 documentation](https://hdbscan.readthedocs.io/en/latest/performance_and_scalability.html) some proof that k-means seems garbage
- 15:26 Decided to test HDBSCAN myself
- 15:38 自己直接进行 hdbscan 得不到好的结果（7/10）都是异常点，参考一下 Phoenix
	```python
	vectors, clustered_events = PointCloud(
		dimensionalityReducer=Umap(n_neighbors=n_neighbors, min_dist=min_dist),
		clustersFinder=Hdbscan(
			min_cluster_size=min_cluster_size,
			min_samples=cluster_min_samples,
			cluster_selection_epsilon=cluster_selection_epsilon,
		),
	).generate(data, n_components=n_components)
	```
- 15:41 一目了然了，确实是先降了维
	```python
	@dataclass(frozen=True)
	class PointCloud:
	    dimensionalityReducer: DimensionalityReducer
	    clustersFinder: ClustersFinder
	
	    def generate(
	        self,
	        data: Mapping[ID, Vector],
	        n_components: int = 3,
	    ) -> Tuple[Dict[ID, Vector], Dict[str, Set[ID]]]:
	        """
	        Given a set of vectors, projects them onto lower dimensions, and
	        finds clusters among the projections.
	
	        Parameters
	        ----------
	        data : mapping
	            Mapping of input vectors by their EventIds.
	
	        n_components: int, default=3
	            Number of dimensions in the projected space.
	
	        Returns
	        -------
	        projections : dictionary
	            Projected vectors in the low dimensional space, mapped back to the
	            input vectors' EventIds.
	
	        cluster_membership: dictionary
	            Cluster membership by way of cluster_ids in the form of integers
	            0,1,2,... mapped back to the input vectors' EventIds. Note that
	            some vectors may not belong to any cluster and are excluded here.
	
	        """
	
	        if not data:
	            return {}, {}
	        event_ids, vectors = zip(*data.items())
	        projections = self.dimensionalityReducer.project(
	            np.stack(vectors), n_components=n_components
	        )
	        clusters = self.clustersFinder.find_clusters(projections)
	        return dict(zip(event_ids, projections)), {
	            str(i): {event_ids[row_index] for row_index in cluster}
	            for i, cluster in enumerate(clusters)
	        }
	```
- 15:56 projector(`Umap`) is a wrapper of `umap.UMAP` https://github.com/Arize-ai/phoenix/blob/1562309da191524761194c2d73299e048194ad25/src/phoenix/pointcloud/projectors.py#L23C9-L23C9
	```python
	import warnings
	from dataclasses import asdict, dataclass
	from typing import cast
	
	import numpy as np
	import numpy.typing as npt
	from typing_extensions import TypeAlias
	
	with warnings.catch_warnings():
	    from numba.core.errors import NumbaWarning
	
	    warnings.simplefilter("ignore", category=NumbaWarning)
	    from umap import UMAP
	
	Matrix: TypeAlias = npt.NDArray[np.float64]
	
	
	def _center(arr: Matrix) -> Matrix:
	    return cast(Matrix, arr - np.mean(arr, axis=0))
	
	
	@dataclass(frozen=True)
	class Umap:
	    n_neighbors: int = 15
	    min_dist: float = 0.1
	
	    def project(self, mat: Matrix, n_components: int) -> Matrix:
	        config = asdict(self)
	        config["n_components"] = n_components
	        if len(mat) <= n_components:
	            # init='spectral', the default, cannot be used when n_components
	            # is greater or equal to the number of samples.
	            # see https://github.com/lmcinnes/umap/issues/201#issuecomment-462097103
	            config["init"] = "random"
	        return _center(UMAP(**config).fit_transform(mat))
	```
- 17:15 才注意到这次的数据居然有 40wan
- 17:51 待参考 --> [How To Tune HDBSCAN | by Charles Frenzel | Towards Data Science](https://towardsdatascience.com/tuning-with-hdbscan-149865ac2970)
	- > "**Kmeans** assume that data is numerical and sphere-shaped. Those types of assumptions do not fair well when the data has high dimensionality and includes categorical values."
	- Silhouette Score **does not** work for density-based algos 
		1. not considering noise in the index calculation
		2. makde use of distances
	- DBCV(Density Based Clustering Validation) [[DBCV]]
	 $$
	\text{DBCV}(C)=\sum_{i=1}^{i=l} \frac{|C_{i}|}{|O|}V_{C}(C_{i})
	$$
	- Clustering Solution: $C=\{C_{i}\}, i \le i \le l$ 
	- **Intuition**: weighted average of the Validity Index of all clusters in $C$
- Further Reading: [[What are the true clusters--2015arxiv.pdf]]
## [[06-12-23]]
- 10:59 [[DBCV]]
- 10:59 continue reading [How To Tune HDBSCAN | by Charles Frenzel | Towards Data Science](https://towardsdatascience.com/tuning-with-hdbscan-149865ac2970)
- 11:34 figured out how to use DBCV (implemented by `hdbscan` lib)
- 12:23 Come up with a task --> 写一个 循环来 对比不同参数的
	- 耗时
	- DBCV
		- weighted average
		- per cluster distribution -> make bins
	- anomaly ratio
	- tree plot
- 13:14 hdbscan papers
	- [[hdbscan.pdf]]
	- [[Accelerated Hierarchical Density Based Clustering.pdf]]
- 13:22 [[HDBSCAN]]
- 14:56 done reading [[HDBSCAN]] for now
- 14:56 checkout the $\lambda$ of our data
	- 有的十分的大,400 多, 打算 rescale data 看看是否会影响
- 16:00 cluster_utils (plots, metric cals)
- 16:44 准备 wrap `project_and_cluster` into 一个方便调用 sklearn
- 17:56 改 `sklearn.model_selection.StratifiedKFold` 让它返回 K 次相同原封不动的数据集
- 18:00 run grid
	```python
	param_grid = {
	    'min_samples': [1, 2, 4, 8, 16],
	    'min_cluster_size': [10, 20, 40, 80],
	    'cluster_selection_method' : ['eom'],
	}
	```
- 19:20 有点问题 用了 projection 来 grid search
	- 不确定 Umap 在input 的维度为 3 时，设置 `n_components=3` 的行为， nonetheless, 有一组参数上了 0.4, ( as an empirical referrence 0.48 为不错的数字)
		```log
		DBCV score :0.4377849103136721
		Best Parameters {'cluster_selection_method': 'eom', 'min_cluster_size': 10, 'min_samples': 1}
		```
## [[07-12-23]]
- 09:15 Paramsearch 应如下
	- 耗时(粗筛指标--> 作为参考)
	- DBCV
		- weighted average (粗筛指标-->按此选best)
	- anomaly ratio (粗筛指标 --> 作为参考)
- 09:18 精筛应如下
	- 耗时
	- DBCV
		- weighted average
		- per cluster distribution -> make bins
	- anomaly ratio
	- tree plots
	- 簇密度 distribution
- 09:56 improve estimator class
	- [ ] predict 的时候带上训练的一起算 score
- 11:31 看、讨论了一堆 retvec 和 模型蒸馏的东西
	- [BERT蒸馏完全指南｜原理/技巧/代码 (qq.com)](https://mp.weixin.qq.com/s/tKfHq49heakvjM0EVQPgHw)
- 13:40 code read [[Sklearn Model Selection]]
	- 15:57 了解到调用怎么走到 score 的 [[Sklearn Model Selection#最终的 score 函数]]
- 15:57 what's `min_sample` in HDBSCAN?
- 16:05 需要看看 sklearn 原生 predictor 的 score 是怎么实现的
	- 16:07 [sklearn.cluster.HDBSCAN — scikit-learn 1.3.2 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html#sklearn.cluster.HDBSCAN) 直接抄一抄这里的 score
	- 16:10 这个 class 没有 score
	- 16:16 Kmeans 的 score
		```python
		def score(self, X, y=None, sample_weight=None):
			"""Opposite of the value of X on the K-means objective.
		
			Parameters
			----------
			X : {array-like, sparse matrix} of shape (n_samples, n_features)
				New data.
		
			y : Ignored
				Not used, present here for API consistency by convention.
		
			sample_weight : array-like of shape (n_samples,), default=None
				The weights for each observation in X. If None, all observations
				are assigned equal weight.
		
			Returns
			-------
			score : float
				Opposite of the value of X on the K-means objective.
			"""
			check_is_fitted(self)
		
			X = self._check_test_data(X)
			sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
		
			_, scores = _labels_inertia_threadpool_limit(
				X, sample_weight, self.cluster_centers_, self._n_threads
			)
			return -scores
		```
- 17:26 两个东西怎么不一样？
	- ![[Pasted image 20231207172700.png]]
	- see [API Reference — hdbscan 0.8.1 documentation](https://hdbscan.readthedocs.io/en/latest/api.html) or below

> [`relative_validity_`](https://hdbscan.readthedocs.io/en/latest/api.html#id92): float
> 
> A fast approximation of the Density Based Cluster Validity (DBCV) score [4]. The only differece, and the speed, comes from the fact that this [relative_validity_](https://hdbscan.readthedocs.io/en/latest/api.html#id94) is computed using the mutual- reachability minimum spanning tree, i.e. [minimum_spanning_tree_](https://hdbscan.readthedocs.io/en/latest/api.html#id96), instead of the all-points minimum spanning tree used in the reference. This score might not be an objective measure of the goodness of clusterering. It may only be used to compare results across different choices of hyper-parameters, therefore is only a relative score.

## [[08-12-23]]
- [I] 思路：3d 上调好 umap 参数，在进行聚类

**问题分析**：
- 两大难题：
	- 高维度空间的问题 (regardless of 算法)
	- 语义局部结构问题：语义可能不仅是个球, 同样类别的语义可能在一个超平面上 

- 解决高维度问题 -> 降维
	- PCA -> 用线性（旋转、伸缩、投影）的方法重新找到一组坐标轴，把信息量（方差）低的坐标轴去掉
	- [[My Work/Concepts/t-SNE]]
		- 建立 高维数据分布（Gassuain）、低维数据分布 (learnable)
		- 用 KL 散度拉近两个分布的距离
		- [i] the Gaussian teacher maybe not good enough
	- [[My Work/Concepts/UMAP]]
		- 建立 高维数据分布（某种曲面上的分布）、低维数据分布 (learnable)
		- 用 CE 拉近两个分布的距离
		- [?] Umap 能够多大程度的学到局部结构？
	- Autoencoder
		- mlp 降维 mlp 升维
		- minimize 重构误差
		- [!] 高能耗
		- [!] 泛化性难以保证 （不是直接在概率空间上建模）--> 可用性变窄，为了降维这个任务要多次调整这个模型

- 解决局部结构问题
	- 连通图能被 HDBSCAN capture 到

**当前发现**：
- 降维十分必要 (从质量的角度)
- HDBSCAN 的调参十分容易 （相对于 UMAP）
- 聚类的效果 largely depends on UMAP

**策略**：
生成过程
- 人工精调 umap
- 机器暴力搜索 HDBSCAN 超参 -> 输出聚类

评估过程
- 对输出的聚类进行总结（LLM or 主题模型？ )
- 输出聚类/主题数量

LOG
- 14:57 调参 UMAP on 10,000 sample points
	- 招聘信息：
		![[Pasted image 20231208145934--quartz.png|214]]



## UMAP and HDBSCAN Tuning
> started:: [[05-12-23]]


由于自由度没那么高，暂时只用 umap + HDBSCAN 来做 instead of the wrapped phoenix

`min_distance`: `[0,1)`

- [!] 注意：当调整 umap 的时候，hdbscan 的数量也会变
- [x] 看看 phoenix 的实现 
	- 确实是先降到 3 dim，再聚类

top3 bottom3 mid`n` 

### Base param set

> [!NOTE] `n_sample` 10000 下的还行的参数, 就是有点挤(UMAP)
> clusters: "D:\windyd\Projects\lab-32\LAB-22\experiments\tune-HDBSCAN\base_param"
> ![[Pasted image 20231205112241.png|475]]
> INPUT
> - HDBSCAN
> 	- `min_cluster_size`: 40
> 	- `cluster_min_sample`: 10
> 	- `cluster_selection_epsilon`: 0
> - UMAP
> 	- `min_distance`: 0
> 	- `n_neighbors`: 100 (max already)
> 	- `n_sample`: 10000
> 
> OUTPUT
> - num_cluster: 276 (可能会变)
> - normal points: 7160


#### Tune Min Size

##### min_cluster_size 80
> [!NOTE] min_cluster_size 80
> ![[Pasted image 20231205135751.png|475]]
> INPUT
> - HDBSCAN
> 	- `min_cluster_size`: 80
> 	- `cluster_min_sample`: 10
> 	- `cluster_selection_epsilon`: 0
> - UMAP
> 	- `min_distance`: 0
> 	- `n_neighbors`: 100 (max already)
> 	- `n_sample`: 10000
> 
> OUTPUT
> - num_cluster: 5 (可能会变)
> - normal points: 9660





### Deprecated due to random seed
#### Base param set(deprecated)

> [!NOTE] `n_sample` 10000 下的还行的参数, 就是有点挤(UMAP)
> clusters: "D:\windyd\Projects\lab-32\LAB-22\experiments\tune-HDBSCAN\base_param"
> ![[Pasted image 20231205112241.png|475]]
> INPUT
> - HDBSCAN
> 	- `min_cluster_size`: 40
> 	- `cluster_min_sample`: 10
> 	- `cluster_selection_epsilon`: 0
> - UMAP
> 	- `min_distance`: 0
> 	- `n_neighbors`: 100 (max already)
> 	- `n_sample`: 10000
> 
> OUTPUT
> - num_cluster: 40 (可能会变)
> - normal points: 6442


#### Tune Min Size

##### min_cluster_size 80(deprecated)
> [!NOTE] min_cluster_size 80
> ![[Pasted image 20231205135751.png|475]]
> INPUT
> - HDBSCAN
> 	- `min_cluster_size`: 80
> 	- `cluster_min_sample`: 10
> 	- `cluster_selection_epsilon`: 0
> - UMAP
> 	- `min_distance`: 0
> 	- `n_neighbors`: 100 (max already)
> 	- `n_sample`: 10000
> 
> OUTPUT
> - num_cluster: 21 (可能会变)
> - normal points: 5935



