# amcs2022

以下の3つはレーダー雨量,土地利用,水位など,異なる複数のデータを用いて機械学習モデルを構築するためのコード.
バッチサイズによっては,データ容量が大きくメモリ不足になるため,学科サーバーにsingularity(コンテナ技術)で環境構築して実行.
create_train.py,create_test.py: training,testデータ作成用のコード.
dense_model.py: モデル作成のためのコード.pytorchを利用.

以下は分布型流出モデルを構築し,計算するためのコード.
流出現象は偏微分方程式で記述されるため,空間方向はFEM(有限要素法)でメッシュ間の水のやり取りに近似し,
時間方向は5次の適応時間ルンゲクッタ法で近似.pandas等を用いると実行時間が大きくなるので,
listを多用し,jit(just in time)コンパイラを用いて高速化を実現.(cythonでも可能)
dist_model.py: 流出計算のコード.