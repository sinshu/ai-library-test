# Iris GMM comparison: NumFlat vs AiDotNet

このリポジトリは、Irisデータセットに対して以下の2つの.NETライブラリで3成分のfull-covariance Gaussian Mixture Model (GMM) をフィットし、推定パラメータの一致度を確認する小さな検証プログラムです。

- [NumFlat](https://www.nuget.org/packages/NumFlat)
- [AiDotNet](https://www.nuget.org/packages/AiDotNet)

## 実装内容

`Program.cs` では次の流れで検証しています。

1. Iris CSVを `https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv` から取得する。
2. sepal length / sepal width / petal length / petal width の4特徴量を `double[][]` として読み込む。
3. NumFlatの `Clustering.ToGmm` で3成分GMMをフィットする。
4. AiDotNetの `GaussianMixtureModel<double>` で3成分GMMをフィットする。
5. 各成分の重み、平均、共分散を抽出する。
6. 平均ベクトル同士の距離が最小になるように成分を対応付ける。
7. 以下の指標でパラメータの一致度を出力する。
   - Mean RMSE
   - Covariance RMSE
   - Weight MAE

## 実行方法

```bash
dotnet run
```

## 今回の実行結果

今回の環境では、NumFlatとAiDotNetの両方でGMMのフィッティングに成功しました。

代表的な出力は以下です。

```text
Best component matching (NumFlat -> AiDotNet): 0->0, 1->1, 2->2
Mean RMSE:       0.000007
Covariance RMSE: 0.000002
Weight MAE:      0.000005
```

この結果から、今回の設定・データでは両ライブラリのGMM推定パラメータは非常によく一致しています。

## AiDotNet実行時の注意

AiDotNet実行時に、この環境ではOpenCLランタイムが見つからない旨の診断ログが出ました。

ただし、今回の実行ではその診断ログの後も処理は継続し、AiDotNetのGMMフィッティングとパラメータ比較は最後まで完了しました。

## 依存パッケージ

このプロジェクトでは以下のNuGetパッケージを使用しています。

- `NumFlat` 1.2.4
- `AiDotNet` 0.192.0
