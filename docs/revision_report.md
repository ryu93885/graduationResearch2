# 修正レポート（検討・相談の記録）

- **対象**：研究の構成（[`docs/design/2026-11-18_research_structure.md`](design/2026-11-18_research_structure.md)）と、それにもとづくコード
- **作成日**：2026-11-19

このファイルには、チャットで出た指摘・相談・決定事項を日付ごとに記録する。新しい内容は**末尾に追記**する。

## 記録のルール

- 各エントリは `## YYYY-MM-DD 見出し` で始める。過去のエントリは書き換えず、状態が変わったときは新しいエントリに書く。
- 指摘事項には通し番号 `R-xx`、ユーザーへの確認事項には `Q-xx` を付ける。後のエントリでは番号で参照する。
- 状態は次のいずれかとする：`未決定`（提案のみ）／`採用`／`不採用`／`対応中`／`対応済`／`保留`／`対象外`
- **番号は graduationResearch からの続き番号とする。** この研究の指摘は R-28、確認事項は Q-18 から始まる。R-01〜R-27、Q-01〜Q-17 は graduationResearch（以前の設計）の番号で、このリポジトリでは使わない。graduationResearch の番号を参照するときは「graduationResearch の R-03」のように書く。

---

## 2026-11-19 研究の開始と構成の検討（R-28〜R-35、Q-18〜Q-23）

### 経緯

- 2026-11-18：ユーザーが研究の構成（docx）を作成した。graduationResearch のチャットで内容を把握し、指摘 R-28〜R-35 と確認事項 Q-18〜Q-23 を整理した。
- 2026-11-19：ユーザーが graduationResearch2（このリポジトリ）を作成した。graduationResearch とは**独立した研究**として進める。構成と、2026-11-18 の指摘・確認事項をこのリポジトリに移し、ユーザーの回答を受けて決定した。

### 実施したこと

- 作業ルール `CLAUDE.md` を作った。graduationResearch のルールを引き継ぎ、次の2つを加えた。
  - 研究の構成（`docs/design/`）を、コードのレビューの基準にする。
  - 目標値・評価指標・手法の設定は、参考にした論文の値を最優先し、次に一般的な値を使う（今回のユーザーの指示）。
- `.gitignore`（データ・学習済み重み・実験出力を除外）を graduationResearch から写した。
- 構成の原本（docx）と Markdown 版を `docs/design/2026-11-18_research_structure.{docx,md}` に置き、誤記とパラメータ数を直した（R-29）。
- 最終レポートの草案 `docs/final_report_draft.md` を作り、R-32・R-33・R-34・Q-19・Q-22 の決定を書き入れた。

### 構成の要約

| 段階 | 内容 | モデル・設定 | 目標の判定に使う値 |
|---|---|---|---|
| ⓪ | ルールベースで分類・検出する。深層学習の有用性を確かめる基準 | ルールの中身は未定（Q-20） | 未定 |
| ① | 3クラス分類（Lung Opacity / No Lung Opacity・Not Normal / Normal） | modelA：ResNet-101（ImageNet の IMAGENET1K_V2）、全層を学習、512×512、交差エントロピー、SGD | 検証用データのマクロ平均 ROC-AUC |
| ② | Lung Opacity の BBox 検出 | modelB：Faster R-CNN + ResNet-101-FPN、conv1 と layer1 は固定、512×512、SGD | 検証用データの mAP（IoU 0.5） |
| ③ | テストデータで Grad-CAM を出し、医師の BBox と整合するかを調べる。整合しない根拠は、ほかの画像と比べて確からしいかを調べる | ①②のモデル | R-33 の指標 |

### パラメータ数の検算

torchvision と同じ構造で、層ごとの式から数えた（この作業環境には PyTorch を入れられなかったため）。

| 項目 | 構成の値（修正前） | 検算の値 | 判定 |
|---|---|---|---|
| ResNet-101（fc を除く） | 4250160 | 42,500,160 | 誤記（0 が1つ抜けていた）→ 修正済み |
| 新しい fc（2048→3） | 6147 | 6,147 | 一致 |
| modelA の合計 | — | 42,506,307 | — |
| modelB（クラス数 2：背景＋Lung Opacity） | 60239065 | 60,239,065 | 一致 |
| modelB のうち学習する部分（conv1・layer1 を固定） | — | 60,016,665 | — |

- modelB の値は、torchvision の既定どおり BN を固定（`FrozenBatchNorm2d`。BN の重みをパラメータに数えない）にしたときの値である。BN を学習可能にすると 60,344,409、元の COCO 用 91 クラスのままなら 60,695,190 になる。
- 内訳：FPN 3,344,384、RPN 593,935、全結合層（TwoMLPHead）13,895,680、予測層 10,250。

### 指摘事項一覧

| 番号 | 区分 | 指摘 | 優先度 | 状態 |
|---|---|---|---|---|
| R-28 | 記述 | 新しい modelA/modelB と、graduationResearch の旧設計の Model A/B/C の名前が重なる | 中 | **対象外**（独立したリポジトリになり、旧設計の名前と混ざらないため） |
| R-29 | 記述 | 構成表の誤記（パラメータ数、予測層、全結合層、RoI など） | 低 | **対応済**（下記） |
| R-30 | 実装 | Faster R-CNN は内部で画像を 800 px に拡大し、正規化もする。BBox の座標の変換と陰性画像の扱いも決める | 高 | **保留**（modelB のプログラムを作るときに説明する。ユーザーの指示） |
| R-31 | 評価 | ⓪・modelA・modelB を比べる共通の軸（Lung Opacity か否かの ROC-AUC）を置く。modelB の閾値は val で決める | 中 | 未決定 |
| R-32 | 評価 | 目標値を決める。再学習は val だけで判断し、テストは最後に1回だけ使う | 高 | **採用・決定**（下記） |
| R-33 | 評価 | Grad-CAM と医師の BBox の比べ方を決める | 高 | **採用・決定**（下記） |
| R-34 | 評価 | 「医師と異なる根拠」は、ショートカットと付け忘れを先に疑って除いてから判断する | 高 | **採用**（最終レポート草案の 4.9 節に追記した） |
| R-35 | 実装 | データ拡張の記載がない。使う場合は BBox も同じ変換をし、左右反転は避ける | 低 | 未決定 |

### 指摘事項の詳細

#### R-29 構成表の誤記（対応済）

原本（docx）と Markdown 版の両方を直した。直した箇所は [`docs/design/2026-11-18_research_structure.md`](design/2026-11-18_research_structure.md) の「修正の記録」にある。主なものは次のとおりである。

- パラメータ数：4250160 → 42,500,160
- 予測層：「1024->2から2024->8」→「分類用は 1024->91 から 1024->2 に、BOX 回帰用は 1024->364 から 1024->8 に」
- 全結合層：12,544->1024->->1024->1024 → 12,544->1024->1024
- 用語：RolHead → RoI Heads、RolAlign → RoIAlign、画像検証 → 画像検出、交差円トロぴー → 交差エントロピー、制度 → 精度、継続学習 → 追加学習、「使用する機械学習モデル」→「使用する機械学習モデルと手法」
- 事前学習の重み（modelB）：「FPNとRPNは新規学習」→「FPN・RPN・RoI Headsは新規学習」。RoI Heads（全結合層と予測層）も ImageNet の重みを持たないためである。
- 「継続学習」は、保存した重みから学習を続ける意味と解釈して「追加学習」とした。意味が違う場合は戻す。

#### R-30 Faster R-CNN の実装上の注意（保留）

modelB のプログラムを作るときに、次の点を説明する（2026-11-18 の整理の要点）。

1. 入力の大きさ（内部で短辺 800 に拡大される。`min_size=512, max_size=512` を指定する）
2. 正規化（内部で行うので、データセット側ではしない）
3. BBox の座標（1024 px の (x, y, w, h) を 1/2 にして (x1, y1, x2, y2) に変える。Lung Opacity = 1）
4. 陰性画像（空の BBox として学習と評価に入れる）
5. バックボーンの作り方（キーワード引数、`trainable_layers=3`、BN は固定）
6. 学習率（画像1枚あたり 0.001 → バッチ 4 で 0.004、8 で 0.008）

#### R-32 目標値（決定）

ユーザーの指示により、「参考にした論文 → 一般的な値」の順に根拠を探して決めた。

**1. 参考にした論文**：構成の参考文献（He et al. 2016、Ren et al. 2015、Lin et al. 2017）と、データセット・Grad-CAM の原論文（Shih et al. 2019、Selvaraju et al. 2017）を確認した。He・Ren・Lin の性能は ImageNet・PASCAL VOC・COCO での値で、胸部X線とは課題が違う。Shih et al. (2019) はデータセットの論文で、モデルの性能を報告していない。Selvaraju et al. (2017) は説明手法の論文である。したがって、**参考にした論文からは目標値を決められない。**

**2. 一般的な値**：

| モデル | 根拠 | 決定した目標値 |
|---|---|---|
| modelA | AUC の評価の目安（Hosmer, Lemeshow & Sturdivant, *Applied Logistic Regression*）：0.7〜0.8 は acceptable、0.8〜0.9 は excellent、0.9 以上は outstanding。RSNA の3クラス分類のマクロ平均 AUC を報告した論文は、検索では見つからなかった | **検証用データのマクロ平均 ROC-AUC 0.80 以上**（excellent の下限） |
| modelB | RSNA での検出の報告値：改良した Faster R-CNN で mAP 39.23%（Yao et al. 2021）。FPAENet の論文の比較表では RetinaNet（ResNet-101）46.93%、EfficientDet（ResNet-50）46.38%、FPAENet 45.17〜49.51%。これらは改良した手法の値なので、改良していない Faster R-CNN は下の方の値になると考えた | **検証用データの mAP（IoU 0.5）0.40 以上** |

- 【要確認】modelB の根拠の論文は、この作業環境から本文にアクセスできず、検索結果の抜粋で値を確かめた。IoU の閾値や、陰性画像を評価に含めているかは本文で確認する。陰性画像を含めて評価すると、誤検出が増えて値が下がる。
- **学習のやり直しの規則**：目標値に届かないときは、構成の4章①②どおり、損失関数・最適化手法・学習率などを見直して学習をやり直す。この判断は検証用データ（val）の値だけで行う。テストデータは、モデルと閾値を固定したあとに1回だけ使う。

#### R-33 Grad-CAM と医師の BBox の比べ方（決定）

**1. 参考にした論文（Selvaraju et al. 2017、Grad-CAM の原論文）**から、次の2つを主な指標にする。

| 指標 | 定義 | 原論文での使い方 |
|---|---|---|
| 局在の正解率 | CAM を最大値の 15% で二値化し、最も大きい連結領域を囲む BOX を作る。この BOX と医師の BOX のいずれかの IoU が 0.5 以上なら「正解」 | ImageNet での弱教師あり局在（ILSVRC の局在誤差）の評価 |
| Pointing game の的中率 | CAM の最大値の位置が、医師の BOX のいずれかの中にあれば「的中」 | Grad-CAM で 70.58%（c-MWP は 60.30%） |

- **層**：原論文にならい、最後の畳み込み層（ResNet-101 の `layer4`）を使う。512 px 入力では 16×16 になるので、512 に拡大して比べる。
- **スコア**：原論文にならい、softmax をかける前のスコア（Lung Opacity クラスのロジット）の勾配を使う。

**2. 一般的な指標（Saporta et al. 2022、胸部X線での説明手法の比較）**から、次を補助の指標にする。

- **mIoU**：二値化した CAM と、医師の BOX を塗った領域（複数の BOX は和集合）の IoU を、画像で平均した値。二値化の閾値は、検証用データで mIoU が最大になる値を探して決める（Saporta et al. の方法）。
- Saporta et al. の hit rate は pointing game と同じ定義なので、上の的中率で代える。

**3. 判定と報告の方法**

- **画像ごとの判定**：局在の正解（IoU 0.5 以上）を満たす画像を、「判断根拠が医師のアノテーションと整合する」とみなす。
- **対象の画像**：医師の BOX があるのは Lung Opacity の画像だけなので、テストデータの Lung Opacity の画像で評価する。モデルが正しく判定した画像と、判定しなかった画像を分けて報告する。
- **偶然の水準**：pointing game の偶然の的中率は、BOX が画像に占める面積の割合に等しい。これを併せて報告し、的中率がこれを上回るかを見る。
- **ばらつき**：割合には、患者単位のブートストラップによる 95% 信頼区間を付ける。

#### R-34 「医師と異なる根拠」の検証の順番（採用）

最終レポート草案の 4.9 節に追記した。要点は次のとおりである。

1. 撮影条件（AP/PA）によるショートカットを、AP・PA に分けた評価で確かめる。
2. 写り込み（文字マーカー、チューブ類、電極など）を目視で確かめる。
3. BBox の付け忘れを、3人読影の画像で同じ傾向が出るかで確かめる。

これらで説明できない反応が、多くの画像で同じ解剖学的な位置に繰り返し出るときに、「確からしい根拠」の候補とする。最終的な判断には、指導教員や専門医の意見を聞く。

### 確認事項

| 番号 | 内容 | 状態 |
|---|---|---|
| Q-18 | 旧設計の「BBox を隠す実験」（旧 Model C など）を、新しい構成に置き換えるか | **回答済み（解釈）**：graduationResearch とは独立した研究として進めるため、この研究には旧設計の実験を含めない |
| Q-19 | データの分け方 | **回答済み**：まず1回の分割で行う。患者単位で train 70% / val 10% / test 20% に分ける |
| Q-20 | ⓪ ルールベースの中身：何の規則で分類し、どう BBox を出すか | 未回答 |
| Q-21 | modelB の事前学習の重み「modelAと同じ」は、ImageNet の重み（IMAGENET1K_V2）か、学習済みの modelA の重みか | 未回答 |
| Q-22 | Grad-CAM を modelB にも使うか。使うなら何の値をどの層で可視化するか | **回答済み**：「参考にした論文 → 一般的な値」の順で決める。決定は下記 |
| Q-23 | `docs/` と `COMMANDS.md` を graduationResearch の `archive/` から戻すか | **対象外**：新しいリポジトリで作り直す。`docs/setup.md`、`docs/procedure.md`、`COMMANDS.md` は、最初のプログラムを追加するときに作る |

#### Q-22 modelB の Grad-CAM（決定）

**1. 参考にした論文（Selvaraju et al. 2017）**：Grad-CAM は、構造を変えずにどの CNN にも使える手法で、「説明したい出力のスコア」の、最後の畳み込み層の特徴マップに対する勾配を使う。これを modelB に当てはめ、次のように決めた。

- **対象のスコア**：画像の中で確信度が最も高い検出 BOX について、その BOX の Lung Opacity の softmax 前のスコア（RoI Heads の分類の出力）。確信度の閾値（R-31）によらず、最も高い BOX を使う。
- **層**：バックボーンの ResNet-101 の `layer4`（modelA と同じ層）。modelA と同じ 16×16 の解像度で比べられる。FPN を通して、どの大きさの BOX からも `layer4` まで勾配が届く。
- **評価**：R-33 と同じ指標で、modelA と並べて報告する。

**2. 一般的な方法（pytorch-grad-cam。Gildenblat ほか）**：検出モデル向けの例では、検出 BOX と IoU で対応づけた出力のスコアを対象にし（`FasterRCNNBoxScoreTarget`）、勾配を使わない EigenCAM・AblationCAM を勧めている。

- 【案】Grad-CAM の結果が崩れる場合（画像全体がほぼ一様になるなど）に、比較として EigenCAM を使う。

### 現時点で未回答・保留の項目

| 番号 | 内容 | 状態 |
|---|---|---|
| R-30 | Faster R-CNN の実装上の注意 | 保留（modelB のプログラムを作るときに説明する） |
| R-31 | 共通の比較軸と modelB の閾値 | 未決定 |
| R-35 | データ拡張 | 未決定 |
| Q-20 | ⓪ ルールベースの中身 | 未回答 |
| Q-21 | modelB の事前学習の重み | 未回答 |

### 参照した資料

この作業環境からは論文の本文（arXiv、PubMed Central など）にアクセスできなかったため、検索結果の抜粋で値を確かめた。最終レポートに載せる前に原典で確認する。

- Selvaraju RR, et al. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." ICCV 2017.（arXiv:1610.02391）
- Saporta A, et al. "Benchmarking saliency methods for chest X-ray interpretation." *Nature Machine Intelligence* 4, 867–878 (2022).
- Hosmer DW, Lemeshow S, Sturdivant RX. *Applied Logistic Regression*, 3rd ed. Wiley, 2013.
- Yao S, et al. "Pneumonia Detection Using an Improved Algorithm Based on Faster R-CNN." 2021.（PMC8081632）【要確認：著者・掲載誌】
- "FPAENet: Pneumonia Detection Network Based on Feature Pyramid Attention Enhancement." arXiv:2011.08706, 2020.【要確認：著者・掲載先】
- Shih G, et al. "Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia." *Radiology: Artificial Intelligence* 1(1), e180041 (2019).
- Gildenblat J, et al. pytorch-grad-cam（GitHub）、検出モデル向けの例（Faster R-CNN）。

---

## 2026-11-19 教授の指摘への回答と、未解決の確認事項（Q-24〜Q-34）

### 教授の指摘（原文）とユーザーの回答

| 番号 | 教授の指摘（原文） | ユーザーの回答 |
|---|---|---|
| 1 | I am not a big fan of xAI (in my opinion there is lots of bad science being done under that label), but I can see that you are interested in it, go for it.. | 十分な研究がされていないからこそ魅力があり興味があるので、続けたいと思っている |
| 2 | I am not sure what a rules-based system means in the context of your application. If you mean some sort of classical computer vision algorithm, it will likely take lots of effort and yield little results, please explain with more detail.. | これから画像の特徴を分析し、アルゴリズムを設計し分類に使用する。（例えば、白いピクセルが多い画像は肺炎の画像にみられる特徴である、や、白いピクセルの密度が関係しているなど、基本的には、いくつかの特徴を組み合わせて簡易的なルールベースモデルを作成する） |
| 3 | Resnet 101 is a decent model but I would start with resenet50 which is faster and often gets similar results. I would evaluate the training based on accuracy first and then use the confusion matrix (more than precision and recall) to get a more detailed idea of where the model makes the mistakes. Here I am not sure what you mean by "retraining and continued training". You should train for a fixed number of epochs with preset patience, be careful of data leaks and properly define training/validataion/testing sets. For this type of problem probably 10 epochs are enough but we can try up to 50 initially (I would set patience as 5 and 10). We can have a look at possible failed training settings but in general I would not waste too much time on this, decide on a standard set and try a couple of alternative settings to make sure. I would only change the loss function if we want to bias the training towards better results for one particular class.. | 要検討 |
| 4 | Similar comments for the object detection part. Here I would start YOLO (easy to implement) or MaskRCNN (you already have code). Again, I am not sure ROC is going to help much. | ResNet と同じ構造で評価しやすい Faster R-CNN を使用する予定。ただし要検討 |
| 5 | Grad CAM may be interesting and is always good to learn but I think it is mostly indicated for the classification task, object detectors already output the boxes that tell you at what parts they are looking. Actually as most object detector networks start with a classifier, I am not sure if we will get anything new. I would drop initially drop grad cam as we are already getting the information of where the networks are looking with the bounding boxes. | （回答なし） |

### 解決した事項

- **XAI を続けるか（指摘1）**：続ける（ユーザーの回答）。教授の「XAI には質の低い研究が多い」という懸念には、評価の方法を結果を見る前に決めておくこと（R-33）、偶然の水準と比べること（R-33）、ショートカットを先に疑うこと（R-34）で応える。健全性チェックを加えるかは Q-34。
- **データのリークと分割（指摘3）**：Q-19 で、患者単位で train 70% / val 10% / test 20% に分けると決めている。⓪・modelA・modelB で同じ分割を使い、テストデータは最後に1回だけ使う（R-32）。
- **Q-20（⓪ の中身）**：一部回答済み。「白いピクセルの量や密度などの画像の特徴をいくつか組み合わせ、分類に使う簡易的なルールベースモデル」とする。残る点は Q-25〜Q-27。

### 指摘で状態が変わる既存の項目

| 番号 | 内容 | 状態 |
|---|---|---|
| Q-20 | ⓪ の中身 | 一部回答済み → 残りは Q-25〜Q-27 |
| Q-21 | modelB の事前学習の重み | 未回答（Q-28・Q-32 の回答とあわせて決める） |
| Q-22 | modelB の Grad-CAM | **再検討**（2026-11-19 の決定は、Q-24 の回答で取り消す可能性がある） |
| R-31 | 共通の比較軸（ROC-AUC） | **再検討**（指摘4で ROC の有用性に疑問。Q-33 で決める） |
| R-32 | 目標値と学習のやり直しの規則 | **再検討**（指摘3で「再学習・追加学習」の考え方を変える可能性がある。Q-29・Q-30 で決める） |
| R-35 | データ拡張 | 未決定のまま（modelA のプログラムを作るときに決める） |

### 確認事項（追加）

★は、最初のプログラム（データの分割と modelA）を書く前に決めたいもの。

| 番号 | 関係する指摘 | 内容 | 選択肢 | 推奨 |
|---|---|---|---|---|
| Q-24 ★ | 5 | Grad-CAM をどのモデルに使うか | (a) modelA だけに使う。modelB は、検出した BOX を「見ている場所」として医師の BOX と比べる／(b) 当面はどちらにも使わず、モデルができてから考える／(c) 両方に使う（2026-11-19 の Q-22 の決定のまま） | (a) |
| Q-25 | 2 | ⓪ で行う範囲 | (a) 2値分類（Lung Opacity か否か）だけ／(b) 3クラス分類／(c) 分類と BBox の検出 | (a) |
| Q-26 | 2 | ⓪ で特徴を計算する範囲 | (a) 画像全体（明るさを画像ごとにそろえてから）／(b) 画像の中央など、あらかじめ決めた範囲／(c) 肺野（肺野を切り出す処理が別に必要） | (a) から始める |
| Q-27 | 2 | ⓪ のルール（閾値と組み合わせ）の決め方と、かける期間 | (a) train の特徴の分布を見て、閾値を手で決める／(b) train から浅い決定木などで自動で決める。期間の上限も決める | (a)、上限 1 週間 |
| Q-28 ★ | 3 | バックボーンを ResNet-50 にするか | (a) modelA・modelB とも ResNet-50 にする／(b) ResNet-101 のまま／(c) modelA だけ ResNet-50 | (a) |
| Q-29 ★ | 3 | 学習の進め方を教授の方法にするか | (a) 教授の方法にする：最大 50 epoch、patience 5 と 10、「標準の設定」と「別の設定 2〜3 個」を学習して val で選ぶ。構成の「精度が担保されるまで再学習・追加学習」は削除する／(b) 今の構成のまま | (a) |
| Q-30 ★ | 3 | 学習の評価（epoch の打ち切りとモデルの選択）に使う指標と、目標値の扱い | (a) val の正解率で判断し、誤りの分析は混同行列で行う。マクロ平均 ROC-AUC は補助にする。目標値は「届くまで学習をやり直す基準」ではなく「結果を見る参考値」にする／(b) 今のまま（val のマクロ平均 ROC-AUC、目標値 0.80） | (a) |
| Q-31 | 3 | 損失関数 | (a) 標準の交差エントロピーのまま／(b) Lung Opacity を重視するよう重みを付ける | (a) |
| Q-32 ★ | 4 | 検出モデル | (a) 手元の Mask R-CNN のコードから、マスクを予測する部分を外して Faster R-CNN として使う／(b) Faster R-CNN を新しく書く／(c) YOLO | (a)（既存のコードの中身を見てから確定） |
| Q-33 | 4 | modelB の評価から ROC を外すか | (a) 外す。BOX 単位は適合率・再現率・F値・mAP（IoU 0.5）、画像単位は正解率と混同行列（陽性・陰性）にする。⓪・modelA・modelB の比較（R-31）も、画像単位の正解率と混同行列で行う／(b) ROC を残す | (a) |
| Q-34 | 1 | Grad-CAM の健全性チェックを加えるか | (a) 加える：モデルの重みをランダムにしたときに CAM が変わることを確かめる（Adebayo et al. 2018）／(b) 加えない | (a) |

### 確認事項の補足

- **Q-24**：教授の指摘のとおり、検出モデルは BOX そのものが「どこを見て判定したか」を示す。modelB の BOX と医師の BOX の一致は、BOX 単位の評価（IoU、mAP）でそのまま測れる。(a) の場合、modelB については「確信度が高いのに医師の BOX と重ならない検出」を「医師と異なる根拠」の候補とし、R-34 の手順で調べる。研究の主題（判断根拠の可視化）は modelA の Grad-CAM で扱う。
- **Q-25〜Q-27**：教授は「古典的な画像処理は手間のわりに成果が小さい」と指摘している。白いピクセルの量は、陰影だけでなく骨・心臓・チューブ類・文字や、撮影条件（AP/PA）による明るさの違いでも変わる。そのため、範囲を2値分類に絞り、期間の上限を決めて、「深層学習との比較の基準」という役割に限るのがよい。肺野の切り出し（Q-26 の (c)）は、それ自体が大きな作業になる。
- **Q-28**：パラメータ数は、modelA が 42,506,307 → 23,514,179、modelB（BN 固定・2クラス）が 60,239,065 → 41,299,161 になる（torchvision と同じ構造で計算。ResNet-50 の Faster R-CNN を 91 クラスで数えると 41,755,286 で、torchvision の公表値と一致した）。ResNet-50 なら、torchvision に COCO で学習済みの Faster R-CNN の重みがあり、Q-21 の選択肢が増える。
- **Q-29**：(a) の場合、「標準の設定」と「別の設定」の中身は、作業ルールに従い「参考にした論文 → 一般的な値」の順で決めて記録する。教授の方法では「目標に届くまで学習をやり直す」ことをしないので、R-32 の「学習のやり直しの規則」は「標準の設定と別の設定の中から val で選ぶ」に置き換わる。
- **Q-30**：クラスの割合は、Lung Opacity・Not Normal・Normal でおよそ 24%・43%・33% で（graduationResearch での確認。本研究のデータで再確認する）、大きな偏りはないので、正解率で判断しても大きな問題はない。(a) の場合、R-32 で決めた目標値（AUC 0.80、mAP 0.40）は参考値として残す。
- **Q-32**：RSNA には BBox しかなく、マスク（陰影の輪郭）の正解がない。そのため、Mask R-CNN のマスクを予測する部分は学習できない。torchvision の Mask R-CNN は、Faster R-CNN にマスクを予測する部分を加えたものなので、手元のコードからその部分を外せば Faster R-CNN として使える。これで「既存のコードを使う」（教授）と「Faster R-CNN を使う」（ユーザー）の両方を満たせる。確定には、手元のコードを添付してもらう必要がある。
- **Q-21（継続）**：Q-28 と Q-32 の回答のあとに決める。意図がはっきりしない場合は、作業ルールに従い、参考にした論文（Ren et al. 2015、Lin et al. 2017）と同じ ImageNet の重みにする。

### 現時点で未回答・保留の項目

| 番号 | 内容 | 状態 |
|---|---|---|
| Q-21 | modelB の事前学習の重み | 未回答 |
| Q-24〜Q-34 | 上記 | 未回答 |
| R-30 | Faster R-CNN の実装上の注意 | 保留（modelB のプログラムを作るときに説明する） |
| R-35 | データ拡張 | 未決定 |
