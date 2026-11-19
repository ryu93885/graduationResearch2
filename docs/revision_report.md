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
