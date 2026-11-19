# graduationResearch2

胸部X線画像（RSNA Pneumonia Detection Challenge 2018）で、肺炎の所見（Lung Opacity）を判定する分類モデル（ResNet-101）と検出モデル（Faster R-CNN）を作り、Grad-CAM で可視化した判断根拠が医師のアノテーション（BBox）と整合するかを検証する卒業研究のリポジトリ。

graduationResearch とは独立した研究として進める（2026-11-19）。

## 文書

| ファイル | 内容 |
|---|---|
| [`docs/design/2026-11-18_research_structure.md`](docs/design/2026-11-18_research_structure.md) | 研究の構成（原本は同じフォルダの docx）。コードのレビューの基準 |
| [`docs/revision_report.md`](docs/revision_report.md) | チャットでの指摘・相談・決定事項の記録（追記形式） |
| [`docs/final_report_draft.md`](docs/final_report_draft.md) | 最終レポートの草案（上書き形式） |
| [`docs/reports/`](docs/reports/) | 指導教員に提出する英語の進捗報告書（docx と Markdown 版） |
| [`CLAUDE.md`](CLAUDE.md) | このリポジトリでの作業ルール |

実行環境の手順（`docs/setup.md`）、実行手順（`docs/procedure.md`）、実行コマンドの一覧（`COMMANDS.md`）は、最初のプログラムを追加するときに作る。

## データについて

画像データ・アノテーションファイル・学習済み重みはコミットしない（`.gitignore` 参照）。
