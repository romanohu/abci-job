# ABCI Job Submitter

ABCIのログインノードで、単一ノード用のPBSジョブスクリプトを生成・投入するツールです。
通常キューと予約キューに対応し、実行中のログを共有ストレージから確認できます。
Python 3.11以上が必要です。ファイル転送やSSH接続、ノードの予約取得は行いません。

編集する設定は次の2ファイルです。

| ファイル | 編集するタイミング | 内容 |
| --- | --- | --- |
| `configs/environment.toml` | 初回・環境変更時 | グループ、作業ディレクトリ、環境準備、監視 |
| `configs/job.toml` | 資源・既定コマンドの変更時 | ジョブ名、キュー、実行時間、省略可能なコマンド |

ジョブの生成・投入には `submit.py` を使います。テンプレートやPythonコードを編集する必要はありません。

## 初回の準備

ABCIのログインノードでリポジトリをクローンし、そのディレクトリで実行します。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp configs/environment_example.toml configs/environment.toml
cp configs/job_example.toml configs/job.toml
```

コピーした2ファイルはGitの管理対象外です。
`environment.toml` の `group` と `workdir` を自分の環境に合わせて編集します。
`workdir` は、ログインノードと計算ノードからアクセスできる共有ストレージ上の絶対パスを指定してください。
コマンドはこのディレクトリで実行され、ログもその配下に保存されます。

学習に使うPython環境やCUDAなどは、`setup_commands` で準備します。
ここで指定する環境は、投入ツール自身を動かす `.venv` とは別でも構いません。

```toml
group = "your-abci-group"
workdir = "/groups/your-abci-group/your-user/your-project"
setup_commands = [
  "source /etc/profile.d/modules.sh",
  "source .venv/bin/activate",
]

[monitor]
enabled = true
interval_seconds = 600
commands = ["nvidia-smi"]
```

## ジョブの設定と投入

`configs/job.toml` に実行条件を記述します。`command` を設定しておけば、毎回のコマンド指定を省略できます。

```toml
name = "example-job"
queue = "rt_HG"
walltime = "01:00:00"
command = ["python", "train.py", "seed=1"]
```

スクリプト生成のみが既定の動作です。`--submit` を付けると、生成後に `qsub` で投入します。

```bash
# 設定ファイルの command で生成・表示
python submit.py --print-script

# 設定ファイルの command で生成・投入
python submit.py --submit

# 今回だけコマンド全体を置き換えて生成・投入
python submit.py --submit -- python train.py seed=2 batch_size=32768
```

`--` 以降は実行ファイルと引数としてそのまま渡します。設定の `command` に追記するのではなく、
コマンド全体を置き換えます。毎回CLIで指定する場合、設定ファイルの `command` は省略できます。
どちらにも指定がなければエラーになります。

投入ツールはHydraを使いません。実験リポジトリがHydraを使う場合も、通常の引数を使う場合も、
引数の解釈は実行対象に任せます。実験の設定は実験リポジトリで管理できます。

```bash
# Hydraの引数
python submit.py --submit -- python train.py seed=2 model=resnet

# argparseなどの引数
python submit.py --submit -- python train.py --seed 2 --model resnet

# シェルスクリプト
python submit.py --submit -- bash scripts/run.sh
```

コマンドは `environment.toml` の `workdir` で実行します。相対パスはそのディレクトリが基準です。
`command` は実行ファイルと引数の文字列配列です。空白を含む引数も1つの文字列として指定できます。
シェルのパイプやリダイレクトが必要なら `["bash", "-c", "コマンド列"]` を使います。
CLIでも `-- bash -c 'コマンド列'` のように指定します。

環境設定とジョブ設定は、このリポジトリの `configs/environment.toml` と `configs/job.toml` を
自動で読み込みます。別の設定を使う場合だけ `--environment <ファイル>` / `--job <ファイル>` を指定します。
投入側のオプションは `--` より前に書いてください。

生成のたびに、ジョブ設定の `name` に `_YYYY-MM-DD_HH-MM-SS_ランダム英数字8文字` を付けます。
日時にはスクリプト生成時のローカル時刻を使い、生成名は64文字以内に収めます。
元の `name` が35文字を超える場合は、先頭35文字に切り詰めます。
たとえば `name = "example-job"` なら、`jobs/example-job_2026-09-06_17-30-00_aB3dE6gH.sh` のように保存します。
PBSのジョブ名とログ保存先にも同じ生成名を使います。設定ファイルの `name` 自体は変更しません。
生成だけではログディレクトリは作成しません。投入に成功すると、ジョブIDとログディレクトリを表示します。

## 予約ノードを使う

予約済みノードに投入する場合は、ジョブ設定の `queue` を予約キュー名に変更し、
`rtype` で資源タイプを指定します。以下はノード占有（`rt_HF`）の例です。

```toml
name = "reserved-job"
queue = "R1234"
rtype = "rt_HF"
walltime = "12:00:00"
command = ["python", "train.py"]
```

`R1234` は例です。`qrstat` の `Queue` 欄に表示される、自分が利用できる予約キュー名に置き換えます。
生成スクリプトには `#PBS -q R1234` と `#PBS -v RTYPE=rt_HF` が出力されます。
単一実験なら、GPU1基を使う `rt_HG` やCPUのみの `rt_HC` も指定できます。
通常の `rt_*` キューでは `rtype` を省略してください。
予約の取得は別途行います。資源や予約の詳細は
[ABCI公式のジョブ実行ドキュメント](https://docs.abci.ai/v3/ja/job-execution/)を参照してください。

## 実行中のログを確認する

ジョブ開始時に、`workdir` 配下へログを直接書き込みます。
PBSによる終了後の出力回収を待つ必要はありません。

```text
logs/<ジョブ名>/<PBS_JOBID>/
├── job.log                 # 環境準備、監視、コマンドの起動・終了結果
└── experiments/
    └── main.log             # コマンドの標準出力・標準エラー
```

投入時に表示されたログディレクトリを使って確認します。次の例では `workdir` に移動済みとします。

```bash
tail -F logs/example-job_2026-09-06_17-30-00_aB3dE6gH/12345.pbs1/job.log
tail -F logs/example-job_2026-09-06_17-30-00_aB3dE6gH/12345.pbs1/experiments/main.log
```

ジョブIDごとに分かれるため、同じジョブ名で再投入しても別の実行のログは上書きしません。
待機中はログがまだ存在しません。状態の確認や取り消しにはスケジューラのコマンドを使います。

```bash
qstat
qdel <job-id>
```

Pythonの出力遅延を抑えるため、スクリプトは `PYTHONUNBUFFERED=1` を設定します。
実行対象やコンテナが環境変数を引き継がない場合、独自にバッファリングする場合は、
実行対象側でも出力のフラッシュを設定してください。
ログディレクトリの作成に失敗するなど、ログ初期化前のエラーはPBSの既定出力に残ります。

## 設定項目

### 環境設定

| 項目 | 必須 | 説明 |
| --- | --- | --- |
| `group` | はい | ABCI利用グループ |
| `workdir` | はい | 実行とログ保存に使う絶対パス |
| `setup_commands` | いいえ | 記載順に実行するシェル文。既定は空 |
| `monitor.enabled` | いいえ | 監視を有効にするか。既定は `false` |
| `monitor.interval_seconds` | 有効時 | 監視間隔。指定する場合は正の整数（秒） |
| `monitor.commands` | 有効時 | 監視で実行するシェル文。有効時は1件以上 |

`setup_commands` と `monitor.commands` はそのままシェルで実行します。
自分で内容を確認した設定を使用してください。

### ジョブ設定

| 項目 | 必須 | 説明 |
| --- | --- | --- |
| `name` | はい | ジョブ名の接頭辞。生成時に日時とランダム英数字を付加 |
| `queue` | はい | 通常の資源タイプ名、または予約キュー名 |
| `rtype` | 予約時 | `rt_HF`・`rt_HG`・`rt_HC` のいずれか。通常の `rt_*` キューでは指定不可 |
| `walltime` | はい | `HHH:MM:SS` 形式の実行時間上限。時間は1〜3桁、全体で0より大きい値 |
| `command` | いいえ | 既定の実行ファイルと引数の文字列配列。CLI指定で全体を置き換え |

設定の `name` は1〜64文字で、先頭は半角英数字、以降は半角英数字・`.`・`_`・`-` を使えます。
未知の設定キーや不正な値は、投入前にエラーになります。

## リポジトリ構成

```text
abci-job/
├── submit.py                         # 共通のCLI
├── abci_job/
│   ├── config.py                     # 環境・ジョブ設定の読み込みと検証
│   ├── submitter.py                  # PBS生成、スクリプト保存、qsub実行
│   └── __init__.py
├── configs/
│   ├── environment_example.toml      # 環境設定のひな形
│   └── job_example.toml              # ジョブ設定のひな形
├── templates/abci.pbs.j2              # PBSジョブのテンプレート
├── jobs/                             # 生成スクリプト（Git管理外）
├── pyproject.toml
└── README.md
```
