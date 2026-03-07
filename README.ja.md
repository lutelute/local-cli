```
 ██╗      ██████╗  ██████╗ █████╗ ██╗          ██████╗██╗     ██╗
 ██║     ██╔═══██╗██╔════╝██╔══██╗██║         ██╔════╝██║     ██║
 ██║     ██║   ██║██║     ███████║██║  █████╗ ██║     ██║     ██║
 ██║     ██║   ██║██║     ██╔══██║██║  ╚════╝ ██║     ██║     ██║
 ███████╗╚██████╔╝╚██████╗██║  ██║███████╗    ╚██████╗███████╗██║
 ╚══════╝ ╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝     ╚═════╝╚══════╝╚═╝
```

**ローカルで動くAIコーディングエージェント。Ollama搭載。**

外部依存ゼロ。すべてあなたのマシンで完結します。

[English](README.md) | 日本語 | [やさしいにほんご](README.easy-ja.md)

---

## これは何？

Local CLIは、[Ollama](https://ollama.com)を使ってローカルで動作するAIコーディングエージェントです。自然言語でファイルの読み書き・編集、シェルコマンドの実行、コード検索、Webページの取得ができます。

クラウド型AIコーディングアシスタントの、ローカル・オフライン対応版です。

### 主な機能

- **エージェントループ** — LLMが自律的にツール（read, write, edit, bash, glob, grep）を呼び出してタスクを完了
- **ストリーミング応答** — トークン単位のリアルタイム出力
- **マルチプロバイダ** — Ollama（ローカル）とClaude APIを実行時に切り替え可能
- **モデル管理** — CLIやデスクトップアプリからモデルのインストール・削除・切り替え
- **RAGエンジン** — コードベースをインデックスしてコンテキストを考慮した応答
- **Gitチェックポイント** — 編集中のセーフポイント作成とロールバック
- **セッション永続化** — 会話の保存と再開
- **デスクトップGUI** — ターミナル風UIのElectronアプリ
- **外部依存ゼロ** — Python標準ライブラリのみ（コアにpip installは不要）

## ダウンロード

### デスクトップアプリ（ビルド済み）

**[GitHub Releases](https://github.com/lutelute/local-cli/releases)** から最新版をダウンロード：

| プラットフォーム | ファイル |
|----------|------|
| macOS | `Local-CLI-x.x.x-universal.dmg` |
| Windows | `Local-CLI-Setup-x.x.x.exe` |
| Linux | `Local-CLI-x.x.x.AppImage` または `.deb` |

> **注意:** [Ollama](https://ollama.com)のインストールと起動が必要です。

#### macOS:「アプリが壊れている」「開発元が未確認」と表示される場合

Apple Developer証明書で署名されていないため、Gatekeeperがブロックします。以下で解除できます：

```bash
xattr -cr /Applications/Local\ CLI.app
```

または：**システム設定 > プライバシーとセキュリティ > 下にスクロール >「このまま開く」をクリック**

### CLI（ソースから）

```bash
# 必要なもの: Python 3.10+, Ollama, Git

# 1. クローン
git clone https://github.com/lutelute/local-cli.git
cd local-cli

# 2. 直接実行
python -m local_cli

# またはコマンドとしてインストール
pip install -e .
local-cli
```

### モデルの対話的選択

```bash
# 起動時にモデルを選ぶ
local-cli --select-model

# モデルを指定して起動
local-cli --model qwen3:8b
```

## アップデート

起動時に自動で更新を確認します。更新がある場合は通知が表示されます。

```bash
# CLI: ターミナルから更新
local-cli --update

# CLI: REPL内で更新
/update

# デスクトップ: 通知バーの「Install update」をクリック
```

`pip install -e .` はソースディレクトリへのリンクなので、`git pull` だけで更新されます。`/update` コマンドはこれを自動で行います。

## アーキテクチャ

```
local-cli/
├── local_cli/
│   ├── __main__.py          # エントリーポイント
│   ├── agent.py             # エージェントループ（LLM ↔ ツール）
│   ├── cli.py               # REPL + スラッシュコマンド
│   ├── config.py            # 設定（CLI > 環境変数 > 設定ファイル > デフォルト）
│   ├── server.py            # デスクトップGUI用JSON-lineサーバー
│   ├── ollama_client.py     # Ollama REST APIクライアント
│   ├── orchestrator.py      # マルチプロバイダオーケストレーション
│   ├── model_catalog.py     # モデルカタログ + キャッシュ
│   ├── model_search.py      # ollama.comからのライブ検索
│   ├── model_manager.py     # インストール/削除/情報操作
│   ├── model_registry.py    # タスク→モデルルーティング
│   ├── model_selector.py    # 対話式TUIモデルピッカー
│   ├── rag.py               # 検索拡張生成（RAG）
│   ├── git_ops.py           # Gitチェックポイント/ロールバック
│   ├── session.py           # セッション保存/読込
│   ├── security.py          # 入力バリデーション
│   ├── updater.py           # セルフアップデート（git pull）
│   ├── providers/           # LLMプロバイダ抽象化
│   │   ├── base.py          # 抽象クラス LLMProvider
│   │   ├── ollama_provider.py
│   │   ├── claude_provider.py
│   │   ├── message_converter.py
│   │   └── sse_parser.py
│   └── tools/               # エージェントツール
│       ├── bash_tool.py     # シェルコマンド実行
│       ├── read_tool.py     # ファイル読み取り
│       ├── write_tool.py    # ファイル作成
│       ├── edit_tool.py     # 文字列置換による編集
│       ├── glob_tool.py     # ファイルパターン検索
│       ├── grep_tool.py     # ファイル内容検索
│       ├── web_fetch_tool.py
│       └── ask_user_tool.py
├── desktop/                 # Electron + React + Vite
│   ├── electron/            # メインプロセス + preload
│   ├── src/                 # React UIコンポーネント
│   └── build/               # アプリアイコン
└── tests/                   # テスト961件
```

## ツール

エージェントが使えるツール：

| ツール | 説明 |
|--------|------|
| `bash` | シェルコマンドを実行 |
| `read` | ファイルの内容を読み取り |
| `write` | ファイルの作成・上書き |
| `edit` | 文字列検索・置換による編集 |
| `glob` | パターンでファイルを検索 |
| `grep` | ファイル内容を検索 |
| `web_fetch` | Webページを取得 |
| `ask_user` | ユーザーに質問 |

## スラッシュコマンド

| コマンド | 説明 |
|----------|------|
| `/help` | コマンド一覧を表示 |
| `/model <名前>` | モデルを切り替え |
| `/models` | 対話式モデルセレクターを開く |
| `/status` | 接続状況とモデル情報を表示 |
| `/provider [名前]` | LLMプロバイダを切り替え/表示 |
| `/brain [モデル]` | オーケストレータのブレインモデルを設定 |
| `/install <モデル>` | モデルをダウンロード |
| `/uninstall <モデル>` | モデルを削除 |
| `/info <モデル>` | モデルの詳細を表示 |
| `/running` | VRAMにロード中のモデルを一覧 |
| `/checkpoint` | Gitチェックポイントを作成 |
| `/rollback [タグ]` | チェックポイントにロールバック |
| `/save` | セッションを保存 |
| `/update` | 最新版に更新 |
| `/clear` | 会話をクリア |
| `/exit` | 終了 |

## デスクトップアプリ

ターミナル風GUIを提供するデスクトップアプリ：

- ストリーミングチャットとツール呼び出しの表示
- モデルピッカー: **Catalog**（カテゴリ別厳選モデル）と **Discover**（ollama.comからのライブ検索）
- モデルのダウンロード・切り替え・削除
- 自動アップデート通知バー
- ASCIIアートのウェルカムバナー

ElectronとPython間の通信はstdin/stdout JSONライン方式 — ネットワークサーバー不要、API依存なし。

### ソースから実行

```bash
cd desktop
npm install
npm run dev          # 開発モード（ホットリロード）
# または
npx vite build && npx electron .   # プロダクションプレビュー
```

### インストーラのビルド

```bash
cd desktop
npm run build        # 現在のプラットフォーム向け
npm run build:mac    # macOS → .dmg + .zip（universal）
npm run build:win    # Windows → NSISインストーラ
npm run build:linux  # Linux → AppImage + .deb
```

出力先は `desktop/dist/`。インストーラにはPythonソースとElectronランタイムが含まれます。

### GitHub Releasesへのリリース

```bash
cd desktop

# ビルドしてGitHub Releasesにアップロード（GH_TOKENが必要）
export GH_TOKEN=your_github_token
npx electron-builder --publish always
```

GitHub上にドラフトリリースが作成されます。GitHub UIから編集・公開してください。

## 設定

設定の優先順位: CLIフラグ > 環境変数 > 設定ファイル > デフォルト値

| フラグ | 環境変数 | デフォルト | 説明 |
|--------|----------|-----------|------|
| `--model` | `LOCAL_CLI_MODEL` | `qwen3:8b` | 使用するモデル |
| `--provider` | `LOCAL_CLI_PROVIDER` | `ollama` | LLMプロバイダ |
| `--debug` | `LOCAL_CLI_DEBUG` | `false` | デバッグ出力 |
| `--rag` | — | `false` | RAGを有効化 |
| `--rag-path` | — | `.` | インデックスするディレクトリ |
| `--select-model` | — | `false` | 対話式モデルピッカー |
| `--server` | — | `false` | JSON-lineサーバーモード |
| `--update` | — | `false` | 最新版に更新 |

## おすすめモデル

| モデル | サイズ | 用途 |
|--------|--------|------|
| `qwen3:8b` | 5.2 GB | 汎用、ツール呼び出し |
| `qwen2.5-coder:7b` | 4.7 GB | コード生成 |
| `qwen3:30b` | 18.5 GB | 複雑な推論、エージェント |
| `deepseek-r1:14b` | 9.0 GB | 思考の連鎖による推論 |
| `qwen3:0.6b` | 0.5 GB | テスト用 |

## テスト

```bash
python -m pytest tests/ -q
# 961 passed
```

## ライセンス

MIT
