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
- **決定的ハーネス** — 小型ローカルモデル（1〜9B）の失敗パターンを検出・修復する介入層（後述）。CLI・サーバー・Webモニタ・サブエージェントが共通の単一ループを走る
- **ストリーミング応答** — トークン単位のリアルタイム出力
- **マルチプロバイダ** — Ollama（ローカル）とClaude APIを実行時に切り替え可能
- **モデル管理** — CLIやデスクトップアプリからモデルのインストール・削除・切り替え
- **RAGエンジン** — コードベースをインデックスしてコンテキストを考慮した応答
- **Gitチェックポイント** — 編集中のセーフポイント作成とロールバック
- **セッション永続化** — 会話の保存と再開
- **会話の自動保存と復元** — 毎ターン後に会話をプロジェクト単位で自動保存。アプリを閉じても消えない。CLIは `/resume`、desktopは起動時の「Restore」ボタンで前回の続きから再開（`/clear` で破棄）
- **セッションログ（フライトレコーダー）** — フォルダを開いた時点から全セッションを `~/.local/state/local-cli/projects/<フォルダ名スラグ>/` にJSONLで自動記録（ユーザー発話・ツール呼び出し・ハーネス介入・ターンごとの可視/思考文字数）。不具合報告の事後診断がトランスクリプトだけで可能に
- **プロジェクト指示ファイル** — `LOCAL_CLI.md`（または `AGENTS.md` / `CLAUDE.md`）をプロジェクトに置くと、毎セッション自動でシステム指示として注入。言語・命名・禁止事項などのプロジェクト規約を小型モデルに毎回確実に守らせる（最近接ディレクトリ優先、gitルートで探索打ち切り、8KBクリップ）
- **デスクトップGUI** — ターミナル風UIのElectronアプリ
- **マスコット Loca** — `--mascot` でスピナーが猫 `(=･ω･=)` に、`--mascot pixel` で動くドット絵の子猫に（任意・既定オフ）
- **外部依存ゼロ** — Python標準ライブラリのみ（コアにpip installは不要）

### 決定的ハーネス

大手クラウドモデルを前提とせず、小型ローカルモデルでも長いエージェント作業を完遂させるための「決定的な介入」を多数備えます。すべて実モデルの失敗ログから設計・検証しています。

| 介入 | 直す失敗 |
|------|----------|
| テキストtool call救済 | ツール呼び出しを構造化せずテキストで書くモデルでも実行 — `<tool_call>`タグ / fenced JSON / 素のJSON / 引数トップレベル形（`{"name": "write", "file_path": ...}`）/ ツール名キー形（`{"write": {...}}`）/ Python呼び出し構文（`write(file_path=...)`） |
| tools非対応フォールバック | エンドポイントが`tools`を拒否したら、fenced-JSON形式を教えてテキスト駆動に自動切替 |
| ツール名・引数の補正 | `write_file`→`write`、`path`→`file_path` などの近似名を解決 |
| ループ検出 | 同一呼び出しの繰り返しを検出し、是正リマインダー→強制まとめ |
| 書き込み後の構文検証 | write/edit直後に`.py`/`.json`/`.toml`を構文チェックし、全ファイル型でマージ衝突マーカーも検知、エラーを即フィードバック |
| edit失敗ヒント | `old_text`不一致時に最類似ブロックを行番号付きで提示 |
| ファイル未検出パスヒント | 幻覚パスに対し実在する同名ファイルを提示 |
| 空応答・エラー終了ガード | 沈黙／失敗直後の終了を1回押し戻す（bashの非ゼロ終了 `[exit code: N]` も失敗として検知） |
| コンテキスト圧縮 | 切り詰め（既定）またはLLM要約（`compact_mode=summarize`） |

介入効果は `scripts/harness_eval.py`（実モデル×機械判定タスク、決定的デコード、leave-one-outアブレーション）で測定できます。参考スコア: qwen3.5:4b は英語・日本語とも満点級、sub-1Bは単純タスク中心。

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
└── tests/                   # テスト2310件
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
| `todo_write` | 構造化タスクリストを管理（未着手 / 進行中 / 完了） |
| `agent` | サブエージェントを生成して並列実行 |

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
| `/resume` | このフォルダの前回の会話を復元 |
| `/update` | 最新版に更新 |
| `/clear` | 会話をクリア |
| `/exit` | 終了 |

## デスクトップアプリ

ターミナル風GUIを提供するデスクトップアプリ：

- ストリーミングチャットとツール呼び出しの表示
- **Markdownレンダリング** — 見出し・コードブロック・箇条書き・表・リンクを整形表示（依存ゼロの自前レンダラー、テキストは常にテキストノードでXSS構造不能、リンクは外部ブラウザで開く）
- **ハーネス介入チップ** — rescue / nudge / deliverable_nudge などの介入を紫チップで可視化
- **会話の復元バー** — フォルダに前回の会話があれば起動時に「Restore」を提示
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
| `--model` | `LOCAL_CLI_MODEL` | `qwen3.5:9b-q4_K_M` | 使用するモデル |
| `--provider` | `LOCAL_CLI_PROVIDER` | `ollama` | LLMプロバイダ |
| `--debug` | `LOCAL_CLI_DEBUG` | `false` | デバッグ出力 |
| `--rag` | — | `false` | RAGを有効化 |
| `--rag-path` | — | `.` | インデックスするディレクトリ |
| `--select-model` | — | `false` | 対話式モデルピッカー |
| `--server` | — | `false` | JSON-lineサーバーモード |
| `--yes` / `-y` | — | `false` | 危険コマンドの確認を自動承認 |
| `--update` | — | `false` | 今すぐ更新（git pull + 再インストール） |
| `--auto-update` | `LOCAL_CLI_AUTO_UPDATE` | `false` | 起動時に更新があれば自動インストールして再起動を促す |
| `--mascot [style]` | `LOCAL_CLI_MASCOT` | `off` | マスコット Loca: `--mascot` で顔文字 `(=･ω･=)`、`--mascot pixel` で動くドット絵（TTYのみ、パイプ時は顔文字にフォールバック） |
| — | `LOCAL_CLI_COMPACT_MODE` | `truncate` | コンテキスト圧縮: `truncate` または `summarize` |
| — | `LOCAL_CLI_MAX_ITERATIONS` | `40` | 1ターンの反復上限（`0`=無制限） |
| — | `LOCAL_CLI_SESSION_LOG` | `1` | セッションログ（`0`で無効化）。保存先: `<state_dir>/projects/<cwdスラグ>/` |
| — | `LOCAL_CLI_PROJECT_INSTRUCTIONS` | `1` | プロジェクト指示ファイルの自動注入（`0`で無効化） |

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
# 2310 passed
```

## ライセンス

MIT
