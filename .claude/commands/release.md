GitHub Release: バージョン更新 → ビルド → リリース → アップロード

$ARGUMENTS にバージョン番号を指定 (例: 1.0.0)

## 0. プレフライトチェック

```bash
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
OLD_VER=$(git describe --tags --abbrev=0 2>/dev/null | sed 's/^v//' || echo "unknown")
```

以下を確認。問題があればユーザーに報告して中断:
- `git status` — 未コミットの変更がないか（あればコミットするか確認）
- `git tag -l "v$ARGUMENTS"` — 同じタグが既にないか（あれば中止）
- テスト/lint があれば実行（`npm test`, `pytest`, `cargo test` 等）。失敗したら中止
- プロジェクト構成を確認: package.json, pyproject.toml, Cargo.toml, go.mod, desktop/ の有無

## 1. バージョン更新

旧バージョン文字列を grep して該当ファイルをすべて $ARGUMENTS に書き換える:
```bash
grep -rn "$OLD_VER" --include="*.py" --include="*.ts" --include="*.tsx" --include="*.json" --include="*.toml" --include="*.svg" --include="*.cfg" .
```
対象例: package.json, pyproject.toml, Cargo.toml, __init__.py, APP_VERSION定数, USER_AGENT, バッジSVG等。

## 3. コミット・プッシュ

```bash
git add -A && git commit -m "release: v$ARGUMENTS" && git push origin $(git branch --show-current)
```

## 4. ビルド (該当する場合)

プロジェクト種別に応じてビルド。成果物を `ls -lh` で確認（異常に小さいなら壊れている）。

## 5. アセット準備・アップロード

成果物があれば /tmp/ にスペースなしの名前でコピーしてからアップロード:
```bash
# コピー (スペース→ドット)
for f in dist/*.dmg dist/*.zip dist/*.exe dist/*.AppImage dist/*.tar.gz dist/*.whl; do
  [ -f "$f" ] && cp "$f" "/tmp/$(basename "$f" | tr ' ' '.')"
done

# リリース作成
gh release create "v$ARGUMENTS" --title "v$ARGUMENTS" --generate-notes

# アップロード (curl方式 — 大ファイル・スペース対策で最も確実)
UPLOAD_URL=$(gh api "repos/$REPO/releases/tags/v$ARGUMENTS" --jq '.upload_url' | sed 's/{.*//')
TOKEN=$(gh auth token)
for f in /tmp/*.dmg /tmp/*.zip /tmp/*.exe /tmp/*.AppImage; do
  [ -f "$f" ] || continue
  NAME=$(basename "$f")
  curl --progress-bar -H "Authorization: token $TOKEN" -H "Content-Type: application/octet-stream" \
    "${UPLOAD_URL}?name=$NAME" --data-binary "@$f" -o /tmp/upload-$NAME.json
done
```

成果物がなければ `gh release create` のみ。

## 6. 確認

```bash
gh api "repos/$REPO/releases/tags/v$ARGUMENTS" --jq '.assets[] | "\(.name) — \(.size) bytes"'
```

## 大容量アセット対策 (100MB+)

GitHub Releases のファイル上限は **2GB/ファイル**。それ以上は分割が必要。

### アップロード前
- `ls -lh` でサイズ確認。合計を把握する
- 100MB以上のファイルは **1ファイルずつ** バックグラウンド（`run_in_background`）でアップロード
- アップロード中はスリープ禁止（caffeinate で防止可）:
  ```bash
  caffeinate -i curl --progress-bar ... &
  ```

### 速度の目安
| 回線 | 100MB | 200MB | 500MB |
|------|-------|-------|-------|
| 1 Mbps (≈0.13MB/s) | ~13分 | ~26分 | ~65分 |
| 10 Mbps (≈1.2MB/s) | ~1.5分 | ~3分 | ~7分 |
| 50 Mbps (≈6MB/s) | ~17秒 | ~34秒 | ~1.5分 |

### 2GB超のアセットがある場合
split で分割 → アップロード → READMEに結合手順を記載:
```bash
split -b 1900m /tmp/large-asset.dmg /tmp/large-asset.part-
# → large-asset.part-aa, large-asset.part-ab, ...
# ダウンロード後: cat large-asset.part-* > large-asset.dmg
```

### アップロード失敗時のリカバリ
途中で切れた場合、同名でアップロードするとエラーになる。既存アセットを削除してリトライ:
```bash
ASSET_ID=$(gh api "repos/$REPO/releases/tags/v$ARGUMENTS" --jq '.assets[] | select(.name=="ファイル名") | .id')
gh api -X DELETE "repos/$REPO/releases/assets/$ASSET_ID"
# → 再度 curl でアップロード
```

## ロールバック

リリース後に問題が見つかった場合:
```bash
# リリースとタグを削除
gh release delete "v$ARGUMENTS" --yes --cleanup-tag
# バージョンコミットを戻す
git revert HEAD --no-edit && git push origin $(git branch --show-current)
```

## その他の注意事項
- `dial tcp: no such host` → スリープ復帰直後のDNS。ネット確認してリトライ
- `HTTP 404` → upload_url を再取得
