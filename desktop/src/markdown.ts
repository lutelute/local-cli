// Minimal markdown tokenizer for assistant output — zero dependencies.
//
// The desktop app renders every reply as plain text, so reports full of
// headings, fences and tables arrive as raw markup.  This module parses
// the safe, common subset (fences, headings, lists, tables, inline
// code/bold/links) into plain token objects; the Markdown component
// maps them to React elements.  Text stays text nodes end to end, so
// no HTML injection is possible by construction.
//
// Pure functions only (no React import): node can unit-test this file
// after a bare tsc compile.

export type Inline =
  | { kind: 'text'; text: string }
  | { kind: 'code'; text: string }
  | { kind: 'bold'; text: string }
  | { kind: 'link'; text: string; href: string }

export type Block =
  | { kind: 'p'; lines: Inline[][] }
  | { kind: 'heading'; level: number; inline: Inline[] }
  | { kind: 'code'; lang: string; text: string }
  | { kind: 'list'; ordered: boolean; items: Inline[][] }
  | { kind: 'table'; header: Inline[][]; rows: Inline[][][] }

const FENCE_RE = /^\s*```(.*)$/
const HEADING_RE = /^(#{1,4})\s+(.*)$/
const BULLET_RE = /^\s*[-*+]\s+(.*)$/
const ORDERED_RE = /^\s*\d+[.)]\s+(.*)$/
const TABLE_ROW_RE = /^\s*\|(.+)\|\s*$/
const TABLE_SEP_RE = /^\s*\|?\s*:?-{2,}.*\|.*$/

// Inline syntax: code spans first (nothing nests inside them), then
// bold, then links.
const INLINE_RE = /(`[^`]+`)|(\*\*[^*]+\*\*)|(\[[^\]]+\]\((?:https?:\/\/)[^\s)]+\))/g
const LINK_RE = /^\[([^\]]+)\]\(([^)]+)\)$/

export function parseInline(text: string): Inline[] {
  const out: Inline[] = []
  let last = 0
  for (const match of text.matchAll(INLINE_RE)) {
    const index = match.index ?? 0
    if (index > last) {
      out.push({ kind: 'text', text: text.slice(last, index) })
    }
    const token = match[0]
    if (token.startsWith('`')) {
      out.push({ kind: 'code', text: token.slice(1, -1) })
    } else if (token.startsWith('**')) {
      out.push({ kind: 'bold', text: token.slice(2, -2) })
    } else {
      const link = LINK_RE.exec(token)
      if (link) {
        out.push({ kind: 'link', text: link[1], href: link[2] })
      } else {
        out.push({ kind: 'text', text: token })
      }
    }
    last = index + token.length
  }
  if (last < text.length) {
    out.push({ kind: 'text', text: text.slice(last) })
  }
  return out
}

function splitTableRow(line: string): Inline[][] {
  const inner = line.trim().replace(/^\|/, '').replace(/\|$/, '')
  return inner.split('|').map(cell => parseInline(cell.trim()))
}

export function parseMarkdown(text: string): Block[] {
  const lines = text.split('\n')
  const blocks: Block[] = []
  let paragraph: Inline[][] = []

  const flushParagraph = () => {
    if (paragraph.length > 0) {
      blocks.push({ kind: 'p', lines: paragraph })
      paragraph = []
    }
  }

  let i = 0
  while (i < lines.length) {
    const line = lines[i]

    // Fenced code block (an unterminated fence swallows the rest —
    // graceful while a reply is still streaming).
    const fence = FENCE_RE.exec(line)
    if (fence) {
      flushParagraph()
      const lang = fence[1].trim()
      const body: string[] = []
      i += 1
      while (i < lines.length && !FENCE_RE.test(lines[i])) {
        body.push(lines[i])
        i += 1
      }
      i += 1 // skip the closing fence (or run past the end)
      blocks.push({ kind: 'code', lang, text: body.join('\n') })
      continue
    }

    const heading = HEADING_RE.exec(line)
    if (heading) {
      flushParagraph()
      blocks.push({
        kind: 'heading',
        level: heading[1].length,
        inline: parseInline(heading[2].trim()),
      })
      i += 1
      continue
    }

    // Table: a |row| line whose next line is the separator.
    if (
      TABLE_ROW_RE.test(line)
      && i + 1 < lines.length
      && TABLE_SEP_RE.test(lines[i + 1])
    ) {
      flushParagraph()
      const header = splitTableRow(line)
      const rows: Inline[][][] = []
      i += 2
      while (i < lines.length && TABLE_ROW_RE.test(lines[i])) {
        rows.push(splitTableRow(lines[i]))
        i += 1
      }
      blocks.push({ kind: 'table', header, rows })
      continue
    }

    const bullet = BULLET_RE.exec(line)
    const ordered = ORDERED_RE.exec(line)
    if (bullet || ordered) {
      flushParagraph()
      const isOrdered = !bullet
      const items: Inline[][] = []
      while (i < lines.length) {
        const item = isOrdered
          ? ORDERED_RE.exec(lines[i])
          : BULLET_RE.exec(lines[i])
        if (!item) break
        items.push(parseInline(item[1].trim()))
        i += 1
      }
      blocks.push({ kind: 'list', ordered: isOrdered, items })
      continue
    }

    if (line.trim() === '') {
      flushParagraph()
      i += 1
      continue
    }

    paragraph.push(parseInline(line))
    i += 1
  }
  flushParagraph()
  return blocks
}
