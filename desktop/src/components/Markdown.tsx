import { useMemo } from 'react'
import { parseMarkdown, type Block, type Inline } from '../markdown'

// Renders assistant text as lightweight markdown.  Every string lands
// in a text node (no innerHTML anywhere), so the model's output cannot
// inject markup.  Links open in the system browser via the preload
// bridge instead of navigating the Electron window.

function openLink(href: string) {
  return (e: React.MouseEvent) => {
    e.preventDefault()
    window.api?.openExternalUrl(href)
  }
}

function InlineRun({ runs }: { runs: Inline[] }) {
  return (
    <>
      {runs.map((run, i) => {
        switch (run.kind) {
          case 'code':
            return <code key={i} className="md-inline-code">{run.text}</code>
          case 'bold':
            return <strong key={i} className="md-bold">{run.text}</strong>
          case 'link':
            return (
              <a key={i} className="md-link" href={run.href}
                 onClick={openLink(run.href)} title={run.href}>
                {run.text}
              </a>
            )
          default:
            return <span key={i}>{run.text}</span>
        }
      })}
    </>
  )
}

function BlockView({ block }: { block: Block }) {
  switch (block.kind) {
    case 'heading': {
      const level = Math.min(block.level, 4)
      return (
        <div className={`md-heading md-h${level}`}>
          <InlineRun runs={block.inline} />
        </div>
      )
    }
    case 'code':
      return (
        <div className="md-code">
          {block.lang && <div className="md-code-lang">{block.lang}</div>}
          <pre className="md-code-pre">{block.text}</pre>
        </div>
      )
    case 'list':
      return block.ordered ? (
        <ol className="md-list">
          {block.items.map((item, i) => (
            <li key={i}><InlineRun runs={item} /></li>
          ))}
        </ol>
      ) : (
        <ul className="md-list">
          {block.items.map((item, i) => (
            <li key={i}><InlineRun runs={item} /></li>
          ))}
        </ul>
      )
    case 'table':
      return (
        <div className="md-table-wrap">
          <table className="md-table">
            <thead>
              <tr>
                {block.header.map((cell, i) => (
                  <th key={i}><InlineRun runs={cell} /></th>
                ))}
              </tr>
            </thead>
            <tbody>
              {block.rows.map((row, r) => (
                <tr key={r}>
                  {row.map((cell, c) => (
                    <td key={c}><InlineRun runs={cell} /></td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )
    default:
      return (
        <div className="md-p">
          {block.lines.map((line, i) => (
            <div key={i} className="md-p-line">
              <InlineRun runs={line} />
            </div>
          ))}
        </div>
      )
  }
}

export function Markdown({ text }: { text: string }) {
  const blocks = useMemo(() => parseMarkdown(text), [text])
  return (
    <div className="md-root">
      {blocks.map((block, i) => <BlockView key={i} block={block} />)}
    </div>
  )
}
