import { useState, useCallback } from 'react'
import type { Message } from '../types'

type Props = { message: Message }

export function MessageBlock({ message }: Props) {
  const { role, content, toolCalls, toolResults, streaming, thinking } = message

  if (role === 'system') {
    return (
      <div className="msg">
        <div className="msg-prompt">
          <span className="msg-marker system">!</span>
          <span className="msg-body error">{content}</span>
        </div>
      </div>
    )
  }

  if (role === 'user') {
    return (
      <div className="msg">
        <div className="msg-prompt">
          <span className="msg-marker user">&gt;</span>
          <span className="msg-body user">{content}</span>
        </div>
      </div>
    )
  }

  // Assistant — thinking state.
  if (thinking && !content && !toolCalls?.length) {
    return (
      <div className="msg">
        <div className="msg-prompt">
          <span className="msg-marker agent">$</span>
          <span className="msg-body agent">
            <span className="thinking-indicator">
              <span className="thinking-dot" />
              <span className="thinking-dot" />
              <span className="thinking-dot" />
              <span className="thinking-label">thinking</span>
            </span>
          </span>
        </div>
      </div>
    )
  }

  // Determine if any tool is currently executing (has call but no result yet).
  const pendingToolIndex = toolCalls
    ? toolCalls.findIndex((_, i) => !toolResults?.[i])
    : -1

  return (
    <div className="msg">
      {content && (
        <div className="msg-prompt">
          <span className="msg-marker agent">$</span>
          <span className="msg-body agent">
            {content}
            {streaming && !toolCalls?.length && <span className="cursor" />}
          </span>
        </div>
      )}

      {toolCalls?.map((tc, i) => (
        <ToolBlock
          key={`t-${i}`}
          name={tc.name}
          args={tc.args}
          output={toolResults?.[i]?.output}
          running={i === pendingToolIndex}
        />
      ))}

      {/* Waiting for next LLM response after tools executed */}
      {toolCalls && toolCalls.length > 0 && toolResults && toolResults.length >= toolCalls.length && streaming && (
        <div className="thinking-after-tool">
          <span className="thinking-dot" />
          <span className="thinking-dot" />
          <span className="thinking-dot" />
          <span className="thinking-label">processing results</span>
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Tool display metadata
// ---------------------------------------------------------------------------

type ToolMeta = {
  label: string
  color: string // CSS class suffix: tool-cat-read, tool-cat-search, etc.
}

function getToolMeta(name: string): ToolMeta {
  switch (name) {
    case 'read':
      return { label: 'Reading', color: 'read' }
    case 'write':
      return { label: 'Writing', color: 'write' }
    case 'edit':
      return { label: 'Editing', color: 'write' }
    case 'grep':
      return { label: 'Searching code', color: 'search' }
    case 'glob':
      return { label: 'Searching files', color: 'search' }
    case 'bash':
      return { label: 'Running', color: 'run' }
    case 'agent':
      return { label: 'Agent', color: 'agent' }
    case 'ask_user':
      return { label: 'Asking', color: 'ask' }
    case 'web_fetch':
      return { label: 'Fetching', color: 'fetch' }
    default:
      return { label: name, color: 'default' }
  }
}

// ---------------------------------------------------------------------------
// ToolBlock
// ---------------------------------------------------------------------------

function ToolBlock({ name, args, output, running }: {
  name: string
  args: Record<string, unknown>
  output?: string
  running?: boolean
}) {
  const [collapsed, setCollapsed] = useState(true)
  const meta = getToolMeta(name)
  const detail = formatToolDetail(name, args)
  const done = !running && output !== undefined

  const handleCopy = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    if (output) {
      navigator.clipboard.writeText(output)
    }
  }, [output])

  return (
    <div className={`tool tool-cat-${meta.color}`}>
      <div className="tool-header">
        <div className="tool-cmd">
          {running && <span className="tool-spinner" />}
          {done && <span className="tool-check">&#10003;</span>}
          <span className="tool-label">{meta.label}</span>
        </div>
        <div className="tool-status-badge">
          {running && <span className="tool-status">Running</span>}
          {done && <span className="tool-done">Done</span>}
        </div>
      </div>
      {detail && (
        <div className="tool-detail">{detail}</div>
      )}
      {output && (
        <div className="tool-output-wrap">
          <div className="tool-output-bar">
            {output.length > 100 && (
              <span className="tool-toggle" onClick={() => setCollapsed(!collapsed)}>
                {collapsed ? 'Show output' : 'Collapse'}
              </span>
            )}
            {output.length <= 100 && (
              <span className="tool-toggle" onClick={() => setCollapsed(!collapsed)}>
                {collapsed ? 'Show output' : 'Collapse'}
              </span>
            )}
            <span className="tool-copy" onClick={handleCopy} title="Copy output">
              copy
            </span>
          </div>
          {!collapsed && (
            <div className="tool-output">
              <pre className="tool-output-pre">{output}</pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Format detail line shown under the tool label
// ---------------------------------------------------------------------------

function formatToolDetail(name: string, args: Record<string, unknown>): string {
  switch (name) {
    case 'read':
      return String(args.file_path || args.path || '')
    case 'write':
      return String(args.file_path || '')
    case 'edit':
      return String(args.file_path || '')
    case 'glob':
      return `pattern: ${args.pattern || args.path || ''}`
    case 'bash':
      return String(args.command || '')
    case 'grep': {
      const pat = args.pattern || ''
      const p = args.path || ''
      return p ? `pattern: ${pat}  path: ${p}` : `pattern: ${pat}`
    }
    case 'agent':
      return String(args.description || args.prompt || '').slice(0, 80)
    case 'web_fetch':
      return String(args.url || '')
    default: {
      const entries = Object.entries(args)
      if (entries.length === 0) return ''
      return entries.map(([k, v]) => {
        const s = typeof v === 'string' ? v : JSON.stringify(v)
        return `${k}: ${s.length > 60 ? s.slice(0, 57) + '...' : s}`
      }).join('  ')
    }
  }
}
