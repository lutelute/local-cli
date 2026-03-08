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

function ToolBlock({ name, args, output, running }: {
  name: string
  args: Record<string, unknown>
  output?: string
  running?: boolean
}) {
  const [collapsed, setCollapsed] = useState(true)

  const argStr = formatToolArgs(name, args)

  const handleCopy = useCallback((e: React.MouseEvent) => {
    e.stopPropagation()
    if (output) {
      navigator.clipboard.writeText(output)
    }
  }, [output])

  return (
    <div className="tool">
      <div className="tool-header">
        <div className="tool-cmd">
          {running && <span className="tool-spinner" />}
          {!running && output && <span className="tool-check">&#10003;</span>}
          <span className="tool-name">{name}</span>
          {argStr && <span className="tool-arg"> {argStr}</span>}
        </div>
        {running && <span className="tool-status">running</span>}
      </div>
      {output && (
        <div className="tool-output-wrap">
          <div
            className={`tool-output ${collapsed ? 'collapsed' : ''}`}
            onClick={() => setCollapsed(!collapsed)}
          >
            <pre className="tool-output-pre">{output}</pre>
          </div>
          <div className="tool-output-actions">
            {output.length > 100 && (
              <span className="tool-toggle" onClick={() => setCollapsed(!collapsed)}>
                {collapsed ? 'show more' : 'collapse'}
              </span>
            )}
            <span className="tool-copy" onClick={handleCopy} title="Copy output">
              copy
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

function formatToolArgs(name: string, args: Record<string, unknown>): string {
  switch (name) {
    case 'read':
    case 'write':
    case 'glob':
      return String(args.file_path || args.path || args.pattern || '')
    case 'bash':
      return String(args.command || '')
    case 'grep':
      return `"${args.pattern || ''}" ${args.path || ''}`
    case 'edit': {
      return String(args.file_path || '')
    }
    default: {
      const entries = Object.entries(args)
      if (entries.length === 0) return ''
      return entries.map(([k, v]) => {
        const s = typeof v === 'string' ? v : JSON.stringify(v)
        return `${k}=${s.length > 50 ? s.slice(0, 47) + '...' : s}`
      }).join(' ')
    }
  }
}
