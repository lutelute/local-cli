import { useState } from 'react'
import type { Message } from '../types'

type Props = { message: Message }

export function MessageBlock({ message }: Props) {
  const { role, content, toolCalls, toolResults, streaming } = message

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

  // Assistant
  return (
    <div className="msg">
      <div className="msg-prompt">
        <span className="msg-marker agent">$</span>
        <span className="msg-body agent">
          {content}
          {streaming && <span className="cursor" />}
        </span>
      </div>

      {toolCalls?.map((tc, i) => (
        <ToolBlock
          key={`t-${i}`}
          name={tc.name}
          args={tc.args}
          output={toolResults?.[i]?.output}
        />
      ))}
    </div>
  )
}

function ToolBlock({ name, args, output }: {
  name: string
  args: Record<string, unknown>
  output?: string
}) {
  const [collapsed, setCollapsed] = useState(true)

  const argStr = formatToolArgs(name, args)

  return (
    <div className="tool">
      <div className="tool-cmd">
        <span className="tool-name">{name}</span>
        {argStr && <span className="tool-arg"> {argStr}</span>}
      </div>
      {output && (
        <>
          <div
            className={`tool-output ${collapsed ? 'collapsed' : ''}`}
            onClick={() => setCollapsed(!collapsed)}
          >
            {output}
          </div>
          {output.length > 100 && (
            <span className="tool-toggle" onClick={() => setCollapsed(!collapsed)}>
              {collapsed ? '... show more' : '^ collapse'}
            </span>
          )}
        </>
      )}
    </div>
  )
}

function formatToolArgs(name: string, args: Record<string, unknown>): string {
  // Show args in a terminal-friendly way.
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
      const fp = String(args.file_path || '')
      return fp
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
