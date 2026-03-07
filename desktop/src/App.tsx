import { useState, useEffect, useRef, useCallback } from 'react'
import type { Message, AppStatus, PythonMessage, ToolCall, ToolResult } from './types'
import { Banner } from './components/Banner'
import { MessageBlock } from './components/MessageBlock'
import { ModelPicker, CatalogModel, SearchResult } from './components/ModelPicker'
import { FileExplorer } from './components/FileExplorer'
import { ProviderSelector } from './components/ProviderSelector'
import { FileViewer } from './components/FileViewer'

let reqIdCounter = 0
function nextId() { return ++reqIdCounter }
function uid() { return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}` }

type Catalog = { categories: string[]; models: CatalogModel[] }

export default function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [status, setStatus] = useState<AppStatus>({
    model: '', provider: '', connected: false, tools: [], ready: false,
  })
  const [streaming, setStreaming] = useState(false)
  const [catalog, setCatalog] = useState<Catalog | null>(null)
  const [showPicker, setShowPicker] = useState(false)
  const [pulling, setPulling] = useState<string | null>(null)
  const [pullProgress, setPullProgress] = useState('')
  const [searchResults, setSearchResults] = useState<SearchResult[]>([])
  const [searching, setSearching] = useState(false)
  const [updating, setUpdating] = useState(false)
  const [updateAvailable, setUpdateAvailable] = useState(false)
  const [updateMessage, setUpdateMessage] = useState('')
  const [appUpdating, setAppUpdating] = useState(false)
  const [appUpdateResult, setAppUpdateResult] = useState('')
  const [explorerOpen, setExplorerOpen] = useState(true)
  const [selectedFile, setSelectedFile] = useState<string | null>(null)
  const [hasClaude, setHasClaude] = useState(false)
  const [rootDir, setRootDir] = useState<string | null>(null)
  const terminalRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const activeMessageId = useRef('')
  const [inputText, setInputText] = useState('')

  const scrollToBottom = useCallback(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight
    }
  }, [])

  useEffect(() => { scrollToBottom() }, [messages, scrollToBottom])

  // Initialize hasClaude and rootDir on mount.
  useEffect(() => {
    if (!window.api) return
    window.api.hasClaudeAccess().then(setHasClaude)
    window.api.getHomeDir().then(setRootDir)
  }, [])

  // Ctrl+, keyboard shortcut for app update.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === ',') {
        e.preventDefault()
        if (!appUpdating) {
          window.api.sendToPython({ id: nextId(), type: 'do_update' })
          setAppUpdating(true)
          setAppUpdateResult('')
        }
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [appUpdating])

  // Cmd/Ctrl+B keyboard shortcut to toggle file explorer.
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'b') {
        e.preventDefault()
        setExplorerOpen(prev => !prev)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  const handleTerminalClick = useCallback(() => {
    inputRef.current?.focus()
  }, [])

  const refreshCatalog = useCallback(() => {
    window.api.sendToPython({ id: nextId(), type: 'catalog' })
  }, [])

  // Listen to Python messages.
  useEffect(() => {
    if (!window.api) return

    const cleanup = window.api.onPythonMessage((msg: PythonMessage) => {
      switch (msg.type) {
        case 'ready':
          setStatus({
            model: msg.model || '',
            provider: msg.provider || 'ollama',
            connected: true,
            tools: msg.tools || [],
            ready: true,
          })
          // Read has_claude from ready message.
          if ((msg as any).has_claude !== undefined) {
            setHasClaude(!!(msg as any).has_claude)
          }
          // Fetch catalog.
          window.api.sendToPython({ id: nextId(), type: 'catalog' })
          break

        case 'stream': {
          const content = msg.content || ''
          setMessages(prev => {
            const last = prev[prev.length - 1]
            if (last?.id === activeMessageId.current && last.role === 'assistant') {
              return [...prev.slice(0, -1), { ...last, content: last.content + content, streaming: true }]
            }
            const newId = uid()
            activeMessageId.current = newId
            return [...prev, { id: newId, role: 'assistant', content, streaming: true }]
          })
          break
        }

        case 'tool_call': {
          const tc: ToolCall = { name: msg.name || '', args: msg.args || {} }
          setMessages(prev => {
            const last = prev[prev.length - 1]
            if (last?.id === activeMessageId.current) {
              return [...prev.slice(0, -1), { ...last, toolCalls: [...(last.toolCalls || []), tc] }]
            }
            return prev
          })
          break
        }

        case 'tool_result': {
          const tr: ToolResult = { name: msg.name || '', output: msg.output || '' }
          setMessages(prev => {
            const last = prev[prev.length - 1]
            if (last?.id === activeMessageId.current) {
              return [...prev.slice(0, -1), { ...last, toolResults: [...(last.toolResults || []), tr] }]
            }
            return prev
          })
          break
        }

        case 'done':
          setMessages(prev => {
            const last = prev[prev.length - 1]
            if (last?.id === activeMessageId.current) {
              return [...prev.slice(0, -1), { ...last, streaming: false }]
            }
            return prev
          })
          setStreaming(false)
          break

        case 'error':
          setMessages(prev => [
            ...prev,
            { id: uid(), role: 'system', content: msg.message || 'Unknown error' },
          ])
          setStreaming(false)
          break

        case 'model_changed':
          setStatus(prev => ({ ...prev, model: msg.model || prev.model }))
          break

        case 'provider_changed':
          setStatus(prev => ({ ...prev, provider: msg.provider || prev.provider }))
          break

        case 'cleared':
          setMessages([])
          break

        case 'catalog':
          if (msg.data && typeof msg.data === 'object') {
            setCatalog(msg.data as Catalog)
          }
          break

        case 'search_results':
          if (Array.isArray(msg.data)) {
            setSearchResults(msg.data as SearchResult[])
          }
          setSearching(false)
          break

        case 'catalog_updating':
          setUpdating(true)
          break

        case 'catalog_updated':
          setUpdating(false)
          // Refresh catalog with new data.
          window.api.sendToPython({ id: nextId(), type: 'catalog' })
          break

        case 'pull_progress': {
          const completed = (msg as any).completed
          const total = (msg as any).total
          const pStatus = (msg as any).status || ''
          if (completed != null && total != null && total > 0) {
            const pct = Math.round((completed / total) * 100)
            setPullProgress(`${pStatus} ${pct}%`)
          } else {
            setPullProgress(pStatus)
          }
          break
        }

        case 'pull_done':
          setPulling(null)
          setPullProgress('')
          // Refresh catalog to update installed status.
          window.api.sendToPython({ id: nextId(), type: 'catalog' })
          break

        case 'delete_done':
          // Refresh catalog.
          window.api.sendToPython({ id: nextId(), type: 'catalog' })
          break

        case 'status':
          if (msg.data && typeof msg.data === 'object') {
            const d = msg.data as Record<string, unknown>
            setStatus(prev => ({
              ...prev,
              model: (d.model as string) || prev.model,
              provider: (d.provider as string) || prev.provider,
              connected: (d.connected as boolean) ?? prev.connected,
            }))
          }
          break

        case 'update_available':
          setUpdateAvailable(true)
          setUpdateMessage(msg.message || 'Update available')
          break

        case 'updating':
          setAppUpdating(true)
          break

        case 'update_done': {
          setAppUpdating(false)
          const d = msg as any
          if (d.success) {
            setUpdateAvailable(false)
            setAppUpdateResult(d.message || 'Updated successfully. Restart to apply.')
          } else {
            setAppUpdateResult(d.message || 'Update failed.')
          }
          break
        }
      }
    })

    return cleanup
  }, [])

  const sendMessage = useCallback((text: string) => {
    if (!text.trim() || streaming) return
    activeMessageId.current = ''
    setMessages(prev => [...prev, { id: uid(), role: 'user', content: text }])
    setStreaming(true)
    window.api.sendToPython({ id: nextId(), type: 'chat', content: text })
  }, [streaming])

  const handleKeyDown = useCallback((e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage(inputText)
      setInputText('')
    }
  }, [inputText, sendMessage])

  const handleModelSelect = useCallback((model: string) => {
    setShowPicker(false)
    window.api.sendToPython({ id: nextId(), type: 'switch_model', model })
  }, [])

  const handlePull = useCallback((model: string) => {
    setPulling(model)
    setPullProgress('starting...')
    window.api.sendToPython({ id: nextId(), type: 'pull_model', model })
  }, [])

  const handleUpdateCatalog = useCallback(() => {
    window.api.sendToPython({ id: nextId(), type: 'update_catalog' })
  }, [])

  const handleSearch = useCallback((query: string, sort: string, capability: string) => {
    setSearching(true)
    window.api.sendToPython({ id: nextId(), type: 'search_models', query, sort, capability })
  }, [])

  const handleDelete = useCallback((model: string) => {
    window.api.sendToPython({ id: nextId(), type: 'delete_model', model })
  }, [])

  const handleClear = useCallback(() => {
    window.api.sendToPython({ id: nextId(), type: 'clear' })
  }, [])

  const handleAppUpdate = useCallback(() => {
    setAppUpdating(true)
    setAppUpdateResult('')
    window.api.sendToPython({ id: nextId(), type: 'do_update' })
  }, [])

  const handleProviderSwitch = useCallback((provider: string) => {
    window.api.sendToPython({ id: nextId(), type: 'switch_provider', provider })
  }, [])

  const handleFileSelect = useCallback((path: string) => {
    setSelectedFile(path)
  }, [])

  const handleRootChange = useCallback((path: string) => {
    setRootDir(path)
    setSelectedFile(null)
  }, [])

  const handleInput = useCallback(() => {
    const el = inputRef.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = Math.min(el.scrollHeight, 120) + 'px'
  }, [])

  const statusDotClass = status.ready ? 'on' : status.connected ? 'wait' : 'off'

  return (
    <div className="app">
      <div className="titlebar">
        <span className="titlebar-text">local-cli</span>
      </div>

      <div className="statusbar">
        <div className="status-item">
          <div className={`status-dot ${statusDotClass}`} />
          <span>{status.ready ? 'ready' : 'connecting...'}</span>
        </div>
        <div className="status-item">
          <span
            className="model-select"
            onClick={() => { refreshCatalog(); setShowPicker(true) }}
            title="Click to switch model"
          >
            {status.model || '...'}
          </span>
        </div>
        <div className="status-item">
          <ProviderSelector
            currentProvider={status.provider}
            hasClaude={hasClaude}
            hasMessages={messages.length > 0}
            onSwitch={handleProviderSwitch}
          />
        </div>
        <div className="status-spacer" />
        {pulling && (
          <div className="status-item">
            <span style={{ color: 'var(--yellow)' }}>pulling {pulling}...</span>
          </div>
        )}
        <div className="status-item">
          <span style={{ color: 'var(--text-muted)' }}>{status.tools.length} tools</span>
        </div>
        {messages.length > 0 && (
          <button className="status-btn" onClick={handleClear}>clear</button>
        )}
      </div>

      {(updateAvailable || appUpdateResult) && (
        <div className="update-bar">
          {appUpdating ? (
            <span className="update-bar-text">Updating...</span>
          ) : appUpdateResult ? (
            <>
              <span className="update-bar-text">{appUpdateResult}</span>
              <button className="update-bar-dismiss" onClick={() => setAppUpdateResult('')}>dismiss</button>
            </>
          ) : (
            <>
              <span className="update-bar-text">{updateMessage}</span>
              <button className="update-bar-btn" onClick={handleAppUpdate}>Install update</button>
              <button className="update-bar-dismiss" onClick={() => setUpdateAvailable(false)}>later</button>
            </>
          )}
        </div>
      )}

      <div className="app-body">
        {explorerOpen && (
          <div className="explorer-pane">
            <FileExplorer
              rootDir={rootDir}
              onFileSelect={handleFileSelect}
              onRootChange={handleRootChange}
            />
          </div>
        )}

        <div className="chat-pane">
          <div className="terminal" ref={terminalRef} onClick={handleTerminalClick}>
            <div className="terminal-inner">
              {messages.length === 0 ? (
                <div className="welcome">
                  <Banner version="0.2.0" />
                  <div className="welcome-sub">
                    Local AI coding agent powered by Ollama.
                    Read, write, edit files. Run commands. Search code.
                  </div>
                  <div className="welcome-hint">
                    Type a message below to start. Shift+Enter for newline.
                  </div>
                </div>
              ) : (
                messages.map(msg => <MessageBlock key={msg.id} message={msg} />)
              )}
            </div>
          </div>

          <div className="input-area">
            <div className="input-row">
              <span className="input-marker">&gt;</span>
              <textarea
                ref={inputRef}
                className="input-field"
                value={inputText}
                onChange={e => { setInputText(e.target.value); handleInput() }}
                onKeyDown={handleKeyDown}
                placeholder={!status.ready ? 'waiting for backend...' : 'ask anything...'}
                disabled={!status.ready || streaming}
                rows={1}
              />
            </div>
          </div>
        </div>

        {selectedFile && (
          <div className="viewer-pane">
            <FileViewer
              filePath={selectedFile}
              onClose={() => setSelectedFile(null)}
            />
          </div>
        )}
      </div>

      {showPicker && (
        <ModelPicker
          catalog={catalog}
          searchResults={searchResults}
          current={status.model}
          onSelect={handleModelSelect}
          onPull={handlePull}
          onDelete={handleDelete}
          onSearch={handleSearch}
          onUpdate={handleUpdateCatalog}
          onClose={() => setShowPicker(false)}
          pulling={pulling}
          pullProgress={pullProgress}
          searching={searching}
          updating={updating}
        />
      )}
    </div>
  )
}
