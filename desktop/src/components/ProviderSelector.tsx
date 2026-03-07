import { useState, useRef, useEffect, useCallback } from 'react'

type Props = {
  currentProvider: string
  hasClaude: boolean
  hasMessages: boolean
  onSwitch: (provider: string) => void
}

type ProviderOption = {
  id: string
  label: string
  description: string
}

const PROVIDERS: ProviderOption[] = [
  { id: 'claude', label: 'Claude Code', description: 'Anthropic API' },
  { id: 'ollama', label: 'Local LLM', description: 'Ollama' },
]

export function ProviderSelector({ currentProvider, hasClaude, hasMessages, onSwitch }: Props) {
  const [open, setOpen] = useState(false)
  const [confirming, setConfirming] = useState<string | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  // Close on Escape key.
  useEffect(() => {
    if (!open) return
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setOpen(false)
        setConfirming(null)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [open])

  // Close on click outside.
  useEffect(() => {
    if (!open) return
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setOpen(false)
        setConfirming(null)
      }
    }
    window.addEventListener('mousedown', handler)
    return () => window.removeEventListener('mousedown', handler)
  }, [open])

  const handleSelect = useCallback((providerId: string) => {
    if (providerId === currentProvider) {
      setOpen(false)
      return
    }

    if (providerId === 'claude' && !hasClaude) return

    if (hasMessages) {
      setConfirming(providerId)
      return
    }

    onSwitch(providerId)
    setOpen(false)
    setConfirming(null)
  }, [currentProvider, hasClaude, hasMessages, onSwitch])

  const handleConfirm = useCallback(() => {
    if (confirming) {
      onSwitch(confirming)
      setOpen(false)
      setConfirming(null)
    }
  }, [confirming, onSwitch])

  const handleCancel = useCallback(() => {
    setConfirming(null)
  }, [])

  const displayLabel = PROVIDERS.find(p => p.id === currentProvider)?.description || currentProvider

  return (
    <div className="provider-select-container" ref={containerRef}>
      <span
        className="model-select"
        onClick={() => setOpen(!open)}
        title="Click to switch provider"
      >
        {displayLabel}
      </span>

      {open && (
        <div className="provider-dropdown">
          {confirming ? (
            <div className="provider-confirm">
              <div className="provider-confirm-text">
                Switching providers will clear the conversation. Continue?
              </div>
              <div className="provider-confirm-actions">
                <button className="provider-confirm-btn yes" onClick={handleConfirm}>Switch</button>
                <button className="provider-confirm-btn no" onClick={handleCancel}>Cancel</button>
              </div>
            </div>
          ) : (
            PROVIDERS.map(p => {
              const isActive = p.id === currentProvider
              const isDisabled = p.id === 'claude' && !hasClaude

              return (
                <div
                  key={p.id}
                  className={`provider-option ${isActive ? 'active' : ''} ${isDisabled ? 'disabled' : ''}`}
                  onClick={() => !isDisabled && handleSelect(p.id)}
                  title={isDisabled ? 'API key required' : undefined}
                >
                  <span className="provider-check">{isActive ? '✓' : ''}</span>
                  <div className="provider-option-info">
                    <span className="provider-option-label">{p.label}</span>
                    <span className="provider-option-desc">{p.description}</span>
                  </div>
                </div>
              )
            })
          )}
        </div>
      )}
    </div>
  )
}
