import { useState, useEffect, useCallback } from 'react'

type Props = {
  onClose: () => void
  onAuthenticated: (method: 'api_key' | 'subscription') => void
  isAuthenticated: boolean
  authMethod: string | null
}

export function ClaudeLogin({ onClose, onAuthenticated, isAuthenticated, authMethod }: Props) {
  const [tab, setTab] = useState<'api_key' | 'subscription'>('api_key')
  const [apiKey, setApiKey] = useState('')
  const [validating, setValidating] = useState(false)
  const [error, setError] = useState('')
  const [success, setSuccess] = useState('')

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  // Load existing key (masked) on mount.
  useEffect(() => {
    window.api.getClaudeAuth().then(auth => {
      if (auth.method === 'api_key' && auth.keyHint) {
        setApiKey(auth.keyHint)
      }
    })
  }, [])

  const handleSaveApiKey = useCallback(async () => {
    const key = apiKey.trim()
    if (!key) {
      setError('API key is required.')
      return
    }
    if (!key.startsWith('sk-ant-')) {
      setError('Invalid key format. Anthropic API keys start with sk-ant-')
      return
    }

    setValidating(true)
    setError('')
    setSuccess('')

    const result = await window.api.saveClaudeKey(key)
    setValidating(false)

    if (result.success) {
      setSuccess('API key saved and validated.')
      setApiKey(result.keyHint || key.slice(0, 10) + '...')
      onAuthenticated('api_key')
    } else {
      setError(result.error || 'Failed to save API key.')
    }
  }, [apiKey, onAuthenticated])

  const handleLogout = useCallback(async () => {
    await window.api.deleteClaudeAuth()
    setApiKey('')
    setSuccess('')
    setError('')
    // Notify parent that auth was removed.
    window.api.sendToPython({ id: Date.now(), type: 'claude_logout' })
  }, [])

  const handleSubscriptionLogin = useCallback(async () => {
    setValidating(true)
    setError('')
    setSuccess('')

    const result = await window.api.startClaudeOAuth()

    setValidating(false)
    if (result.success) {
      setSuccess('Logged in successfully.')
      onAuthenticated('subscription')
    } else {
      setError(result.error || 'Login failed.')
    }
  }, [onAuthenticated])

  return (
    <div className="overlay" onClick={onClose}>
      <div className="claude-login" onClick={e => e.stopPropagation()}>
        <div className="claude-login-header">
          <span className="claude-login-title">Claude Authentication</span>
          <button className="settings-close" onClick={onClose}>x</button>
        </div>

        {isAuthenticated && (
          <div className="claude-login-status">
            <span className="claude-login-status-dot" />
            <span>Authenticated via {authMethod === 'api_key' ? 'API Key' : 'Subscription'}</span>
            <button className="claude-logout-btn" onClick={handleLogout}>Logout</button>
          </div>
        )}

        <div className="claude-login-tabs">
          <button
            className={`claude-login-tab ${tab === 'api_key' ? 'active' : ''}`}
            onClick={() => { setTab('api_key'); setError(''); setSuccess('') }}
          >
            API Key
          </button>
          <button
            className={`claude-login-tab ${tab === 'subscription' ? 'active' : ''}`}
            onClick={() => { setTab('subscription'); setError(''); setSuccess('') }}
          >
            Subscription
          </button>
        </div>

        <div className="claude-login-body">
          {tab === 'api_key' ? (
            <div className="claude-login-section">
              <div className="claude-login-desc">
                Enter your Anthropic API key. Get one from{' '}
                <span
                  className="claude-login-link"
                  onClick={() => window.api.openExternalUrl('https://console.anthropic.com/settings/keys')}
                >
                  console.anthropic.com
                </span>
              </div>
              <div className="claude-login-field">
                <input
                  type="password"
                  className="claude-login-input"
                  value={apiKey}
                  onChange={e => setApiKey(e.target.value)}
                  placeholder="sk-ant-api03-..."
                  onKeyDown={e => { if (e.key === 'Enter') handleSaveApiKey() }}
                  autoFocus
                />
              </div>
              <button
                className="claude-login-btn primary"
                onClick={handleSaveApiKey}
                disabled={validating || !apiKey.trim()}
              >
                {validating ? 'Validating...' : 'Save & Validate'}
              </button>
            </div>
          ) : (
            <div className="claude-login-section">
              <div className="claude-login-desc">
                Log in with your Claude Pro or Max subscription.
                A browser window will open for authentication.
              </div>
              <button
                className="claude-login-btn primary"
                onClick={handleSubscriptionLogin}
                disabled={validating}
              >
                {validating ? 'Logging in...' : 'Login with Claude'}
              </button>
              <div className="claude-login-hint">
                Requires an active Claude Pro or Max subscription.
              </div>
            </div>
          )}

          {error && <div className="claude-login-error">{error}</div>}
          {success && <div className="claude-login-success">{success}</div>}
        </div>
      </div>
    </div>
  )
}
