import { useState, useEffect, useCallback } from 'react'

type Props = {
  onClose: () => void
}

type AppUpdateInfo = {
  available: boolean
  version: string
  notes?: string
  downloadUrl?: string
  releaseUrl?: string
  error?: string
}

export function SettingsPanel({ onClose }: Props) {
  const [appVersion, setAppVersion] = useState('')
  const [appUpdate, setAppUpdate] = useState<AppUpdateInfo | null>(null)
  const [checkingApp, setCheckingApp] = useState(false)

  const [cliUpdating, setCliUpdating] = useState(false)
  const [cliResult, setCliResult] = useState('')

  useEffect(() => {
    window.api.getAppVersion().then(setAppVersion)
  }, [])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  const handleCheckAppUpdate = useCallback(async () => {
    setCheckingApp(true)
    setAppUpdate(null)
    const result = await window.api.checkAppUpdate()
    setAppUpdate(result)
    setCheckingApp(false)
  }, [])

  const handleDownloadApp = useCallback((url: string) => {
    window.api.openExternalUrl(url)
  }, [])

  const handleCliUpdate = useCallback(() => {
    setCliUpdating(true)
    setCliResult('')
    // Use the existing Python server update mechanism.
    window.api.sendToPython({ id: Date.now(), type: 'do_update' })

    // Listen for result via onPythonMessage (handled in App.tsx).
    // We'll use a simple approach: poll via a listener.
    const cleanup = window.api.onPythonMessage((msg) => {
      if (msg.type === 'update_done') {
        const d = msg as any
        setCliUpdating(false)
        setCliResult(d.message || (d.success ? 'Updated!' : 'Failed'))
        cleanup()
      }
    })
  }, [])

  return (
    <div className="overlay" onClick={onClose}>
      <div className="settings-panel" onClick={e => e.stopPropagation()}>
        <div className="settings-header">
          <span className="settings-title">Settings</span>
          <button className="settings-close" onClick={onClose}>x</button>
        </div>

        <div className="settings-body">
          {/* Desktop App Update */}
          <div className="settings-section">
            <div className="settings-section-title">Desktop App (GUI)</div>
            <div className="settings-row">
              <span className="settings-label">Version</span>
              <span className="settings-value">v{appVersion}</span>
            </div>
            <div className="settings-row">
              <span className="settings-label">Update</span>
              <div className="settings-value">
                {checkingApp ? (
                  <span className="settings-checking">Checking...</span>
                ) : appUpdate ? (
                  appUpdate.available ? (
                    <div className="settings-update-info">
                      <span className="settings-update-new">v{appUpdate.version} available</span>
                      <button
                        className="settings-update-btn"
                        onClick={() => handleDownloadApp(appUpdate.downloadUrl || appUpdate.releaseUrl || '')}
                      >
                        Download
                      </button>
                      {appUpdate.releaseUrl && (
                        <button
                          className="settings-update-link"
                          onClick={() => handleDownloadApp(appUpdate.releaseUrl!)}
                        >
                          Release notes
                        </button>
                      )}
                    </div>
                  ) : (
                    <span className="settings-up-to-date">Up to date</span>
                  )
                ) : (
                  <button className="settings-check-btn" onClick={handleCheckAppUpdate}>
                    Check for updates
                  </button>
                )}
              </div>
            </div>
          </div>

          {/* Backend CLI Update */}
          <div className="settings-section">
            <div className="settings-section-title">local-cli (Backend)</div>
            <div className="settings-row">
              <span className="settings-label">Update</span>
              <div className="settings-value">
                {cliUpdating ? (
                  <span className="settings-checking">Updating via git pull...</span>
                ) : cliResult ? (
                  <span className="settings-cli-result">{cliResult}</span>
                ) : (
                  <button className="settings-check-btn" onClick={handleCliUpdate}>
                    Update backend (git pull)
                  </button>
                )}
              </div>
            </div>
            <div className="settings-hint">
              Updates Python backend from the git repository.
              Requires restart after update.
            </div>
          </div>

          {/* Keyboard Shortcuts */}
          <div className="settings-section">
            <div className="settings-section-title">Keyboard Shortcuts</div>
            <div className="settings-shortcut">
              <kbd>Cmd+,</kbd> <span>Settings</span>
            </div>
            <div className="settings-shortcut">
              <kbd>Cmd+B</kbd> <span>Toggle file explorer</span>
            </div>
            <div className="settings-shortcut">
              <kbd>Esc</kbd> <span>Stop generation</span>
            </div>
            <div className="settings-shortcut">
              <kbd>Shift+Enter</kbd> <span>New line</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
