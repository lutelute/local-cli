import { useState, useEffect, useCallback } from 'react'

type Props = {
  onClose: () => void
}

type AppUpdateInfo = {
  available: boolean
  version: string
  notes?: string
  downloadUrl?: string
  zipUrl?: string
  releaseUrl?: string
  error?: string
}

type UpdateProgress = {
  stage: string
  percent: number
}

export function SettingsPanel({ onClose }: Props) {
  const [appVersion, setAppVersion] = useState('')
  const [appUpdate, setAppUpdate] = useState<AppUpdateInfo | null>(null)
  const [checkingApp, setCheckingApp] = useState(false)
  const [installing, setInstalling] = useState(false)
  const [installProgress, setInstallProgress] = useState<UpdateProgress | null>(null)
  const [installError, setInstallError] = useState('')

  const [cliUpdating, setCliUpdating] = useState(false)
  const [cliResult, setCliResult] = useState('')

  useEffect(() => {
    window.api.getAppVersion().then(setAppVersion)
  }, [])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !installing) onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose, installing])

  // Listen for update progress events.
  useEffect(() => {
    const cleanup = window.api.onUpdateProgress((progress: UpdateProgress) => {
      setInstallProgress(progress)
    })
    return cleanup
  }, [])

  const handleCheckAppUpdate = useCallback(async () => {
    setCheckingApp(true)
    setAppUpdate(null)
    setInstallError('')
    const result = await window.api.checkAppUpdate()
    setAppUpdate(result)
    setCheckingApp(false)
  }, [])

  const handleInstallUpdate = useCallback(async () => {
    if (!appUpdate?.zipUrl) return
    setInstalling(true)
    setInstallError('')
    setInstallProgress({ stage: 'starting', percent: 0 })
    const result = await window.api.installAppUpdate(appUpdate.zipUrl)
    if (!result.success) {
      setInstalling(false)
      setInstallError(result.error || 'Install failed')
    }
    // On success the app will quit and relaunch automatically.
  }, [appUpdate])

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
                {installing ? (
                  <div className="settings-update-progress">
                    <div className="settings-progress-label">
                      {installProgress?.stage === 'downloading' ? 'Downloading...' :
                       installProgress?.stage === 'extracting' ? 'Extracting...' :
                       installProgress?.stage === 'installing' ? 'Installing... (app will restart)' :
                       'Preparing...'}
                    </div>
                    <div className="settings-progress-bar">
                      <div
                        className="settings-progress-fill"
                        style={{ width: `${installProgress?.percent || 0}%` }}
                      />
                    </div>
                    <div className="settings-progress-pct">{installProgress?.percent || 0}%</div>
                  </div>
                ) : checkingApp ? (
                  <span className="settings-checking">Checking...</span>
                ) : appUpdate ? (
                  appUpdate.available ? (
                    <div className="settings-update-info">
                      <span className="settings-update-new">v{appUpdate.version} available</span>
                      {appUpdate.zipUrl ? (
                        <button className="settings-update-btn" onClick={handleInstallUpdate}>
                          Install update
                        </button>
                      ) : (
                        <button
                          className="settings-update-btn"
                          onClick={() => window.api.openExternalUrl(appUpdate.downloadUrl || appUpdate.releaseUrl || '')}
                        >
                          Download
                        </button>
                      )}
                      {installError && <span className="settings-error">{installError}</span>}
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
