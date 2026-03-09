import { contextBridge, ipcRenderer } from 'electron'

export type PythonMessage = {
  id?: number
  type: string
  content?: string
  name?: string
  args?: Record<string, unknown>
  output?: string
  data?: unknown
  model?: string
  tools?: string[]
  message?: string
  provider?: string
}

contextBridge.exposeInMainWorld('api', {
  sendToPython: (data: object) => {
    ipcRenderer.send('send-to-python', data)
  },
  signalReady: () => {
    ipcRenderer.send('renderer-ready')
  },
  onPythonMessage: (callback: (msg: PythonMessage) => void) => {
    const handler = (_event: Electron.IpcRendererEvent, msg: PythonMessage) => callback(msg)
    ipcRenderer.on('python-message', handler)
    return () => ipcRenderer.removeListener('python-message', handler)
  },
  onPythonStderr: (callback: (text: string) => void) => {
    const handler = (_event: Electron.IpcRendererEvent, text: string) => callback(text)
    ipcRenderer.on('python-stderr', handler)
    return () => ipcRenderer.removeListener('python-stderr', handler)
  },
  onPythonExit: (callback: (code: number | null) => void) => {
    const handler = (_event: Electron.IpcRendererEvent, code: number | null) => callback(code)
    ipcRenderer.on('python-exit', handler)
    return () => ipcRenderer.removeListener('python-exit', handler)
  },
  getPythonStatus: () => ipcRenderer.invoke('get-python-status'),
  listDir: (dirPath: string) => ipcRenderer.invoke('list-directory', dirPath),
  readFile: (filePath: string) => ipcRenderer.invoke('read-file', filePath),
  openDirectoryDialog: () => ipcRenderer.invoke('open-directory-dialog'),
  getHomeDir: () => ipcRenderer.invoke('get-home-dir'),
  hasClaudeAccess: () => ipcRenderer.invoke('has-claude-access'),
  getAppVersion: () => ipcRenderer.invoke('get-app-version'),
  checkAppUpdate: () => ipcRenderer.invoke('check-app-update'),
  installAppUpdate: (zipUrl: string) => ipcRenderer.invoke('install-app-update', zipUrl),
  onUpdateProgress: (callback: (progress: { stage: string; percent: number }) => void) => {
    const handler = (_event: Electron.IpcRendererEvent, progress: { stage: string; percent: number }) => callback(progress)
    ipcRenderer.on('update-progress', handler)
    return () => ipcRenderer.removeListener('update-progress', handler)
  },
  openExternalUrl: (url: string) => ipcRenderer.invoke('open-external-url', url),
  getClaudeAuth: () => ipcRenderer.invoke('get-claude-auth'),
  saveClaudeKey: (key: string) => ipcRenderer.invoke('save-claude-key', key),
  deleteClaudeAuth: () => ipcRenderer.invoke('delete-claude-auth'),
  startClaudeOAuth: () => ipcRenderer.invoke('start-claude-oauth'),
})
