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
})
