import { app, BrowserWindow, dialog, ipcMain, shell } from 'electron'
import { spawn, ChildProcess } from 'child_process'
import https from 'node:https'
import fs from 'node:fs'
import os from 'node:os'
import path from 'path'

const APP_VERSION = '0.4.0'
const GITHUB_REPO = 'lutelute/local-cli'

let mainWindow: BrowserWindow | null = null
let pythonProcess: ChildProcess | null = null
let lineBuffer = ''
let pendingMessages: object[] = []
let rendererReady = false
let lastReadyMessage: object | null = null

function findPython(): string {
  // Try python3 first, then python.
  return process.platform === 'win32' ? 'python' : 'python3'
}

function findProjectRoot(): string {
  // In dev mode: __dirname is desktop/dist-electron, so ../../ is project root.
  // In packaged app: try extraResources path first, fall back to dev path.
  const devRoot = path.resolve(__dirname, '..', '..')
  const resourcesRoot = process.resourcesPath
    ? path.resolve(process.resourcesPath)
    : devRoot
  return devRoot
}

function startPythonServer() {
  const pythonCmd = findPython()
  const projectRoot = findProjectRoot()

  console.log(`Starting Python server: ${pythonCmd} -m local_cli --server`)
  console.log(`Working directory: ${projectRoot}`)

  pythonProcess = spawn(pythonCmd, ['-m', 'local_cli', '--server'], {
    cwd: projectRoot,
    stdio: ['pipe', 'pipe', 'pipe'],
    env: { ...process.env, PYTHONUNBUFFERED: '1' },
  })

  pythonProcess.stdout?.on('data', (data: Buffer) => {
    lineBuffer += data.toString()
    const lines = lineBuffer.split('\n')
    // Keep the last incomplete line in the buffer.
    lineBuffer = lines.pop() || ''

    for (const line of lines) {
      if (!line.trim()) continue
      try {
        const msg = JSON.parse(line)
        // Remember the ready message so we can re-send it on HMR re-mounts.
        if (msg.type === 'ready') {
          lastReadyMessage = msg
        }
        if (rendererReady && mainWindow) {
          mainWindow.webContents.send('python-message', msg)
        } else {
          // Buffer messages until renderer is ready.
          pendingMessages.push(msg)
        }
      } catch {
        console.error('Invalid JSON from Python:', line)
      }
    }
  })

  pythonProcess.stderr?.on('data', (data: Buffer) => {
    const text = data.toString()
    console.error('[python stderr]', text)
    mainWindow?.webContents.send('python-stderr', text)
  })

  pythonProcess.on('exit', (code) => {
    console.log(`Python process exited with code ${code}`)
    mainWindow?.webContents.send('python-exit', code)
    pythonProcess = null
  })
}

function sendToPython(data: object) {
  if (pythonProcess?.stdin?.writable) {
    pythonProcess.stdin.write(JSON.stringify(data) + '\n')
  }
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1000,
    height: 700,
    minWidth: 600,
    minHeight: 400,
    titleBarStyle: 'hiddenInset',
    trafficLightPosition: { x: 16, y: 16 },
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
    },
  })

  // In dev mode, load from Vite dev server.
  if (process.env.VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(process.env.VITE_DEV_SERVER_URL)
  } else {
    mainWindow.loadFile(path.join(__dirname, '../dist/index.html'))
  }

  mainWindow.on('closed', () => {
    mainWindow = null
  })
}

// IPC handlers.
ipcMain.on('send-to-python', (_event, data) => {
  sendToPython(data)
})

ipcMain.handle('get-python-status', () => {
  return { running: pythonProcess !== null && !pythonProcess.killed }
})

ipcMain.on('renderer-ready', () => {
  rendererReady = true
  const hadPending = pendingMessages.length > 0
  // Flush any messages that arrived before the renderer was ready.
  for (const msg of pendingMessages) {
    mainWindow?.webContents.send('python-message', msg)
  }
  pendingMessages = []

  // On HMR re-mounts (and React StrictMode double-mounts) the buffer is
  // empty but the renderer needs the ready message again.
  if (!hadPending && lastReadyMessage) {
    mainWindow?.webContents.send('python-message', lastReadyMessage)
  }
})

ipcMain.handle('open-directory-dialog', async () => {
  const result = await dialog.showOpenDialog({ properties: ['openDirectory'] })
  return result
})

ipcMain.handle('list-directory', async (_event, dirPath: string) => {
  const entries = await fs.promises.readdir(dirPath, { withFileTypes: true })
  const results = await Promise.all(
    entries.map(async (entry) => {
      const fullPath = path.join(dirPath, entry.name)
      let isSymlink = false
      try {
        const stat = await fs.promises.lstat(fullPath)
        isSymlink = stat.isSymbolicLink()
      } catch {
        // Ignore stat errors for individual entries.
      }
      return { name: entry.name, isDirectory: entry.isDirectory(), isSymlink }
    })
  )
  return results
})

const MAX_FILE_SIZE = 1_000_000 // 1MB

ipcMain.handle('read-file', async (_event, filePath: string) => {
  const stat = await fs.promises.stat(filePath)
  if (stat.size > MAX_FILE_SIZE) {
    return { error: 'too_large' as const, size: stat.size }
  }
  const content = await fs.promises.readFile(filePath, 'utf-8')
  return { content }
})

ipcMain.handle('get-home-dir', () => {
  return os.homedir()
})

ipcMain.handle('has-claude-access', () => {
  return !!process.env.ANTHROPIC_API_KEY
})

ipcMain.handle('get-app-version', () => APP_VERSION)

ipcMain.handle('check-app-update', async () => {
  try {
    const data = await fetchJson(`https://api.github.com/repos/${GITHUB_REPO}/releases/latest`)
    const latest = (data.tag_name || '').replace(/^v/, '')
    if (latest && latest !== APP_VERSION) {
      const dmg = data.assets?.find((a: any) => a.name.endsWith('.dmg'))
      return {
        available: true,
        version: latest,
        notes: data.body || '',
        downloadUrl: dmg?.browser_download_url || data.html_url,
        releaseUrl: data.html_url,
      }
    }
    return { available: false, version: APP_VERSION }
  } catch {
    return { available: false, version: APP_VERSION, error: 'Failed to check for updates' }
  }
})

ipcMain.handle('open-external-url', (_event, url: string) => {
  shell.openExternal(url)
})

function fetchJson(url: string): Promise<any> {
  return new Promise((resolve, reject) => {
    const req = https.get(url, { headers: { 'User-Agent': `local-cli/${APP_VERSION}` } }, (res) => {
      // Follow redirects.
      if (res.statusCode === 301 || res.statusCode === 302) {
        const loc = res.headers.location
        if (loc) return fetchJson(loc).then(resolve, reject)
      }
      let body = ''
      res.on('data', (chunk) => { body += chunk })
      res.on('end', () => {
        try { resolve(JSON.parse(body)) }
        catch { reject(new Error('Invalid JSON')) }
      })
    })
    req.on('error', reject)
    req.setTimeout(15000, () => { req.destroy(); reject(new Error('Timeout')) })
  })
}

app.whenReady().then(() => {
  startPythonServer()
  createWindow()
})

app.on('window-all-closed', () => {
  pythonProcess?.kill()
  app.quit()
})

app.on('before-quit', () => {
  pythonProcess?.kill()
})
