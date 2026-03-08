import { app, BrowserWindow, dialog, ipcMain, shell, safeStorage } from 'electron'
import { spawn, ChildProcess } from 'child_process'
import http from 'node:http'
import https from 'node:https'
import crypto from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'path'

const APP_VERSION = '0.5.5'
const GITHUB_REPO = 'lutelute/local-cli'

// Claude auth credential storage path.
const CLAUDE_AUTH_PATH = path.join(
  process.env.XDG_CONFIG_HOME || path.join(os.homedir(), '.config'),
  'local-cli',
  'claude-auth.json',
)

let mainWindow: BrowserWindow | null = null
let pythonProcess: ChildProcess | null = null
let lineBuffer = ''
let pendingMessages: object[] = []
let rendererReady = false
let lastReadyMessage: object | null = null
let storedApiKey: string | null = null

// ---------------------------------------------------------------------------
// Claude credential storage (encrypted with safeStorage when available)
// ---------------------------------------------------------------------------

type ClaudeAuth = {
  method: 'api_key' | 'subscription'
  key: string // encrypted (base64) or plain
  encrypted: boolean
}

function loadClaudeAuth(): ClaudeAuth | null {
  try {
    if (!fs.existsSync(CLAUDE_AUTH_PATH)) return null
    const raw = fs.readFileSync(CLAUDE_AUTH_PATH, 'utf-8')
    const data = JSON.parse(raw) as ClaudeAuth
    if (data.encrypted && safeStorage.isEncryptionAvailable()) {
      data.key = safeStorage.decryptString(Buffer.from(data.key, 'base64'))
    }
    return data
  } catch {
    return null
  }
}

function saveClaudeAuth(method: 'api_key' | 'subscription', key: string): void {
  const dir = path.dirname(CLAUDE_AUTH_PATH)
  fs.mkdirSync(dir, { recursive: true })

  let storedKey = key
  let encrypted = false
  if (safeStorage.isEncryptionAvailable()) {
    storedKey = safeStorage.encryptString(key).toString('base64')
    encrypted = true
  }
  fs.writeFileSync(
    CLAUDE_AUTH_PATH,
    JSON.stringify({ method, key: storedKey, encrypted } as ClaudeAuth),
    { mode: 0o600 },
  )
}

function deleteClaudeAuth(): void {
  try { fs.unlinkSync(CLAUDE_AUTH_PATH) } catch { /* ignore */ }
  storedApiKey = null
}

function initClaudeAuth(): void {
  // Load stored credentials and set env var for Python.
  const auth = loadClaudeAuth()
  if (auth?.key) {
    storedApiKey = auth.key
    process.env.ANTHROPIC_API_KEY = auth.key
  }
}

/** Validate an API key by making a lightweight request to the Anthropic API. */
function validateApiKey(key: string): Promise<{ valid: boolean; error?: string }> {
  return new Promise((resolve) => {
    const body = JSON.stringify({
      model: 'claude-haiku-4-5',
      max_tokens: 1,
      messages: [{ role: 'user', content: 'hi' }],
    })
    const req = https.request(
      'https://api.anthropic.com/v1/messages',
      {
        method: 'POST',
        headers: {
          'x-api-key': key,
          'anthropic-version': '2023-06-01',
          'content-type': 'application/json',
        },
        timeout: 15000,
      },
      (res) => {
        let data = ''
        res.on('data', (c) => { data += c })
        res.on('end', () => {
          if (res.statusCode === 200) {
            resolve({ valid: true })
          } else if (res.statusCode === 401) {
            resolve({ valid: false, error: 'Invalid API key.' })
          } else {
            // 429 (rate limit) or other errors — key format is valid.
            resolve({ valid: true })
          }
        })
      },
    )
    req.on('error', () => resolve({ valid: false, error: 'Could not reach Anthropic API.' }))
    req.on('timeout', () => { req.destroy(); resolve({ valid: false, error: 'Request timed out.' }) })
    req.write(body)
    req.end()
  })
}

// ---------------------------------------------------------------------------
// OAuth flow for Claude subscription
// ---------------------------------------------------------------------------

function startOAuthFlow(): Promise<{ success: boolean; error?: string }> {
  return new Promise((resolve) => {
    // Start a local HTTP server to receive the OAuth callback.
    const state = crypto.randomBytes(16).toString('hex')
    let resolved = false

    const server = http.createServer((req, res) => {
      const url = new URL(req.url || '/', `http://localhost`)
      if (url.pathname !== '/callback') {
        res.writeHead(404)
        res.end()
        return
      }

      const code = url.searchParams.get('code')
      const returnedState = url.searchParams.get('state')
      const error = url.searchParams.get('error')

      if (error) {
        res.writeHead(200, { 'Content-Type': 'text/html' })
        res.end('<html><body><h2>Login failed</h2><p>You can close this window.</p></body></html>')
        if (!resolved) {
          resolved = true
          server.close()
          resolve({ success: false, error: `OAuth error: ${error}` })
        }
        return
      }

      if (!code || returnedState !== state) {
        res.writeHead(400, { 'Content-Type': 'text/html' })
        res.end('<html><body><h2>Invalid request</h2></body></html>')
        return
      }

      res.writeHead(200, { 'Content-Type': 'text/html' })
      res.end('<html><body><h2>Login successful!</h2><p>You can close this window and return to Local CLI.</p></body></html>')

      if (!resolved) {
        resolved = true
        server.close()

        // Exchange the auth code for a session key.
        exchangeOAuthCode(code).then((result) => {
          if (result.apiKey) {
            saveClaudeAuth('subscription', result.apiKey)
            storedApiKey = result.apiKey
            process.env.ANTHROPIC_API_KEY = result.apiKey
            // Notify Python backend.
            sendToPython({ type: 'set_api_key', api_key: result.apiKey })
            resolve({ success: true })
          } else {
            resolve({ success: false, error: result.error || 'Failed to exchange code.' })
          }
        })
      }
    })

    // Listen on a random available port.
    server.listen(0, '127.0.0.1', () => {
      const addr = server.address()
      if (!addr || typeof addr === 'string') {
        resolve({ success: false, error: 'Failed to start callback server.' })
        return
      }
      const port = addr.port
      const redirectUri = `http://127.0.0.1:${port}/callback`
      const authUrl = `https://console.anthropic.com/oauth/authorize?response_type=code&redirect_uri=${encodeURIComponent(redirectUri)}&state=${state}&client_id=local-cli`

      shell.openExternal(authUrl)

      // Timeout after 5 minutes.
      setTimeout(() => {
        if (!resolved) {
          resolved = true
          server.close()
          resolve({ success: false, error: 'Login timed out. Please try again.' })
        }
      }, 300_000)
    })

    server.on('error', () => {
      if (!resolved) {
        resolved = true
        resolve({ success: false, error: 'Failed to start callback server.' })
      }
    })
  })
}

function exchangeOAuthCode(code: string): Promise<{ apiKey?: string; error?: string }> {
  return new Promise((resolve) => {
    const body = JSON.stringify({ code, grant_type: 'authorization_code', client_id: 'local-cli' })
    const req = https.request(
      'https://console.anthropic.com/oauth/token',
      {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        timeout: 15000,
      },
      (res) => {
        let data = ''
        res.on('data', (c) => { data += c })
        res.on('end', () => {
          try {
            const json = JSON.parse(data)
            if (json.access_token) {
              resolve({ apiKey: json.access_token })
            } else {
              resolve({ error: json.error || 'No access token returned.' })
            }
          } catch {
            resolve({ error: 'Invalid response from auth server.' })
          }
        })
      },
    )
    req.on('error', () => resolve({ error: 'Could not reach auth server.' }))
    req.on('timeout', () => { req.destroy(); resolve({ error: 'Auth request timed out.' }) })
    req.write(body)
    req.end()
  })
}

// ---------------------------------------------------------------------------

function findPython(): string {
  if (process.platform === 'win32') return 'python'

  // macOS GUI apps get a minimal PATH that only includes /usr/bin/python3
  // (often Python 3.9), but local-cli requires Python 3.10+.
  // Check common Homebrew/system paths for a modern Python.
  const candidates = [
    '/opt/homebrew/bin/python3',   // macOS ARM Homebrew
    '/usr/local/bin/python3',      // macOS Intel Homebrew / Linux
    'python3',                     // fallback to PATH
  ]
  for (const candidate of candidates) {
    try {
      const { execFileSync } = require('child_process')
      execFileSync(candidate, ['--version'], { stdio: 'ignore' })
      return candidate
    } catch {
      // Not found, try next.
    }
  }
  return 'python3'
}

function findProjectRoot(): string {
  // In dev mode: __dirname is desktop/dist-electron, so ../../ is project root.
  // In packaged app: local_cli/ and pyproject.toml are in Resources/ via extraResources.
  if (app.isPackaged && process.resourcesPath) {
    return path.resolve(process.resourcesPath)
  }
  return path.resolve(__dirname, '..', '..')
}

function startPythonServer() {
  const pythonCmd = findPython()
  const projectRoot = findProjectRoot()

  console.log(`Starting Python server: ${pythonCmd} -m local_cli --server`)
  console.log(`Working directory: ${projectRoot}`)

  const env: Record<string, string> = { ...process.env as Record<string, string>, PYTHONUNBUFFERED: '1' }
  if (storedApiKey) {
    env.ANTHROPIC_API_KEY = storedApiKey
  }

  pythonProcess = spawn(pythonCmd, ['-m', 'local_cli', '--server'], {
    cwd: projectRoot,
    stdio: ['pipe', 'pipe', 'pipe'],
    env,
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
  return !!process.env.ANTHROPIC_API_KEY || !!storedApiKey
})

ipcMain.handle('get-claude-auth', () => {
  const auth = loadClaudeAuth()
  if (!auth) return { method: null, keyHint: null, authenticated: false }
  const hint = auth.key ? auth.key.slice(0, 10) + '...' + auth.key.slice(-4) : null
  return { method: auth.method, keyHint: hint, authenticated: true }
})

ipcMain.handle('save-claude-key', async (_event, key: string) => {
  const validation = await validateApiKey(key)
  if (!validation.valid) {
    return { success: false, error: validation.error }
  }
  saveClaudeAuth('api_key', key)
  storedApiKey = key
  process.env.ANTHROPIC_API_KEY = key
  // Notify Python backend about the new key.
  sendToPython({ type: 'set_api_key', api_key: key })
  const hint = key.slice(0, 10) + '...' + key.slice(-4)
  return { success: true, keyHint: hint }
})

ipcMain.handle('delete-claude-auth', () => {
  deleteClaudeAuth()
  delete process.env.ANTHROPIC_API_KEY
  return { success: true }
})

ipcMain.handle('start-claude-oauth', async () => {
  return await startOAuthFlow()
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
  initClaudeAuth()
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
