import { app, BrowserWindow, ipcMain } from 'electron'
import { spawn, ChildProcess } from 'child_process'
import path from 'path'

let mainWindow: BrowserWindow | null = null
let pythonProcess: ChildProcess | null = null
let lineBuffer = ''

function findPython(): string {
  // Try python3 first, then python.
  return process.platform === 'win32' ? 'python' : 'python3'
}

function startPythonServer() {
  const pythonCmd = findPython()
  const projectRoot = path.resolve(__dirname, '..', '..')

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
        mainWindow?.webContents.send('python-message', msg)
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
