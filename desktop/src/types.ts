export type MessageRole = 'user' | 'assistant' | 'tool' | 'system'

export type ToolCall = {
  name: string
  args: Record<string, unknown>
}

export type ToolResult = {
  name: string
  output: string
}

export type Message = {
  id: string
  role: MessageRole
  content: string
  toolCalls?: ToolCall[]
  toolResults?: ToolResult[]
  streaming?: boolean
}

export type AppStatus = {
  model: string
  provider: string
  connected: boolean
  tools: string[]
  ready: boolean
}

export type FileEntry = {
  name: string
  isDirectory: boolean
  isSymlink: boolean
}

export type ReadFileResult =
  | { content: string; error?: never; size?: never }
  | { error: 'too_large'; size: number; content?: never }

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
  has_claude?: boolean
  has_updates?: boolean
  success?: boolean
}

declare global {
  interface Window {
    api: {
      sendToPython: (data: object) => void
      onPythonMessage: (cb: (msg: PythonMessage) => void) => () => void
      onPythonStderr: (cb: (text: string) => void) => () => void
      onPythonExit: (cb: (code: number | null) => void) => () => void
      getPythonStatus: () => Promise<{ running: boolean }>
      listDir: (dirPath: string) => Promise<FileEntry[]>
      readFile: (filePath: string) => Promise<ReadFileResult>
      openDirectoryDialog: () => Promise<Electron.OpenDialogReturnValue>
      getHomeDir: () => Promise<string>
      hasClaudeAccess: () => Promise<boolean>
    }
  }
}
