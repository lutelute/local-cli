import { useState, useCallback, useEffect } from 'react'
import type { FileEntry } from '../types'

const ENTRIES_PER_PAGE = 500

type TreeNode = {
  name: string
  path: string
  isDirectory: boolean
  isSymlink: boolean
  expanded: boolean
  loading: boolean
  children: TreeNode[] | null
  error: string | null
}

type Props = {
  rootDir: string | null
  onFileSelect: (path: string) => void
  onRootChange: (path: string) => void
}

export function FileExplorer({ rootDir, onFileSelect, onRootChange }: Props) {
  const [nodes, setNodes] = useState<TreeNode[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const loadDirectory = useCallback(async (dirPath: string): Promise<TreeNode[]> => {
    const entries: FileEntry[] = await window.api.listDir(dirPath)
    // Sort: directories first, then files, alphabetical within each group.
    const sorted = entries.slice().sort((a, b) => {
      if (a.isDirectory !== b.isDirectory) return a.isDirectory ? -1 : 1
      return a.name.localeCompare(b.name)
    })
    return sorted.map(entry => ({
      name: entry.name,
      path: dirPath + '/' + entry.name,
      isDirectory: entry.isDirectory,
      isSymlink: entry.isSymlink,
      expanded: false,
      loading: false,
      children: null,
      error: null,
    }))
  }, [])

  // Load root directory when rootDir changes.
  useEffect(() => {
    if (!rootDir) {
      setNodes([])
      return
    }
    setLoading(true)
    setError(null)
    loadDirectory(rootDir)
      .then(children => {
        setNodes(children)
        setLoading(false)
      })
      .catch(err => {
        const msg = err instanceof Error ? err.message : String(err)
        setError(msg.includes('EACCES') || msg.includes('permission')
          ? `Access denied: ${rootDir}`
          : `Failed to read: ${rootDir}`)
        setLoading(false)
      })
  }, [rootDir, loadDirectory])

  const handleOpenFolder = useCallback(async () => {
    const result = await window.api.openDirectoryDialog()
    if (!result.canceled && result.filePaths[0]) {
      onRootChange(result.filePaths[0])
    }
  }, [onRootChange])

  const updateNodeAtPath = useCallback(
    (nodePath: string, updater: (node: TreeNode) => TreeNode) => {
      function updateNodes(items: TreeNode[]): TreeNode[] {
        return items.map(node => {
          if (node.path === nodePath) return updater(node)
          if (node.children && nodePath.startsWith(node.path + '/')) {
            return { ...node, children: updateNodes(node.children) }
          }
          return node
        })
      }
      setNodes(prev => updateNodes(prev))
    },
    []
  )

  const handleToggleFolder = useCallback(async (node: TreeNode) => {
    if (node.expanded) {
      // Collapse.
      updateNodeAtPath(node.path, n => ({ ...n, expanded: false }))
      return
    }

    // Don't auto-expand symlink directories.
    if (node.isSymlink) {
      return
    }

    // Expand — load children if not yet loaded.
    if (node.children === null) {
      updateNodeAtPath(node.path, n => ({ ...n, loading: true, error: null }))
      try {
        const children = await loadDirectory(node.path)
        updateNodeAtPath(node.path, n => ({
          ...n,
          children,
          expanded: true,
          loading: false,
        }))
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err)
        const errorMsg = msg.includes('EACCES') || msg.includes('permission')
          ? 'Access denied'
          : 'Failed to read'
        updateNodeAtPath(node.path, n => ({
          ...n,
          error: errorMsg,
          loading: false,
        }))
      }
    } else {
      updateNodeAtPath(node.path, n => ({ ...n, expanded: true }))
    }
  }, [loadDirectory, updateNodeAtPath])

  const handleFileClick = useCallback((node: TreeNode) => {
    onFileSelect(node.path)
  }, [onFileSelect])

  // No root selected — show open folder prompt.
  if (!rootDir) {
    return (
      <div className="explorer">
        <div className="explorer-header">
          <span className="explorer-title">Explorer</span>
          <button className="explorer-open-btn-small" onClick={handleOpenFolder} title="Open folder">
            +
          </button>
        </div>
        <div className="explorer-empty">
          <button className="explorer-open-btn" onClick={handleOpenFolder}>
            Open Folder
          </button>
          <span className="explorer-hint">Select a working directory</span>
        </div>
      </div>
    )
  }

  const rootName = rootDir.split('/').pop() || rootDir

  return (
    <div className="explorer">
      <div className="explorer-header">
        <span className="explorer-title" title={rootDir}>{rootName}</span>
        <button className="explorer-open-btn-small" onClick={handleOpenFolder} title="Change working directory">
          +
        </button>
      </div>
      <div className="explorer-path-bar" title={rootDir} onClick={handleOpenFolder}>
        {rootDir}
      </div>
      <div className="explorer-tree">
        {loading && <div className="explorer-loading">Loading...</div>}
        {error && <div className="explorer-error">{error}</div>}
        {!loading && !error && nodes.length === 0 && (
          <div className="explorer-empty-dir">Empty directory</div>
        )}
        <NodeList
          nodes={nodes}
          depth={0}
          onToggleFolder={handleToggleFolder}
          onFileClick={handleFileClick}
        />
      </div>
    </div>
  )
}

function NodeList({ nodes, depth, onToggleFolder, onFileClick }: {
  nodes: TreeNode[]
  depth: number
  onToggleFolder: (node: TreeNode) => void
  onFileClick: (node: TreeNode) => void
}) {
  const [visibleCount, setVisibleCount] = useState(ENTRIES_PER_PAGE)

  // Reset visible count when nodes change (e.g. parent re-expanded).
  useEffect(() => {
    setVisibleCount(ENTRIES_PER_PAGE)
  }, [nodes])

  const visible = nodes.slice(0, visibleCount)
  const remaining = nodes.length - visibleCount

  return (
    <>
      {visible.map(node => (
        <TreeNodeRow
          key={node.path}
          node={node}
          depth={depth}
          onToggleFolder={onToggleFolder}
          onFileClick={onFileClick}
        />
      ))}
      {remaining > 0 && (
        <div
          className="explorer-show-more"
          style={{ paddingLeft: (depth + 1) * 16 + 4 }}
          onClick={() => setVisibleCount(prev => prev + ENTRIES_PER_PAGE)}
        >
          {remaining} more items...
        </div>
      )}
    </>
  )
}

function TreeNodeRow({ node, depth, onToggleFolder, onFileClick }: {
  node: TreeNode
  depth: number
  onToggleFolder: (node: TreeNode) => void
  onFileClick: (node: TreeNode) => void
}) {
  const indent = depth * 16

  const handleClick = () => {
    if (node.isDirectory) {
      onToggleFolder(node)
    } else {
      onFileClick(node)
    }
  }

  const icon = getNodeIcon(node)
  const arrow = node.isDirectory
    ? (node.isSymlink ? '\u00B7' : node.expanded ? '\u25BE' : '\u25B8')
    : ' '

  return (
    <>
      <div
        className={`explorer-node ${node.isDirectory ? 'dir' : 'file'} ${node.isSymlink ? 'symlink' : ''}`}
        style={{ paddingLeft: indent + 4 }}
        onClick={handleClick}
        title={node.isSymlink ? `${node.path} (symlink)` : node.path}
      >
        <span className="explorer-arrow">{arrow}</span>
        <span className="explorer-icon">{icon}</span>
        <span className="explorer-name">{node.name}</span>
        {node.isSymlink && <span className="explorer-symlink-badge">\u2192</span>}
      </div>
      {node.loading && (
        <div className="explorer-loading" style={{ paddingLeft: indent + 20 }}>
          Loading...
        </div>
      )}
      {node.error && (
        <div className="explorer-error" style={{ paddingLeft: indent + 20 }}>
          {node.error}
        </div>
      )}
      {node.expanded && node.children && (
        <NodeList
          nodes={node.children}
          depth={depth + 1}
          onToggleFolder={onToggleFolder}
          onFileClick={onFileClick}
        />
      )}
    </>
  )
}

function getNodeIcon(node: TreeNode): string {
  if (node.isDirectory) {
    return node.expanded ? '\uD83D\uDCC2' : '\uD83D\uDCC1'
  }
  const ext = node.name.split('.').pop()?.toLowerCase() || ''
  switch (ext) {
    case 'ts':
    case 'tsx':
    case 'js':
    case 'jsx':
      return '\uD83D\uDFE8' // yellow square
    case 'py':
      return '\uD83D\uDFE9' // green square
    case 'json':
    case 'yaml':
    case 'yml':
    case 'toml':
      return '\u2699'  // gear
    case 'md':
    case 'txt':
    case 'rst':
      return '\uD83D\uDCC4' // page
    case 'css':
    case 'scss':
    case 'less':
      return '\uD83C\uDFA8' // palette
    case 'html':
      return '\uD83C\uDF10' // globe
    case 'sh':
    case 'bash':
    case 'zsh':
      return '\u25B6'  // play
    case 'png':
    case 'jpg':
    case 'jpeg':
    case 'gif':
    case 'svg':
    case 'ico':
      return '\uD83D\uDDBC' // frame
    default:
      return '\uD83D\uDCC4' // page
  }
}
