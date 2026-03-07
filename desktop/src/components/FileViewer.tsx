import { useState, useEffect } from 'react'
import type { ReadFileResult } from '../types'

const BINARY_EXTENSIONS = new Set([
  'png', 'jpg', 'jpeg', 'gif', 'ico', 'exe', 'dll', 'so', 'dylib',
  'zip', 'tar', 'gz', 'pdf', 'woff', 'woff2', 'ttf', 'eot',
])

type Props = {
  filePath: string
  onClose: () => void
}

export function FileViewer({ filePath, onClose }: Props) {
  const [content, setContent] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const ext = filePath.split('.').pop()?.toLowerCase() || ''
  const isBinary = BINARY_EXTENSIONS.has(ext)

  // Split path into directory and basename.
  const lastSlash = filePath.lastIndexOf('/')
  const dir = lastSlash >= 0 ? filePath.slice(0, lastSlash + 1) : ''
  const basename = lastSlash >= 0 ? filePath.slice(lastSlash + 1) : filePath

  useEffect(() => {
    if (isBinary) {
      setLoading(false)
      setContent(null)
      setError(null)
      return
    }

    setLoading(true)
    setContent(null)
    setError(null)

    window.api.readFile(filePath)
      .then((result: ReadFileResult) => {
        if ('error' in result && result.error === 'too_large') {
          const sizeMB = (result.size / (1024 * 1024)).toFixed(1)
          setError(`File too large to preview (${sizeMB} MB)`)
        } else if ('content' in result && result.content !== undefined) {
          setContent(result.content)
        }
        setLoading(false)
      })
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : String(err)
        setError(msg)
        setLoading(false)
      })
  }, [filePath, isBinary])

  return (
    <div className="file-viewer">
      <div className="file-viewer-header">
        <div className="file-viewer-path">
          {dir && <span className="file-viewer-dir">{dir}</span>}
          <span className="file-viewer-name">{basename}</span>
        </div>
        <button className="file-viewer-close" onClick={onClose} title="Close">
          &times;
        </button>
      </div>

      <div className="file-viewer-content">
        {loading && (
          <div className="file-viewer-status">Loading...</div>
        )}

        {!loading && isBinary && (
          <div className="file-viewer-status">Binary file — cannot preview</div>
        )}

        {!loading && error && (
          <div className="file-viewer-status">{error}</div>
        )}

        {!loading && !isBinary && !error && content !== null && (
          <FileContent content={content} />
        )}
      </div>
    </div>
  )
}

function FileContent({ content }: { content: string }) {
  const lines = content.split('\n')
  // Width of the gutter based on number of lines.
  const gutterWidth = String(lines.length).length

  return (
    <div className="file-viewer-lines">
      {lines.map((line, i) => (
        <div key={i} className="file-viewer-line">
          <span className="file-viewer-ln">
            {String(i + 1).padStart(gutterWidth)}
          </span>
          <span className="file-viewer-code">{line || '\n'}</span>
        </div>
      ))}
    </div>
  )
}
