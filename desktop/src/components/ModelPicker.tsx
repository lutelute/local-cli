import { useState, useRef, useEffect, useCallback } from 'react'

export type CatalogModel = {
  name: string
  display: string
  category: string
  params: string
  size_gb: number
  description: string
  tags: string[]
  installed: boolean
}

export type SearchResult = {
  name: string
  description: string
  pulls: number
  pulls_display: string
  tags: string[]
  sizes: string[]
  installed: boolean
  cloud_only: boolean
}

type Props = {
  catalog: { categories: string[]; models: CatalogModel[] } | null
  searchResults: SearchResult[]
  current: string
  onSelect: (model: string) => void
  onPull: (model: string) => void
  onDelete: (model: string) => void
  onSearch: (query: string, sort: string, capability: string) => void
  onUpdate: () => void
  onClose: () => void
  pulling: string | null
  pullProgress: string
  searching: boolean
  updating: boolean
}

type ViewMode = 'catalog' | 'discover'

export function ModelPicker({
  catalog, searchResults, current, onSelect, onPull, onDelete,
  onSearch, onUpdate, onClose, pulling, pullProgress, searching, updating,
}: Props) {
  const [filter, setFilter] = useState('')
  const [activeCategory, setActiveCategory] = useState<string | null>(null)
  const [view, setView] = useState<ViewMode>('catalog')
  const [searchQuery, setSearchQuery] = useState('')
  const [searchSort, setSearchSort] = useState('popular')
  const [searchCap, setSearchCap] = useState('')
  const inputRef = useRef<HTMLInputElement>(null)
  const searchInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (view === 'catalog') inputRef.current?.focus()
    else searchInputRef.current?.focus()
  }, [view])

  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose() }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [onClose])

  // Auto-search on mount for discover tab.
  const didInitialSearch = useRef(false)
  useEffect(() => {
    if (view === 'discover' && !didInitialSearch.current) {
      didInitialSearch.current = true
      onSearch('', 'popular', '')
    }
  }, [view, onSearch])

  const handleSearchSubmit = useCallback(() => {
    onSearch(searchQuery, searchSort, searchCap)
  }, [searchQuery, searchSort, searchCap, onSearch])

  // --- Catalog view ---
  function renderCatalog() {
    if (!catalog) {
      return <div className="picker-empty">Loading models...</div>
    }

    const categories = catalog.categories
    const models = catalog.models
    const lowerFilter = filter.toLowerCase()
    const filtered = models.filter(m => {
      if (activeCategory && m.category !== activeCategory) return false
      if (lowerFilter) {
        return (
          m.name.toLowerCase().includes(lowerFilter) ||
          m.display.toLowerCase().includes(lowerFilter) ||
          m.description.toLowerCase().includes(lowerFilter)
        )
      }
      return true
    })

    const grouped: Record<string, CatalogModel[]> = {}
    for (const m of filtered) {
      if (!grouped[m.category]) grouped[m.category] = []
      grouped[m.category].push(m)
    }

    const displayCategories = activeCategory
      ? [activeCategory]
      : categories.filter(c => grouped[c])

    return (
      <>
        <div className="picker-search">
          <input
            ref={inputRef}
            type="text"
            placeholder="Filter models..."
            value={filter}
            onChange={e => setFilter(e.target.value)}
          />
        </div>
        <div className="picker-tabs">
          <button
            className={`picker-tab ${!activeCategory ? 'active' : ''}`}
            onClick={() => setActiveCategory(null)}
          >All</button>
          {categories.map(cat => (
            <button
              key={cat}
              className={`picker-tab ${activeCategory === cat ? 'active' : ''}`}
              onClick={() => setActiveCategory(cat)}
            >{cat}</button>
          ))}
        </div>
        <div className="picker-list">
          {displayCategories.map(cat => (
            <div key={cat}>
              {!activeCategory && <div className="picker-category">{cat}</div>}
              {grouped[cat]?.map(m => (
                <CatalogRow
                  key={m.name}
                  name={m.name}
                  display={m.display}
                  desc={m.description}
                  size={`${m.size_gb} GB`}
                  params={m.params}
                  installed={m.installed}
                  isCurrent={m.name === current}
                  isPulling={pulling === m.name}
                  onSelect={() => m.installed && onSelect(m.name)}
                  onPull={() => onPull(m.name)}
                  onDelete={() => onDelete(m.name)}
                />
              ))}
            </div>
          ))}
          {filtered.length === 0 && <div className="picker-empty">No models match.</div>}
        </div>
      </>
    )
  }

  // --- Discover view ---
  function renderDiscover() {
    return (
      <>
        <div className="picker-search">
          <input
            ref={searchInputRef}
            type="text"
            placeholder="Search ollama.com..."
            value={searchQuery}
            onChange={e => setSearchQuery(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') handleSearchSubmit() }}
          />
        </div>
        <div className="picker-tabs">
          {(['popular', 'hot', 'newest'] as const).map(s => (
            <button
              key={s}
              className={`picker-tab ${searchSort === s ? 'active' : ''}`}
              onClick={() => { setSearchSort(s); onSearch(searchQuery, s, searchCap) }}
            >{s}</button>
          ))}
          <span className="picker-tab-sep" />
          {([['', 'all'], ['tools', 'tools'], ['code', 'code'], ['vision', 'vision'], ['thinking', 'reasoning']] as const).map(([val, label]) => (
            <button
              key={val}
              className={`picker-tab ${searchCap === val ? 'active' : ''}`}
              onClick={() => { setSearchCap(val as string); onSearch(searchQuery, searchSort, val as string) }}
            >{label}</button>
          ))}
        </div>
        {searching && <div className="pull-bar"><span className="pull-bar-text">Searching...</span></div>}
        <div className="picker-list">
          {searchResults.map(m => (
            <SearchRow
              key={m.name}
              model={m}
              isCurrent={m.name === current || current.startsWith(m.name + ':')}
              isPulling={pulling === m.name}
              onSelect={() => m.installed && onSelect(m.name)}
              onPull={() => onPull(m.name)}
            />
          ))}
          {!searching && searchResults.length === 0 && (
            <div className="picker-empty">No results. Try a different search.</div>
          )}
        </div>
      </>
    )
  }

  const installedCount = catalog?.models.filter(m => m.installed).length ?? 0

  return (
    <div className="overlay" onClick={onClose}>
      <div className="picker picker-wide" onClick={e => e.stopPropagation()}>
        <div className="picker-header">
          <div className="picker-view-tabs">
            <button
              className={`picker-view-tab ${view === 'catalog' ? 'active' : ''}`}
              onClick={() => setView('catalog')}
            >Catalog</button>
            <button
              className={`picker-view-tab ${view === 'discover' ? 'active' : ''}`}
              onClick={() => setView('discover')}
            >Discover</button>
          </div>
          <span className="picker-header-sub">
            {installedCount} installed
            <button
              className={`picker-update-btn ${updating ? 'updating' : ''}`}
              onClick={onUpdate}
              disabled={updating}
              title="Fetch latest models from ollama.com"
            >
              {updating ? 'updating...' : 'sync'}
            </button>
          </span>
        </div>

        {pulling && (
          <div className="pull-bar">
            <span className="pull-bar-text">Downloading {pulling}...</span>
            <span className="pull-bar-progress">{pullProgress}</span>
          </div>
        )}

        {view === 'catalog' ? renderCatalog() : renderDiscover()}
      </div>
    </div>
  )
}

// --- Catalog row ---
function CatalogRow({ name, display, desc, size, params, installed, isCurrent, isPulling, onSelect, onPull, onDelete }: {
  name: string; display: string; desc: string; size: string; params: string
  installed: boolean; isCurrent: boolean; isPulling: boolean
  onSelect: () => void; onPull: () => void; onDelete: () => void
}) {
  const [confirmDelete, setConfirmDelete] = useState(false)

  return (
    <div
      className={`picker-model ${isCurrent ? 'current' : ''} ${installed ? 'installed' : ''}`}
      onClick={installed ? onSelect : undefined}
    >
      <div className="picker-model-main">
        <div className="picker-model-name">
          {display}
          {params && <span className="picker-model-params">{params}</span>}
        </div>
        <div className="picker-model-desc">{desc}</div>
      </div>
      <div className="picker-model-right">
        <span className="picker-model-size">{size}</span>
        {isCurrent ? (
          <span className="picker-badge current-badge">active</span>
        ) : installed ? (
          <div className="picker-model-actions">
            <button className="picker-action use" onClick={e => { e.stopPropagation(); onSelect() }}>Use</button>
            {!confirmDelete ? (
              <button className="picker-action del" onClick={e => {
                e.stopPropagation(); setConfirmDelete(true)
                setTimeout(() => setConfirmDelete(false), 3000)
              }}>Del</button>
            ) : (
              <button className="picker-action del confirm" onClick={e => {
                e.stopPropagation(); setConfirmDelete(false); onDelete()
              }}>Sure?</button>
            )}
          </div>
        ) : isPulling ? (
          <span className="picker-badge pulling-badge">pulling...</span>
        ) : (
          <button className="picker-action download" onClick={e => { e.stopPropagation(); onPull() }}>Download</button>
        )}
      </div>
    </div>
  )
}

// --- Search result row ---
function SearchRow({ model, isCurrent, isPulling, onSelect, onPull }: {
  model: SearchResult; isCurrent: boolean; isPulling: boolean
  onSelect: () => void; onPull: () => void
}) {
  return (
    <div
      className={`picker-model ${isCurrent ? 'current' : ''} ${model.installed ? 'installed' : ''}`}
      onClick={model.installed ? onSelect : undefined}
    >
      <div className="picker-model-main">
        <div className="picker-model-name">
          {model.name}
          {model.tags.map(t => (
            <span key={t} className="picker-model-tag">{t}</span>
          ))}
        </div>
        <div className="picker-model-desc">{model.description}</div>
        <div className="picker-model-meta">
          <span className="picker-model-pulls">{model.pulls_display} pulls</span>
          {model.sizes.length > 0 && (
            <span className="picker-model-sizes">{model.sizes.join(', ')}</span>
          )}
        </div>
      </div>
      <div className="picker-model-right">
        {isCurrent ? (
          <span className="picker-badge current-badge">active</span>
        ) : model.installed ? (
          <button className="picker-action use" onClick={e => { e.stopPropagation(); onSelect() }}>Use</button>
        ) : model.cloud_only ? (
          <span className="picker-badge cloud-badge">cloud</span>
        ) : isPulling ? (
          <span className="picker-badge pulling-badge">pulling...</span>
        ) : (
          <button className="picker-action download" onClick={e => { e.stopPropagation(); onPull() }}>Download</button>
        )}
      </div>
    </div>
  )
}
