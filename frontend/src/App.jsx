import { useState, useRef, useCallback } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// ── Loading Steps ──
const STEPS = [
  { key: 'upload', label: 'Uploading protein structure...' },
  { key: 'p2rank', label: 'Detecting binding pockets (P2Rank)...' },
  { key: 'generate', label: 'Generating molecules (Flow Matching)...' },
  { key: 'bonds', label: 'Reconstructing bonds & SMILES...' },
]

export default function App() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingStep, setLoadingStep] = useState(0)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [copied, setCopied] = useState(false)
  const fileInputRef = useRef(null)

  // ── File Handling ──
  const handleFile = useCallback((f) => {
    if (f && f.name.endsWith('.pdb')) {
      setFile(f)
      setError(null)
      setResult(null)
    } else {
      setError('Please upload a valid .pdb file')
    }
  }, [])

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const f = e.dataTransfer.files[0]
    handleFile(f)
  }, [handleFile])

  const onDragOver = useCallback((e) => {
    e.preventDefault()
    setDragOver(true)
  }, [])

  const onDragLeave = useCallback(() => setDragOver(false), [])

  // ── Generate ──
  const handleGenerate = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    setLoadingStep(0)

    // Simulate step progression
    const stepTimer = setInterval(() => {
      setLoadingStep((prev) => Math.min(prev + 1, STEPS.length - 1))
    }, 3000)

    try {
      const formData = new FormData()
      formData.append('pdb_file', file)

      const res = await fetch(`${API_URL}/api/generate`, {
        method: 'POST',
        body: formData,
      })

      clearInterval(stepTimer)

      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail || `Server error (${res.status})`)
      }

      const data = await res.json()
      setResult(data)
      setLoadingStep(STEPS.length)
    } catch (err) {
      clearInterval(stepTimer)
      setError(err.message || 'Failed to connect to server')
    } finally {
      setLoading(false)
    }
  }

  // ── Copy SMILES ──
  const copySmiles = () => {
    if (result?.smiles) {
      navigator.clipboard.writeText(result.smiles)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }
  }

  // ── Reset ──
  const reset = () => {
    setFile(null)
    setResult(null)
    setError(null)
    setLoading(false)
    setLoadingStep(0)
  }

  return (
    <>
      <div className="app-bg" />
      <div className="app">
        {/* Header */}
        <header className="header">
          <span className="header__icon">🧬</span>
          <h1 className="header__title">SBDD Drug Designer</h1>
          <p className="header__subtitle">
            Upload a protein structure. AI designs a drug molecule that fits its binding pocket.
          </p>
        </header>

        {/* Main Card */}
        <main className="glass-card" id="main-card">
          {!loading && !result && (
            <>
              {/* Upload Zone */}
              <div
                id="upload-zone"
                className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
                onClick={() => fileInputRef.current?.click()}
                onDrop={onDrop}
                onDragOver={onDragOver}
                onDragLeave={onDragLeave}
              >
                <span className="upload-zone__icon">📂</span>
                <div className="upload-zone__text">
                  {file ? 'File ready' : 'Drop your .pdb file here'}
                </div>
                <div className="upload-zone__hint">
                  {file ? '' : 'or click to browse'}
                </div>
                {file && (
                  <div className="upload-zone__file-info">
                    📄 {file.name} ({(file.size / 1024).toFixed(1)} KB)
                  </div>
                )}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdb"
                  style={{ display: 'none' }}
                  onChange={(e) => handleFile(e.target.files[0])}
                  id="file-input"
                />
              </div>

              {/* Error */}
              {error && <div className="error-box" id="error-box">⚠️ {error}</div>}

              {/* Generate Button */}
              <button
                id="btn-generate"
                className="btn-generate"
                onClick={handleGenerate}
                disabled={!file}
              >
                🚀 Generate Drug Molecule
              </button>
            </>
          )}

          {/* Loading State */}
          {loading && (
            <div className="loading-container" id="loading-state">
              <div className="loading-spinner">
                <div className="loading-spinner__ring" />
                <div className="loading-spinner__ring" />
                <div className="loading-spinner__ring" />
              </div>
              <p style={{ color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                Designing your molecule...
              </p>
              <ul className="loading-steps">
                {STEPS.map((step, i) => (
                  <li
                    key={step.key}
                    className={`loading-step ${
                      i < loadingStep ? 'done' : i === loadingStep ? 'active' : ''
                    }`}
                  >
                    <span className="loading-step__icon">
                      {i < loadingStep ? '✅' : i === loadingStep ? '⏳' : '○'}
                    </span>
                    {step.label}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Results */}
          {result && (
            <div className="results" id="results-view">
              {/* SMILES */}
              <div className="results__section">
                <div className="results__label">Generated SMILES</div>
                <div className="results__smiles-box">
                  <span className="results__smiles-text" id="smiles-output">
                    {result.smiles}
                  </span>
                  <button className="btn-copy" onClick={copySmiles} id="btn-copy">
                    {copied ? '✓ Copied' : '📋 Copy'}
                  </button>
                </div>
              </div>

              {/* Molecular Properties */}
              {result.properties && (
                <div className="results__section">
                  <div className="results__label">Molecular Properties</div>
                  <div className="props-grid">
                    {result.properties.molecular_weight && (
                      <div className="prop-card">
                        <div className="prop-card__value">{result.properties.molecular_weight}</div>
                        <div className="prop-card__label">MW (Da)</div>
                      </div>
                    )}
                    {result.properties.qed !== undefined && (
                      <div className="prop-card">
                        <div className="prop-card__value">{result.properties.qed}</div>
                        <div className="prop-card__label">QED</div>
                      </div>
                    )}
                    {result.properties.logp !== undefined && (
                      <div className="prop-card">
                        <div className="prop-card__value">{result.properties.logp}</div>
                        <div className="prop-card__label">LogP</div>
                      </div>
                    )}
                    {result.properties.hbd !== undefined && (
                      <div className="prop-card">
                        <div className="prop-card__value">{result.properties.hbd}</div>
                        <div className="prop-card__label">H-Bond Donors</div>
                      </div>
                    )}
                    {result.properties.hba !== undefined && (
                      <div className="prop-card">
                        <div className="prop-card__value">{result.properties.hba}</div>
                        <div className="prop-card__label">H-Bond Acceptors</div>
                      </div>
                    )}
                    {result.properties.num_rings !== undefined && (
                      <div className="prop-card">
                        <div className="prop-card__value">{result.properties.num_rings}</div>
                        <div className="prop-card__label">Rings</div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Pocket Info */}
              {result.pocket && (
                <div className="results__section">
                  <div className="results__label">Detected Pocket</div>
                  <div className="pocket-info">
                    <span className="pocket-badge">
                      🎯 Rank #{result.pocket.rank}
                    </span>
                    <span className="pocket-badge">
                      ⭐ Score: {result.pocket.score?.toFixed(2)}
                    </span>
                    <span className="pocket-badge">
                      🧩 {result.pocket.num_residues} residues
                    </span>
                    <span className="pocket-badge">
                      📍 {result.pocket.total_pockets_found} pockets found
                    </span>
                  </div>
                </div>
              )}

              {/* Generation Stats */}
              {result.stats && (
                <div className="results__section">
                  <div className="results__label">Generation Statistics</div>
                  <div className="stats-bar">
                    <div className="stat-item">
                      <div className="stat-item__value">{result.stats.valid_count}</div>
                      <div className="stat-item__label">Valid</div>
                    </div>
                    <div className="stat-item">
                      <div className="stat-item__value">{result.stats.total_generated}</div>
                      <div className="stat-item__label">Generated</div>
                    </div>
                    <div className="stat-item">
                      <div className="stat-item__value">{result.stats.validity_rate}%</div>
                      <div className="stat-item__label">Success Rate</div>
                    </div>
                    {result.timings && (
                      <div className="stat-item">
                        <div className="stat-item__value">
                          {Object.values(result.timings).reduce((a, b) => a + b, 0).toFixed(1)}s
                        </div>
                        <div className="stat-item__label">Total Time</div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* All Generated SMILES */}
              {result.all_smiles && result.all_smiles.length > 1 && (
                <div className="results__section">
                  <div className="results__label">
                    All Valid Candidates ({result.all_smiles.length})
                  </div>
                  <div className="smiles-list">
                    {result.all_smiles.map((s, i) => (
                      <div key={i} className="smiles-list__item">
                        {i + 1}. {s}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Reset */}
              <button className="btn-reset" onClick={reset} id="btn-reset">
                ↩ Design Another Molecule
              </button>
            </div>
          )}
        </main>

        {/* Footer */}
        <footer style={{
          marginTop: '2rem',
          fontSize: '0.75rem',
          color: 'var(--text-muted)',
          textAlign: 'center',
        }}>
          SBDD Drug Designer • RL-Guided Flow Diffusion • K-HUB
        </footer>
      </div>
    </>
  )
}
