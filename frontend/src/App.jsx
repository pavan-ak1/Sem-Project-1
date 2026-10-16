import React, { useState, useEffect, useRef } from 'react';
import {
  Activity,
  Layers,
  Play,
  Pause,
  ChevronLeft,
  ChevronRight,
  Upload,
  FileText,
  ShieldCheck,
  Cpu,
  BarChart2,
  Maximize2,
  Download,
  Copy,
  Check,
  AlertTriangle,
  RotateCw,
  Eye,
  Sliders,
  Folder,
  FolderPlus
} from 'lucide-react';
import './App.css';

export default function App() {
  // Navigation & View State
  const [activeTab, setActiveTab] = useState('explorer'); // 'explorer' | 'upload'
  const [health, setHealth] = useState(null);
  const [cases, setCases] = useState([]);
  const [selectedCaseId, setSelectedCaseId] = useState('case_00110');
  
  // Slice Viewer State
  const [sliceIndex, setSliceIndex] = useState(15);
  const [totalSlices, setTotalSlices] = useState(34);
  const [sliceData, setSliceData] = useState(null);
  const [loadingSlice, setLoadingSlice] = useState(false);
  const [viewMode, setViewMode] = useState('ai_overlay'); // 'ai_overlay' | 'gt_overlay' | 'split' | 'pure_ct'
  const [opacity, setOpacity] = useState(0.65);
  const [visibleLayers, setVisibleLayers] = useState({ kidney: true, tumor: true, cyst: true });
  const [isPlaying, setIsPlaying] = useState(false);

  // 3D Case Analytics State
  const [metrics3D, setMetrics3D] = useState(null);
  const [isComputing3D, setIsComputing3D] = useState(false);

  // Upload State (Supports both Whole Case Folder & Single CT Slice)
  const [uploadMode, setUploadMode] = useState('case'); // 'case' | 'slice'
  const [uploadedCaseResult, setUploadedCaseResult] = useState(null);
  const [uploadedCaseSliceIndex, setUploadedCaseSliceIndex] = useState(0);
  const [uploadedSliceDetail, setUploadedSliceDetail] = useState(null);
  const [loadingUploadedSlice, setLoadingUploadedSlice] = useState(false);
  const [isUploadedPlaying, setIsUploadedPlaying] = useState(false);
  const [singleSliceResult, setSingleSliceResult] = useState(null);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgressText, setUploadProgressText] = useState('');
  const [dragActive, setDragActive] = useState(false);

  // Modal State
  const [reportModalOpen, setReportModalOpen] = useState(false);
  const [reportMarkdown, setReportMarkdown] = useState('');
  const [copiedReport, setCopiedReport] = useState(false);

  const fileInputRef = useRef(null);
  const folderInputRef = useRef(null);
  const multiFileInputRef = useRef(null);

  // 1. Initial Health & Case Cohort Fetch
  useEffect(() => {
    fetch('/api/health')
      .then(res => res.json())
      .then(data => setHealth(data))
      .catch(err => console.error("Health check error:", err));

    fetch('/api/cases')
      .then(res => res.json())
      .then(data => {
        if (data && data.length > 0) {
          setCases(data);
          setSelectedCaseId(data[0].case_id);
          setTotalSlices(data[0].slice_count);
        }
      })
      .catch(err => console.error("Fetch cases error:", err));
  }, []);

  // 2. Fetch Case Slices list when selectedCaseId changes
  useEffect(() => {
    if (!selectedCaseId) return;
    fetch(`/api/case/${selectedCaseId}/slices`)
      .then(res => res.json())
      .then(data => {
        if (data && data.slices) {
          setTotalSlices(data.total_slices);
          setSliceIndex(Math.floor(data.total_slices / 2));
        }
      })
      .catch(err => console.error("Error fetching slices list:", err));

    // Check if 3D metrics are already computed or cached
    fetch(`/api/case/${selectedCaseId}/metrics`)
      .then(res => {
        if (res.ok) return res.json();
        return null;
      })
      .then(cached => {
        if (cached && cached.tnm_stage) {
          setMetrics3D(cached);
        } else {
          setMetrics3D(null);
        }
      })
      .catch(() => setMetrics3D(null));
  }, [selectedCaseId]);

  // 3. Fetch specific slice detail
  useEffect(() => {
    if (!selectedCaseId) return;
    setLoadingSlice(true);
    fetch(`/api/case/${selectedCaseId}/slice/${sliceIndex}`)
      .then(res => res.json())
      .then(data => {
        setSliceData(data);
        setLoadingSlice(false);
      })
      .catch(err => {
        console.error("Error fetching slice:", err);
        setLoadingSlice(false);
      });
  }, [selectedCaseId, sliceIndex]);

  // 4. Cine Loop Playback
  useEffect(() => {
    let interval = null;
    if (isPlaying) {
      interval = setInterval(() => {
        setSliceIndex(prev => (prev + 1 >= totalSlices ? 0 : prev + 1));
      }, 140);
    }
    return () => clearInterval(interval);
  }, [isPlaying, totalSlices]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (activeTab !== 'explorer') return;
      if (e.key === 'ArrowRight') {
        setSliceIndex(prev => Math.min(totalSlices - 1, prev + 1));
      } else if (e.key === 'ArrowLeft') {
        setSliceIndex(prev => Math.max(0, prev - 1));
      } else if (e.key === ' ') {
        e.preventDefault();
        setIsPlaying(p => !p);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [totalSlices, activeTab]);

  // 5. Trigger 3D Patient Reconstruction
  const trigger3DReconstruction = (cid) => {
    setIsComputing3D(true);
    fetch(`/api/predict/case/${cid}`, { method: 'POST' })
      .then(res => res.json())
      .then(data => {
        setMetrics3D(data);
        setIsComputing3D(false);
      })
      .catch(err => {
        console.error("3D analysis error:", err);
        setIsComputing3D(false);
      });
  };

  // 6. Drag & Drop Directory / File Traversal
  const extractFilesFromDataTransfer = async (items) => {
    const files = [];
    const entries = [];

    for (let i = 0; i < items.length; i++) {
      const item = items[i];
      if (item.webkitGetAsEntry) {
        const entry = item.webkitGetAsEntry();
        if (entry) entries.push(entry);
      } else if (item.kind === 'file') {
        const f = item.getAsFile();
        if (f) files.push(f);
      }
    }

    const readEntry = async (entry) => {
      if (entry.isFile) {
        return new Promise((resolve) => {
          entry.file((file) => {
            files.push(file);
            resolve();
          }, () => resolve());
        });
      } else if (entry.isDirectory) {
        const dirReader = entry.createReader();
        const readBatch = () => {
          return new Promise((resolve) => {
            dirReader.readEntries(async (batchEntries) => {
              if (batchEntries.length === 0) {
                resolve();
              } else {
                for (const sub of batchEntries) {
                  await readEntry(sub);
                }
                await readBatch();
                resolve();
              }
            }, () => resolve());
          });
        };
        await readBatch();
      }
    };

    for (const entry of entries) {
      await readEntry(entry);
    }
    return files;
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    setDragActive(false);

    let droppedFiles = [];
    if (e.dataTransfer.items && e.dataTransfer.items.length > 0) {
      droppedFiles = await extractFilesFromDataTransfer(e.dataTransfer.items);
    } else if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      droppedFiles = Array.from(e.dataTransfer.files);
    }

    if (!droppedFiles || droppedFiles.length === 0) return;

    if (droppedFiles.length > 1) {
      setUploadMode('case');
      handleCaseFolderUpload(droppedFiles);
    } else {
      if (uploadMode === 'case') {
        handleCaseFolderUpload(droppedFiles);
      } else {
        handleSingleSliceUpload(droppedFiles[0]);
      }
    }
  };

  const handleFolderSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleCaseFolderUpload(Array.from(e.target.files));
    }
  };

  const handleMultiFilesSelect = (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleCaseFolderUpload(Array.from(e.target.files));
    }
  };

  const handleSingleFileSelect = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleSingleSliceUpload(e.target.files[0]);
    }
  };

  // 7. Case Folder Upload Handler (3D Pipeline)
  const handleCaseFolderUpload = (filesList) => {
    const validExts = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.dcm'];
    const imageFiles = filesList.filter(f => {
      const name = f.name.toLowerCase();
      return validExts.some(ext => name.endsWith(ext));
    });

    if (imageFiles.length === 0) {
      alert("No valid CT slice images found (supported: PNG, JPG, TIFF).");
      return;
    }

    let detectedName = '';
    if (imageFiles[0].webkitRelativePath) {
      const parts = imageFiles[0].webkitRelativePath.split('/');
      if (parts.length > 1) detectedName = parts[0];
    }
    if (!detectedName) {
      const match = imageFiles[0].name.match(/case_\d+/i);
      if (match) detectedName = match[0].toLowerCase();
    }

    const sliceWord = imageFiles.length === 1 ? 'slice' : 'slices';
    setIsUploading(true);
    setUploadProgressText(`Uploading ${imageFiles.length} CT ${sliceWord} & executing Residual U-Net AI pipeline...`);

    const formData = new FormData();
    if (detectedName) formData.append('case_id', detectedName);
    imageFiles.forEach(f => formData.append('files', f, f.name));

    fetch('/api/predict/case-upload', {
      method: 'POST',
      body: formData
    })
      .then(async res => {
        if (!res.ok) {
          const errDetail = await res.text();
          throw new Error(`HTTP ${res.status}: ${errDetail}`);
        }
        return res.json();
      })
      .then(data => {
        setUploadedCaseResult(data);
        setUploadedCaseSliceIndex(data.best_preview_slice);
        setIsUploading(false);
        setUploadProgressText('');

        if (data.key_preview_slices && data.key_preview_slices.length > 0) {
          const match = data.key_preview_slices.find(s => s.slice_index === data.best_preview_slice);
          if (match) {
            setUploadedSliceDetail(match);
          } else {
            fetchUploadedSlice(data.case_id, data.best_preview_slice);
          }
        } else {
          fetchUploadedSlice(data.case_id, data.best_preview_slice);
        }

        // Refresh cohort cases so uploaded case is available in main explorer
        fetch('/api/cases')
          .then(r => r.json())
          .then(cList => setCases(cList))
          .catch(e => console.error(e));
      })
      .catch(err => {
        console.error("Case upload error:", err);
        alert(`Case upload failed: ${err.message}`);
        setIsUploading(false);
        setUploadProgressText('');
      });
  };

  // 8. Single CT Slice Upload Handler (2D Instant)
  const handleSingleSliceUpload = (file) => {
    if (!file) return;
    setIsUploading(true);
    setUploadProgressText("Analyzing single slice with Residual U-Net...");
    const formData = new FormData();
    formData.append('file', file);

    fetch('/api/predict/slice', {
      method: 'POST',
      body: formData
    })
      .then(async res => {
        if (!res.ok) {
          const errDetail = await res.text();
          throw new Error(`HTTP ${res.status}: ${errDetail}`);
        }
        return res.json();
      })
      .then(data => {
        setSingleSliceResult(data);
        setIsUploading(false);
        setUploadProgressText('');
      })
      .catch(err => {
        console.error("Single slice upload error:", err);
        alert(`Single slice upload failed: ${err.message}`);
        setIsUploading(false);
        setUploadProgressText('');
      });
  };

  // 9. Fetch On-Demand Uploaded Slice
  const fetchUploadedSlice = (cid, sIdx) => {
    setLoadingUploadedSlice(true);
    fetch(`/api/uploaded-case/${cid}/slice/${sIdx}`)
      .then(res => res.json())
      .then(data => {
        setUploadedSliceDetail(data);
        setLoadingUploadedSlice(false);
      })
      .catch(err => {
        console.error("Uploaded slice fetch error:", err);
        setLoadingUploadedSlice(false);
      });
  };

  // 10. Scrubbing and Cine Loop for Uploaded Case
  useEffect(() => {
    if (!uploadedCaseResult) return;
    fetchUploadedSlice(uploadedCaseResult.case_id, uploadedCaseSliceIndex);
  }, [uploadedCaseSliceIndex]);

  useEffect(() => {
    if (!isUploadedPlaying || !uploadedCaseResult) return;
    const interval = setInterval(() => {
      setUploadedCaseSliceIndex(prev => (prev + 1) % uploadedCaseResult.slices_analyzed);
    }, 90);
    return () => clearInterval(interval);
  }, [isUploadedPlaying, uploadedCaseResult]);

  // 11. Navigation & Reporting for Uploaded Case
  const openUploadedInExplorer = () => {
    if (!uploadedCaseResult) return;
    setSelectedCaseId(uploadedCaseResult.case_id);
    setActiveTab('explorer');
  };

  const openUploadedReport = () => {
    if (!uploadedCaseResult) return;
    setReportMarkdown(uploadedCaseResult.report_markdown);
    setReportModalOpen(true);
  };

  // 7. Open Clinical Report
  const openReportModal = () => {
    fetch(`/api/report/${selectedCaseId}`)
      .then(res => res.json())
      .then(data => {
        setReportMarkdown(data.markdown_content);
        setReportModalOpen(true);
      })
      .catch(err => console.error("Report fetch error:", err));
  };

  const copyReportToClipboard = () => {
    navigator.clipboard.writeText(reportMarkdown);
    setCopiedReport(true);
    setTimeout(() => setCopiedReport(false), 2000);
  };

  return (
    <div className="app-container">
      {/* --- TOP HEADER BAR --- */}
      <header className="app-header">
        <div className="brand-section">
          <div className="brand-icon-wrapper">
            <Activity size={20} />
          </div>
          <div>
            <div className="brand-title">
              NephroAI
              <span className="badge badge-cyan">KiTS23 Edition</span>
            </div>
            <div className="brand-subtitle">
              Automated Renal Segmentation, Quantitative Volumetrics & TNM Staging
            </div>
          </div>
        </div>

        {/* Center Mode Switcher Tabs */}
        <div className="header-center-tabs">
          <button
            className={`nav-tab ${activeTab === 'explorer' ? 'active' : ''}`}
            onClick={() => setActiveTab('explorer')}
          >
            <Layers size={15} />
            Cohort Case Explorer
          </button>
          <button
            className={`nav-tab ${activeTab === 'upload' ? 'active' : ''}`}
            onClick={() => setActiveTab('upload')}
          >
            <Upload size={15} />
            Drag & Drop CT Upload
          </button>
        </div>

        {/* Hardware Status Pill */}
        <div className="header-status-section">
          <div className="status-pill">
            <div className="pulse-dot" />
            <span>{health ? `${health.device.toUpperCase()} Acceleration` : 'Connecting...'}</span>
          </div>
          <button className="btn-secondary" onClick={openReportModal}>
            <FileText size={15} />
            Clinical Report
          </button>
        </div>
      </header>

      {/* --- MAIN WORKSPACE --- */}
      {activeTab === 'explorer' ? (
        <main className="main-layout">
          {/* LEFT: CT VIEWER PANEL */}
          <section className="glass-panel viewer-panel">
            {/* Header Controls */}
            <div className="viewer-header">
              <div className="case-select-wrapper">
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>
                  PATIENT CASE:
                </span>
                <select
                  className="select-input"
                  value={selectedCaseId}
                  onChange={e => setSelectedCaseId(e.target.value)}
                >
                  {cases.map(c => (
                    <option key={c.case_id} value={c.case_id}>
                      {c.case_id} — ({c.slice_count} slices {c.has_tumor ? '• Neoplasm' : ''})
                    </option>
                  ))}
                </select>
              </div>

              {/* View Mode Buttons */}
              <div className="view-mode-selector">
                <button
                  className={`mode-btn ${viewMode === 'ai_overlay' ? 'active' : ''}`}
                  onClick={() => setViewMode('ai_overlay')}
                >
                  AI Overlay
                </button>
                <button
                  className={`mode-btn ${viewMode === 'gt_overlay' ? 'active' : ''}`}
                  onClick={() => setViewMode('gt_overlay')}
                >
                  Ground Truth
                </button>
                <button
                  className={`mode-btn ${viewMode === 'split' ? 'active' : ''}`}
                  onClick={() => setViewMode('split')}
                >
                  Side-by-Side
                </button>
                <button
                  className={`mode-btn ${viewMode === 'pure_ct' ? 'active' : ''}`}
                  onClick={() => setViewMode('pure_ct')}
                >
                  Grayscale CT
                </button>
              </div>
            </div>

            {/* Viewport Frame */}
            <div className="viewport-frame">
              {viewMode === 'split' ? (
                <div className="split-view-container">
                  <div className="split-pane">
                    <span className="split-pane-label">AI RESIDUAL U-NET</span>
                    {sliceData && (
                      <img
                        src={sliceData.overlay_image}
                        alt="AI Prediction"
                        className="viewport-image"
                      />
                    )}
                  </div>
                  <div className="split-pane">
                    <span className="split-pane-label">EXPERT GROUND TRUTH</span>
                    {sliceData && (
                      <img
                        src={sliceData.ground_truth_overlay || sliceData.original_image}
                        alt="Ground Truth"
                        className="viewport-image"
                      />
                    )}
                  </div>
                </div>
              ) : (
                <>
                  {sliceData && (
                    <img
                      src={
                        viewMode === 'pure_ct'
                          ? sliceData.original_image
                          : viewMode === 'gt_overlay'
                          ? (sliceData.ground_truth_overlay || sliceData.overlay_image)
                          : sliceData.overlay_image
                      }
                      alt="CT Slice"
                      className="viewport-image"
                      style={{
                        opacity: viewMode === 'pure_ct' ? 1.0 : opacity
                      }}
                    />
                  )}
                  {loadingSlice && (
                    <div style={{ position: 'absolute', bottom: 12, right: 12, background: 'rgba(0,0,0,0.7)', padding: '4px 8px', borderRadius: 4, fontSize: '0.7rem' }}>
                      Inference active...
                    </div>
                  )}
                </>
              )}
            </div>

            {/* Slice Slider & Cine Bar */}
            <div className="slice-control-bar">
              <div className="slice-slider-row">
                <button
                  className="playback-btn"
                  onClick={() => setIsPlaying(p => !p)}
                  title={isPlaying ? "Pause Cine Loop" : "Play Cine Loop"}
                >
                  {isPlaying ? <Pause size={15} /> : <Play size={15} />}
                </button>
                <button
                  className="playback-btn"
                  onClick={() => setSliceIndex(prev => Math.max(0, prev - 1))}
                  title="Previous Slice (←)"
                >
                  <ChevronLeft size={16} />
                </button>
                <input
                  type="range"
                  min="0"
                  max={Math.max(0, totalSlices - 1)}
                  value={sliceIndex}
                  onChange={e => setSliceIndex(parseInt(e.target.value))}
                />
                <button
                  className="playback-btn"
                  onClick={() => setSliceIndex(prev => Math.min(totalSlices - 1, prev + 1))}
                  title="Next Slice (→)"
                >
                  <ChevronRight size={16} />
                </button>
                <div className="slice-counter">
                  {sliceIndex + 1} / {totalSlices}
                </div>
              </div>

              {/* Legend & Layer Toggles */}
              <div className="legend-row">
                <div className="legend-chips">
                  <div
                    className={`legend-chip ${visibleLayers.kidney ? '' : 'inactive'}`}
                    style={{ background: 'rgba(16, 185, 129, 0.12)', color: '#34D399', border: '1px solid rgba(16, 185, 129, 0.3)' }}
                    onClick={() => setVisibleLayers(v => ({ ...v, kidney: !v.kidney }))}
                  >
                    <div className="chip-color-box" style={{ background: 'var(--color-kidney)' }} />
                    Kidney Parenchyma
                  </div>
                  <div
                    className={`legend-chip ${visibleLayers.tumor ? '' : 'inactive'}`}
                    style={{ background: 'rgba(239, 68, 68, 0.12)', color: '#F87171', border: '1px solid rgba(239, 68, 68, 0.3)' }}
                    onClick={() => setVisibleLayers(v => ({ ...v, tumor: !v.tumor }))}
                  >
                    <div className="chip-color-box" style={{ background: 'var(--color-tumor)' }} />
                    Renal Neoplasm (Tumor)
                  </div>
                  <div
                    className={`legend-chip ${visibleLayers.cyst ? '' : 'inactive'}`}
                    style={{ background: 'rgba(59, 130, 246, 0.12)', color: '#60A5FA', border: '1px solid rgba(59, 130, 246, 0.3)' }}
                    onClick={() => setVisibleLayers(v => ({ ...v, cyst: !v.cyst }))}
                  >
                    <div className="chip-color-box" style={{ background: 'var(--color-cyst)' }} />
                    Renal Cyst
                  </div>
                </div>

                <div className="opacity-slider-wrapper">
                  <Sliders size={13} />
                  <span>Opacity</span>
                  <input
                    type="range"
                    min="0.2"
                    max="1.0"
                    step="0.05"
                    value={opacity}
                    onChange={e => setOpacity(parseFloat(e.target.value))}
                  />
                </div>
              </div>
            </div>
          </section>

          {/* RIGHT: QUANTITATIVE ANALYTICS & CLINICAL STAGING */}
          <section className="analytics-panel">
            {/* AJCC TNM Staging Card */}
            <div className="glass-panel staging-card">
              <div className="card-title-row">
                <div className="card-title">
                  <ShieldCheck size={18} color="var(--accent-primary)" />
                  AJCC 8th Edition Clinical Staging
                </div>
                {metrics3D && (
                  <span className="badge badge-cyan">
                    {metrics3D.inference_time_sec}s 3D Inference
                  </span>
                )}
              </div>

              {metrics3D ? (
                <>
                  <div className="tnm-banner">
                    <div className={`stage-badge-large ${metrics3D.tnm_stage === 'cT1a' ? 'stage-badge-t1' : ''}`}>
                      {metrics3D.tnm_stage}
                    </div>
                    <div className="stage-info">
                      <div className="stage-headline">
                        Clinical Stage: {metrics3D.tnm_stage}
                      </div>
                      <div className="stage-detail">
                        Maximum 3D Euclidean Diameter: <strong>{metrics3D.gt_comparison ? metrics3D.gt_comparison.gt_d_max_cm : metrics3D.width_cm} cm</strong>
                      </div>
                    </div>
                  </div>

                  <div className="surgical-box">
                    <strong>Surgical Guidance:</strong> {metrics3D.surgical_recommendation}
                  </div>
                </>
              ) : (
                <div style={{ textAlign: 'center', padding: '24px', color: 'var(--text-muted)' }}>
                  {isComputing3D ? "Reconstructing 3D Volume..." : "Click below to compute 3D Clinical Staging."}
                </div>
              )}
            </div>

            {/* 3D Volumetrics Card */}
            <div className="glass-panel volumetrics-card">
              <div className="card-title-row">
                <div className="card-title">
                  <BarChart2 size={18} color="var(--color-kidney)" />
                  Physical Volumetrics (cm³ / mL)
                </div>
                <button
                  className="btn-primary"
                  style={{ fontSize: '0.775rem', padding: '5px 10px' }}
                  onClick={() => trigger3DReconstruction(selectedCaseId)}
                  disabled={isComputing3D}
                >
                  <RotateCw size={13} className={isComputing3D ? 'animate-spin' : ''} />
                  {isComputing3D ? "Reconstructing..." : (metrics3D ? "Recompute 3D" : "Compute 3D")}
                </button>
              </div>

              {metrics3D && (
                <>
                  <div className="metric-grid-4">
                    <div className="metric-tile tile-kidney">
                      <span className="tile-label">Right Kidney</span>
                      <span className="tile-value">{metrics3D.right_kidney_vol_cm3}</span>
                      <span className="tile-sub">cm³ functional parenchyma</span>
                    </div>

                    <div className="metric-tile tile-kidney">
                      <span className="tile-label">Left Kidney</span>
                      <span className="tile-value">{metrics3D.left_kidney_vol_cm3}</span>
                      <span className="tile-sub">cm³ functional parenchyma</span>
                    </div>

                    <div className="metric-tile tile-tumor">
                      <span className="tile-label">Renal Neoplasm</span>
                      <span className="tile-value" style={{ color: '#F87171' }}>{metrics3D.tumor_vol_cm3}</span>
                      <span className="tile-sub">
                        {metrics3D.gt_comparison ? `GT: ${metrics3D.gt_comparison.gt_tumor_vol_cm3} cm³` : 'Solid Tumor Volume'}
                      </span>
                    </div>

                    <div className="metric-tile tile-cyst">
                      <span className="tile-label">Renal Cyst</span>
                      <span className="tile-value" style={{ color: '#60A5FA' }}>{metrics3D.cyst_vol_cm3}</span>
                      <span className="tile-sub">Benign cystic volume</span>
                    </div>
                  </div>

                  {/* Tumor Burden Bar */}
                  <div style={{ marginTop: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.75rem', fontWeight: 600 }}>
                      <span style={{ color: 'var(--text-muted)' }}>ORGAN TUMOR BURDEN:</span>
                      <span style={{ color: '#F87171', fontFamily: 'var(--font-mono)' }}>
                        {metrics3D.tumor_burden_pct}%
                      </span>
                    </div>
                    <div className="progress-track">
                      <div
                        className="progress-fill"
                        style={{ width: `${Math.min(100, metrics3D.tumor_burden_pct)}%` }}
                      />
                    </div>
                  </div>

                  {/* Ground Truth Agreement Pill */}
                  {metrics3D.gt_comparison && (
                    <div style={{ marginTop: '14px', background: 'var(--bg-surface-elevated)', padding: '10px', borderRadius: 8, fontSize: '0.75rem', display: 'flex', justifyContent: 'space-between' }}>
                      <span><strong>Kidney Dice:</strong> {metrics3D.gt_comparison.dice_kidney}</span>
                      <span><strong>Tumor Dice:</strong> {metrics3D.gt_comparison.dice_tumor}</span>
                      <span><strong>Cyst Dice:</strong> {metrics3D.gt_comparison.dice_cyst}</span>
                    </div>
                  )}
                </>
              )}
            </div>

            {/* 3D PCA Morphometry Card */}
            <div className="glass-panel pca-card">
              <div className="card-title-row">
                <div className="card-title">
                  <Cpu size={18} color="var(--accent-amber)" />
                  3D PCA Renal Calipers & Traditional Ellipsoid Error
                </div>
              </div>

              {metrics3D && (
                <>
                  <div className="pca-grid">
                    <div className="pca-box">
                      <div className="pca-axis-name">Length (L)</div>
                      <div className="pca-axis-val">{metrics3D.length_cm} cm</div>
                    </div>
                    <div className="pca-box">
                      <div className="pca-axis-name">Width (W)</div>
                      <div className="pca-axis-val">{metrics3D.width_cm} cm</div>
                    </div>
                    <div className="pca-box">
                      <div className="pca-axis-name">Thickness (T)</div>
                      <div className="pca-axis-val">{metrics3D.thickness_cm} cm</div>
                    </div>
                  </div>

                  <div className="citation-note">
                    <strong>Koukoutegos et al. (2024) Audit:</strong> Manual prolate ellipsoid approximation ({metrics3D.ellipsoid_vol_cm3} cm³) overestimates true 3D volume by <strong>+{metrics3D.ellipsoid_error_pct}%</strong> due to the non-elliptical renal hilum.
                  </div>
                </>
              )}
            </div>
          </section>
        </main>
      ) : (
        /* --- DUAL MODE UPLOAD VIEW: WHOLE CASE FOLDER & SINGLE CT SLICE --- */
        <main className="upload-container">
          {/* Mode Switcher Tabs */}
          <div className="upload-mode-selector">
            <button
              className={`upload-mode-pill ${uploadMode === 'case' ? 'active' : ''}`}
              onClick={() => {
                setUploadMode('case');
                setSingleSliceResult(null);
              }}
            >
              <FolderPlus size={16} />
              <span>Whole Case Folder (3D Volumetric Pipeline)</span>
            </button>
            <button
              className={`upload-mode-pill ${uploadMode === 'slice' ? 'active' : ''}`}
              onClick={() => {
                setUploadMode('slice');
                setUploadedCaseResult(null);
                setUploadedSliceDetail(null);
              }}
            >
              <Upload size={16} />
              <span>Single CT Slice (2D Instant Segmentation)</span>
            </button>
          </div>

          {/* Hidden File / Folder Inputs */}
          <input
            type="file"
            ref={folderInputRef}
            style={{ display: 'none' }}
            webkitdirectory="true"
            directory="true"
            multiple
            onChange={handleFolderSelect}
          />
          <input
            type="file"
            ref={multiFileInputRef}
            style={{ display: 'none' }}
            multiple
            accept="image/*,.png,.jpg,.jpeg,.tif,.tiff"
            onChange={handleMultiFilesSelect}
          />
          <input
            type="file"
            ref={fileInputRef}
            style={{ display: 'none' }}
            accept="image/*,.png,.jpg,.jpeg,.tif,.tiff"
            onChange={handleSingleFileSelect}
          />

          {/* Drag & Drop Zone */}
          {(!uploadedCaseResult && !singleSliceResult) && (
            <div
              className={`dropzone-box ${dragActive ? 'drag-active' : ''}`}
              onDragOver={e => { e.preventDefault(); setDragActive(true); }}
              onDragLeave={() => setDragActive(false)}
              onDrop={handleDrop}
              onClick={() => {
                if (uploadMode === 'case') {
                  folderInputRef.current.click();
                } else {
                  fileInputRef.current.click();
                }
              }}
            >
              {uploadMode === 'case' ? (
                <>
                  <FolderPlus className="dropzone-icon" />
                  <div className="dropzone-title">Drop Patient Case Folder Here</div>
                  <div className="dropzone-desc">
                    Drag and drop an entire case directory (e.g., <code>case_00000/</code>) or multiple axial CT slices.
                    Automatically executes 3D volume reconstruction, 3D CCA cleaning, PCA morphometry, and AJCC TNM staging.
                  </div>
                  <div className="dropzone-actions" onClick={e => e.stopPropagation()}>
                    <button className="btn-primary" onClick={() => folderInputRef.current.click()}>
                      <FolderPlus size={16} /> Select Case Folder
                    </button>
                    <button className="btn-secondary" onClick={() => multiFileInputRef.current.click()}>
                      <Layers size={16} /> Select Multiple Slices
                    </button>
                  </div>
                </>
              ) : (
                <>
                  <Upload className="dropzone-icon" />
                  <div className="dropzone-title">Drop Single CT Slice Image Here</div>
                  <div className="dropzone-desc">
                    Supports PNG, JPEG, TIFF axial cross-sectional kidney CT scans. (Standardized to 512×512 with 16-bit / 8-bit auto-scaling).
                  </div>
                  <div className="dropzone-actions" onClick={e => e.stopPropagation()}>
                    <button className="btn-primary" onClick={() => fileInputRef.current.click()}>
                      <Upload size={16} /> Select CT Image File
                    </button>
                  </div>
                </>
              )}
            </div>
          )}

          {/* Live Progress Banner during Inference */}
          {isUploading && (
            <div className="upload-loading-banner glass-panel">
              <RotateCw className="spin" size={24} color="var(--accent-primary)" />
              <div>
                <div style={{ fontWeight: 700, color: 'var(--text-white)' }}>
                  Processing Deep Learning Pipeline
                </div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
                  {uploadProgressText}
                </div>
              </div>
            </div>
          )}

          {/* ============================================================================== */}
          {/* WHOLE CASE UPLOAD RESULTS DASHBOARD */}
          {/* ============================================================================== */}
          {uploadedCaseResult && (
            <div className="glass-panel uploaded-case-card">
              {/* Header with Case Metadata, Staging & Surgical Guidance */}
              <div className="uploaded-case-header">
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <h2 style={{ fontSize: '1.4rem', fontWeight: 800, color: 'var(--text-white)', margin: 0 }}>
                      Patient Case: {uploadedCaseResult.case_id}
                    </h2>
                    <span className="badge badge-emerald">Custom Uploaded Case</span>
                    <span className="badge badge-cyan">{uploadedCaseResult.slices_analyzed} Slices Analyzed</span>
                  </div>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: 4, display: 'block' }}>
                    Inference & 3D Volumetrics Latency: <strong>{uploadedCaseResult.inference_time_sec}s</strong> (FP16 AMP Tensor Core Accelerated)
                  </span>
                </div>

                <div style={{ display: 'flex', gap: 10 }}>
                  <button className="btn-primary" onClick={openUploadedReport}>
                    <FileText size={15} /> View Formal Report
                  </button>
                  <button className="btn-secondary" onClick={openUploadedInExplorer}>
                    <Eye size={15} /> Open in 3D Explorer
                  </button>
                  <button className="btn-secondary" onClick={() => { setUploadedCaseResult(null); setUploadedSliceDetail(null); }}>
                    <RotateCw size={15} /> Upload Another
                  </button>
                </div>
              </div>

              {/* TNM Stage & Surgical Guidance Banner */}
              <div className="tnm-banner">
                <div className={`stage-badge-large ${uploadedCaseResult.tnm_stage.startsWith('cT1') ? 'stage-badge-t1' : ''}`}>
                  {uploadedCaseResult.tnm_stage}
                </div>
                <div className="stage-info">
                  <div className="stage-headline">
                    AJCC / UICC 8th Edition Clinical Staging
                  </div>
                  <div className="stage-detail">
                    {uploadedCaseResult.surgical_recommendation}
                  </div>
                </div>
              </div>

              {/* Interactive Axial Slice Navigator for Uploaded Case */}
              <div className="uploaded-navigator-box">
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                    <Layers size={18} color="var(--accent-primary)" />
                    <span style={{ fontWeight: 700, color: 'var(--text-white)' }}>
                      Axial Slice Inspection: Slice {uploadedCaseSliceIndex + 1} of {uploadedCaseResult.slices_analyzed}
                    </span>
                    {uploadedCaseSliceIndex === uploadedCaseResult.best_preview_slice && (
                      <span className="badge badge-amber" style={{ fontSize: '0.7rem' }}>
                        ★ Maximum Pathology Finding
                      </span>
                    )}
                  </div>

                  {uploadedSliceDetail && (
                    <div style={{ fontSize: '0.75rem', display: 'flex', gap: 12 }}>
                      <span style={{ color: '#34D399' }}>Kidney: {uploadedSliceDetail.kidney_area_cm2} cm²</span>
                      <span style={{ color: '#F87171' }}>Tumor: {uploadedSliceDetail.tumor_area_cm2} cm²</span>
                      <span style={{ color: '#60A5FA' }}>Cyst: {uploadedSliceDetail.cyst_area_cm2} cm²</span>
                    </div>
                  )}
                </div>

                {/* 3-Column Image Display: CT | Predicted Mask | Color Overlay */}
                {uploadedSliceDetail ? (
                  <div className="uploaded-slices-grid">
                    <div className="uploaded-slice-card">
                      <img src={uploadedSliceDetail.original_image} alt="Original CT" />
                      <span className="uploaded-slice-label">1. Grayscale Patient CT Scan</span>
                    </div>
                    <div className="uploaded-slice-card">
                      <img src={uploadedSliceDetail.predicted_mask} alt="AI Mask" />
                      <span className="uploaded-slice-label">2. Residual U-Net Segmented Mask</span>
                    </div>
                    <div className="uploaded-slice-card">
                      <img src={uploadedSliceDetail.overlay_image} alt="Diagnostic Overlay" />
                      <span className="uploaded-slice-label">3. Diagnostic Color Overlay</span>
                    </div>
                  </div>
                ) : (
                  <div style={{ padding: 40, textAlign: 'center', color: 'var(--text-muted)' }}>
                    {loadingUploadedSlice ? "Loading slice visualization..." : "Select slice to inspect"}
                  </div>
                )}

                {/* Scrubber and Cine-Loop Controls */}
                <div className="uploaded-scrubber-row">
                  <button
                    className="btn-icon"
                    onClick={() => setUploadedCaseSliceIndex(prev => Math.max(0, prev - 1))}
                    disabled={uploadedCaseSliceIndex === 0}
                  >
                    <ChevronLeft size={18} />
                  </button>

                  <button
                    className="btn-icon"
                    onClick={() => setIsUploadedPlaying(p => !p)}
                    style={{ background: isUploadedPlaying ? 'rgba(6, 182, 212, 0.2)' : undefined }}
                  >
                    {isUploadedPlaying ? <Pause size={18} color="var(--accent-primary)" /> : <Play size={18} />}
                  </button>

                  <button
                    className="btn-icon"
                    onClick={() => setUploadedCaseSliceIndex(prev => Math.min(uploadedCaseResult.slices_analyzed - 1, prev + 1))}
                    disabled={uploadedCaseSliceIndex === uploadedCaseResult.slices_analyzed - 1}
                  >
                    <ChevronRight size={18} />
                  </button>

                  <input
                    type="range"
                    min="0"
                    max={uploadedCaseResult.slices_analyzed - 1}
                    value={uploadedCaseSliceIndex}
                    onChange={e => setUploadedCaseSliceIndex(parseInt(e.target.value))}
                    style={{ flex: 1 }}
                  />

                  <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', minWidth: 60, textAlign: 'right' }}>
                    {uploadedCaseSliceIndex + 1} / {uploadedCaseResult.slices_analyzed}
                  </span>
                </div>

                {/* Quick Navigation Jump Chips */}
                <div className="uploaded-quick-jumps">
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 600 }}>Quick Jump:</span>
                  <button
                    className={`jump-chip ${uploadedCaseSliceIndex === uploadedCaseResult.best_preview_slice ? 'active' : ''}`}
                    onClick={() => setUploadedCaseSliceIndex(uploadedCaseResult.best_preview_slice)}
                  >
                    🔴 Max Pathology Slice (#{uploadedCaseResult.best_preview_slice + 1})
                  </button>
                  <button
                    className="jump-chip"
                    onClick={() => setUploadedCaseSliceIndex(Math.floor(uploadedCaseResult.slices_analyzed / 2))}
                  >
                    Center Slice (#{Math.floor(uploadedCaseResult.slices_analyzed / 2) + 1})
                  </button>
                  <button
                    className="jump-chip"
                    onClick={() => setUploadedCaseSliceIndex(0)}
                  >
                    Superior Margin (Slice #1)
                  </button>
                  <button
                    className="jump-chip"
                    onClick={() => setUploadedCaseSliceIndex(uploadedCaseResult.slices_analyzed - 1)}
                  >
                    Inferior Margin (Slice #{uploadedCaseResult.slices_analyzed})
                  </button>
                </div>
              </div>

              {/* 3D Physical Volumetrics Grid */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
                <div className="metric-tile tile-cyan">
                  <span className="tile-label">Total Parenchyma</span>
                  <span className="tile-value">{uploadedCaseResult.total_parenchyma_vol_cm3}</span>
                  <span className="tile-sub">cm³ functional reserve</span>
                </div>
                <div className="metric-tile tile-tumor">
                  <span className="tile-label">Renal Neoplasm</span>
                  <span className="tile-value" style={{ color: '#F87171' }}>{uploadedCaseResult.tumor_vol_cm3}</span>
                  <span className="tile-sub">cm³ solid tumor</span>
                </div>
                <div className="metric-tile tile-cyst">
                  <span className="tile-label">Renal Cyst</span>
                  <span className="tile-value" style={{ color: '#60A5FA' }}>{uploadedCaseResult.cyst_vol_cm3}</span>
                  <span className="tile-sub">cm³ benign volume</span>
                </div>
                <div className="metric-tile tile-kidney">
                  <span className="tile-label">Tumor Burden</span>
                  <span className="tile-value">{uploadedCaseResult.tumor_burden_pct}%</span>
                  <span className="tile-sub">of functional parenchyma</span>
                </div>
              </div>

              {/* 3D PCA Morphometry & Ellipsoid Overestimate Error */}
              <div className="glass-panel pca-card" style={{ padding: '16px' }}>
                <div className="card-title-row">
                  <div className="card-title">
                    <Cpu size={16} color="var(--accent-amber)" />
                    3D PCA Calipers & Traditional Ellipsoid Audit (Koukoutegos et al., 2024)
                  </div>
                </div>

                <div className="pca-grid">
                  <div className="pca-box">
                    <div className="pca-axis-name">Length (L)</div>
                    <div className="pca-axis-val">{uploadedCaseResult.length_cm} cm</div>
                  </div>
                  <div className="pca-box">
                    <div className="pca-axis-name">Width (W)</div>
                    <div className="pca-axis-val">{uploadedCaseResult.width_cm} cm</div>
                  </div>
                  <div className="pca-box">
                    <div className="pca-axis-name">Thickness (T)</div>
                    <div className="pca-axis-val">{uploadedCaseResult.thickness_cm} cm</div>
                  </div>
                </div>

                <div className="citation-note" style={{ marginTop: 10 }}>
                  <strong>Prolate Ellipsoid Volume:</strong> {uploadedCaseResult.ellipsoid_vol_cm3} cm³ | 
                  Traditional 3-axis caliper approximation overestimates true volume by <strong>+{uploadedCaseResult.ellipsoid_error_pct}%</strong> due to complex renal hilum geometry.
                </div>
              </div>
            </div>
          )}

          {/* ============================================================================== */}
          {/* SINGLE CT SLICE UPLOAD RESULT */}
          {/* ============================================================================== */}
          {singleSliceResult && (
            <div className="glass-panel" style={{ maxWidth: 960, width: '100%', padding: '24px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <h3 style={{ color: 'var(--text-white)', margin: 0 }}>Custom Slice 2D Segmentation Result</h3>
                  <span className="badge badge-cyan">{singleSliceResult.inference_time_ms} ms</span>
                </div>
                <button
                  className="btn-secondary"
                  onClick={() => setSingleSliceResult(null)}
                >
                  <RotateCw size={14} /> Analyze Another Slice
                </button>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '14px', marginBottom: '20px' }}>
                <div style={{ textAlign: 'center', background: 'var(--bg-deep)', padding: 10, borderRadius: 8 }}>
                  <img src={singleSliceResult.original_image} alt="Original" style={{ width: '100%', borderRadius: 6, background: '#000' }} />
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 6, display: 'block' }}>Grayscale CT</span>
                </div>
                <div style={{ textAlign: 'center', background: 'var(--bg-deep)', padding: 10, borderRadius: 8 }}>
                  <img src={singleSliceResult.predicted_mask} alt="Mask" style={{ width: '100%', borderRadius: 6, background: '#000' }} />
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 6, display: 'block' }}>AI Predicted Mask</span>
                </div>
                <div style={{ textAlign: 'center', background: 'var(--bg-deep)', padding: 10, borderRadius: 8 }}>
                  <img src={singleSliceResult.overlay_image} alt="Overlay" style={{ width: '100%', borderRadius: 6, background: '#000' }} />
                  <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 6, display: 'block' }}>Color-Coded Overlay</span>
                </div>
              </div>

              <div className="metric-grid-4">
                <div className="metric-tile tile-kidney">
                  <span className="tile-label">Kidney Area</span>
                  <span className="tile-value">{singleSliceResult.kidney_area_cm2} cm²</span>
                </div>
                <div className="metric-tile tile-tumor">
                  <span className="tile-label">Tumor Area</span>
                  <span className="tile-value" style={{ color: '#F87171' }}>{singleSliceResult.tumor_area_cm2} cm²</span>
                </div>
                <div className="metric-tile tile-cyst">
                  <span className="tile-label">Cyst Area</span>
                  <span className="tile-value" style={{ color: '#60A5FA' }}>{singleSliceResult.cyst_area_cm2} cm²</span>
                </div>
                <div className="metric-tile tile-cyan">
                  <span className="tile-label">Kidney Voxels</span>
                  <span className="tile-value">{singleSliceResult.pixel_counts.kidney} px</span>
                </div>
              </div>
            </div>
          )}
        </main>
      )}

      {/* --- CLINICAL RADIOLOGY REPORT MODAL --- */}
      {reportModalOpen && (
        <div className="modal-backdrop" onClick={() => setReportModalOpen(false)}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <div className="modal-header">
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, fontWeight: 700, color: 'var(--text-white)' }}>
                <FileText size={18} color="var(--accent-primary)" />
                Formal Clinical Radiology & Decision Support Report
              </div>
              <button
                style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: '1.2rem' }}
                onClick={() => setReportModalOpen(false)}
              >
                ✕
              </button>
            </div>

            <div className="modal-body">
              <pre>{reportMarkdown}</pre>
            </div>

            <div className="modal-footer">
              <button className="btn-secondary" onClick={copyReportToClipboard}>
                {copiedReport ? <Check size={14} color="#10B981" /> : <Copy size={14} />}
                {copiedReport ? "Copied!" : "Copy Markdown"}
              </button>
              <button className="btn-primary" onClick={() => window.print()}>
                <Download size={14} />
                Print / Save PDF
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
