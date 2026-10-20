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
  const [caseSlices, setCaseSlices] = useState([]);
  const [loadingCase, setLoadingCase] = useState(false);
  const caseCacheRef = useRef({});
  const uploadedSliceCacheRef = useRef({});
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
  const [uploadPreviewUrl, setUploadPreviewUrl] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [uploadError, setUploadError] = useState('');

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

  // 2. Fetch Entire Case Bundle at once (Batch-computed slices + 3D metrics)
  useEffect(() => {
    if (!selectedCaseId) return;

    // Check client-side memory cache first for immediate zero-latency load
    if (caseCacheRef.current[selectedCaseId]) {
      const cached = caseCacheRef.current[selectedCaseId];
      const slices = cached.slices || [];
      setCaseSlices(slices);
      const total = cached.total_slices || slices.length;
      setTotalSlices(total);

      // Default to slice with tumor or middle slice
      const tumorIdx = slices.findIndex(s => (s.tumor_area_cm2 > 0) || (s.pixel_counts && s.pixel_counts.tumor > 0));
      const initialIdx = (tumorIdx !== -1) ? tumorIdx : Math.floor(total / 2);
      setSliceIndex(initialIdx);
      if (slices[initialIdx]) {
        setSliceData(slices[initialIdx]);
      }
      if (cached.metrics_3d) {
        setMetrics3D(cached.metrics_3d);
      }
      setLoadingCase(false);
      setLoadingSlice(false);
      return;
    }

    setLoadingCase(true);
    setLoadingSlice(true);
    setSliceData(null);
    setCaseSlices([]);

    fetch(`/api/case/${selectedCaseId}/full`)
      .then(res => {
        if (!res.ok) throw new Error(`Status ${res.status}`);
        return res.json();
      })
      .then(data => {
        if (data && data.slices && data.slices.length > 0) {
          caseCacheRef.current[selectedCaseId] = data;
          setCaseSlices(data.slices);
          const total = data.total_slices || data.slices.length;
          setTotalSlices(total);

          const tumorIdx = data.slices.findIndex(s => (s.tumor_area_cm2 > 0) || (s.pixel_counts && s.pixel_counts.tumor > 0));
          const initialIdx = (tumorIdx !== -1) ? tumorIdx : Math.floor(total / 2);
          setSliceIndex(initialIdx);
          setSliceData(data.slices[initialIdx]);

          if (data.metrics_3d) {
            setMetrics3D(data.metrics_3d);
          }
        }
        setLoadingCase(false);
        setLoadingSlice(false);
      })
      .catch(err => {
        console.warn("Full case fetch fallback:", err);
        fetch(`/api/case/${selectedCaseId}/slices`)
          .then(res => res.json())
          .then(data => {
            if (data && data.slices) {
              setTotalSlices(data.total_slices);
              setSliceIndex(Math.floor(data.total_slices / 2));
            }
            setLoadingCase(false);
          })
          .catch(() => setLoadingCase(false));

        fetch(`/api/case/${selectedCaseId}/metrics`)
          .then(res => (res.ok ? res.json() : null))
          .then(cached => {
            if (cached && cached.tnm_stage) setMetrics3D(cached);
            else setMetrics3D(null);
          })
          .catch(() => setMetrics3D(null));
      });
  }, [selectedCaseId]);

  // 3. Instant Slice Navigation (Displays suddenly from in-memory case data with ZERO server roundtrip)
  useEffect(() => {
    if (caseSlices && caseSlices.length > 0) {
      const slice = caseSlices[sliceIndex] || caseSlices.find(s => s.slice_index === sliceIndex);
      if (slice) {
        setSliceData(slice);
        setLoadingSlice(false);
        return; // Zero network latency!
      }
    }

    // Fallback: only if whole-case bundle is not yet loaded in memory
    if (selectedCaseId && !loadingCase) {
      setLoadingSlice(true);
      fetch(`/api/case/${selectedCaseId}/slice/${sliceIndex}`)
        .then(res => res.json())
        .then(data => {
          setSliceData(data);
          setLoadingSlice(false);
        })
        .catch(err => {
          console.error("Error fetching slice fallback:", err);
          setLoadingSlice(false);
        });
    }
  }, [selectedCaseId, sliceIndex, caseSlices, loadingCase]);

  // 4. Cine Loop Playback (Fluid 60fps playback from RAM)
  useEffect(() => {
    let interval = null;
    if (isPlaying) {
      interval = setInterval(() => {
        setSliceIndex(prev => (prev + 1 >= totalSlices ? 0 : prev + 1));
      }, 100);
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

  const analysisReady = () => {
    if (health && !health.model_loaded) {
      setUploadError('Analysis is unavailable because no trained model checkpoint is configured. Add the checkpoint, then restart the API before uploading a study.');
      return false;
    }
    return true;
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
    if (!analysisReady()) return;
    const validExts = ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.dcm'];
    const imageFiles = filesList.filter(f => {
      const name = f.name.toLowerCase();
      return validExts.some(ext => name.endsWith(ext));
    });

    if (imageFiles.length === 0) {
      setUploadError('No compatible CT slices were selected. Use PNG, JPG, TIFF, BMP, or DICOM files.');
      return;
    }

    setUploadError('');

    let detectedName = '';
    if (imageFiles[0].webkitRelativePath) {
      const parts = imageFiles[0].webkitRelativePath.split('/');
      if (parts.length > 1) detectedName = parts[0];
    }
    if (!detectedName) {
      const match = imageFiles[0].name.match(/case_\d+/i);
      if (match) detectedName = match[0].toLowerCase();
    }

    try {
      const url = URL.createObjectURL(imageFiles[0]);
      setUploadPreviewUrl(url);
    } catch {
      setUploadPreviewUrl(null);
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
        setUploadPreviewUrl(null);

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
        setUploadError(err.message.includes('Checkpoint not found')
          ? 'Analysis could not start because the trained model checkpoint is missing. Configure the model, then try again.'
          : 'The study could not be analyzed. Check that the files are supported and the analysis service is running.');
        setIsUploading(false);
        setUploadProgressText('');
        setUploadPreviewUrl(null);
      });
  };

  // 8. Single CT Slice Upload Handler (2D Instant)
  const handleSingleSliceUpload = (file) => {
    if (!file) return;
    if (!analysisReady()) return;
    setUploadError('');
    try {
      const url = URL.createObjectURL(file);
      setUploadPreviewUrl(url);
    } catch {
      setUploadPreviewUrl(null);
    }
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
        setUploadPreviewUrl(null);
      })
      .catch(err => {
        console.error("Single slice upload error:", err);
        setUploadError(err.message.includes('Checkpoint not found')
          ? 'Analysis could not start because the trained model checkpoint is missing. Configure the model, then try again.'
          : 'The slice could not be analyzed. Check the file and service status, then try again.');
        setIsUploading(false);
        setUploadProgressText('');
        setUploadPreviewUrl(null);
      });
  };

  // 9. Fetch On-Demand Uploaded Slice (Cached in memory for instant display)
  const fetchUploadedSlice = (cid, sIdx) => {
    const cacheKey = `${cid}_${sIdx}`;
    if (uploadedSliceCacheRef.current[cacheKey]) {
      setUploadedSliceDetail(uploadedSliceCacheRef.current[cacheKey]);
      setLoadingUploadedSlice(false);
      return;
    }
    setLoadingUploadedSlice(true);
    fetch(`/api/uploaded-case/${cid}/slice/${sIdx}`)
      .then(res => res.json())
      .then(data => {
        uploadedSliceCacheRef.current[cacheKey] = data;
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
        cases.length === 0 ? (
          <main className="empty-workspace">
            <section className="empty-hero glass-panel">
              <div className="empty-icon"><Layers size={26} /></div>
              <div className="eyebrow">CASE REVIEW WORKSPACE</div>
              <h1>No imaging studies available</h1>
              <p>
                Connect a dataset or upload a CT study to start renal segmentation,
                volumetric analysis, and clinical staging.
              </p>
              <div className="empty-actions">
                <button className="btn-primary" onClick={() => setActiveTab('upload')}>
                  <Upload size={16} /> Upload a study
                </button>
                <button className="btn-secondary" onClick={() => window.open('/docs', '_blank')}>
                  <FileText size={16} /> View API documentation
                </button>
              </div>
            </section>

            <section className="empty-guidance glass-panel">
              <div>
                <div className="card-kicker">WHAT YOU CAN REVIEW</div>
                <h2>From slice-level anatomy to clinical context</h2>
              </div>
              <div className="guidance-items">
                <div><span>01</span><p><strong>Segment anatomy</strong><br />Review kidney, tumor, and cyst overlays.</p></div>
                <div><span>02</span><p><strong>Measure volume</strong><br />Inspect organ volumes and tumor burden.</p></div>
                <div><span>03</span><p><strong>Generate a report</strong><br />Translate findings into a clinical summary.</p></div>
              </div>
            </section>
          </main>
        ) : (
        <div className="caseboard-shell">
          <aside className="study-tray" aria-label="Study navigation">
            <div className="tray-top">
              <div className="tray-heading">Studies</div>
              <button className="tray-upload" onClick={() => setActiveTab('upload')} title="Upload a study">
                <Upload size={16} />
              </button>
            </div>
            <div className="tray-search">Search studies</div>
            <div className="tray-section-label">AVAILABLE CASES</div>
            <div className="tray-case-list">
              {cases.map((c, index) => (
                <button
                  key={c.case_id}
                  className={`tray-case ${selectedCaseId === c.case_id ? 'selected' : ''}`}
                  onClick={() => {
                    if (c.case_id !== selectedCaseId) {
                      setLoadingCase(true);
                      setLoadingSlice(true);
                      setSliceData(null);
                      setSelectedCaseId(c.case_id);
                    }
                  }}
                >
                  <span className="tray-case-index">{String(index + 1).padStart(2, '0')}</span>
                  <span className="tray-case-content">
                    <strong>{c.case_id}</strong>
                    <small>{c.slice_count} slices · {c.has_tumor ? 'Finding detected' : 'Review pending'}</small>
                  </span>
                  {c.has_tumor && <span className="tray-case-dot" />}
                </button>
              ))}
            </div>
            <div className="tray-footer">
              <div className="tray-footer-label">WORKSPACE</div>
              <span><i className="tray-status-dot" /> {health?.device ? `${health.device.toUpperCase()} connected` : 'Checking connection'}</span>
            </div>
          </aside>
          <main className="main-layout">
          <div className="review-context">
            <div>
              <div className="context-kicker">CASE REVIEW / EVIDENCE WORKSPACE</div>
              <h1>Read the study in layers.</h1>
              <p>Move from image evidence to quantitative findings, then document the clinical interpretation.</p>
            </div>
            <div className="context-notes">
              <span><strong>← →</strong> Navigate slices</span>
              <span><strong>Space</strong> Cine loop</span>
              <span className="context-divider" />
              <span className="review-state"><i /> Review in progress</span>
            </div>
          </div>
          {/* LEFT: CT VIEWER PANEL */}
          <section className="glass-panel viewer-panel">
            {/* Header Controls */}
            <div className="viewer-header">
              <div className="viewer-study-meta">
                <div className="viewer-overline">AXIAL / PORTAL VENOUS PHASE</div>
                <div className="viewer-case-name">{selectedCaseId}</div>
                {loadingCase && <span className="viewer-loading"><RotateCw size={12} className="spin" /> Preparing study</span>}
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
              {loadingCase ? (
                <div className="viewport-loading-overlay">
                  <div className="loading-image-preview-frame" style={{ width: 190, height: 190, marginBottom: 4 }}>
                    <Layers size={52} color="var(--accent-primary)" style={{ opacity: 0.7 }} />
                    <div className="scanner-line" />
                  </div>
                  <div className="loading-image-title" style={{ fontSize: '1.25rem' }}>
                    <RotateCw size={22} className="spin" color="var(--accent-primary)" />
                    Loading Image...
                  </div>
                  <div className="loading-image-subtitle">
                    Loading patient case <strong>{selectedCaseId}</strong> — preparing CT slices & batch segmentation
                  </div>
                  <div className="loading-pulse-bar" style={{ width: 220 }}>
                    <div className="loading-pulse-fill" />
                  </div>
                </div>
              ) : viewMode === 'split' ? (
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
                  {loadingCase ? (
                    <span style={{ color: 'var(--accent-primary)', display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                      <RotateCw size={11} className="spin" /> Loading...
                    </span>
                  ) : (
                    `${sliceIndex + 1} / ${totalSlices}`
                  )}
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
                {loadingCase ? (
                  <span className="badge badge-cyan" style={{ display: 'inline-flex', alignItems: 'center', gap: 5 }}>
                    <RotateCw size={11} className="spin" /> Loading Image...
                  </span>
                ) : metrics3D && (
                  <span className="badge badge-cyan">
                    {metrics3D.inference_time_sec}s 3D Inference
                  </span>
                )}
              </div>

              {loadingCase ? (
                <div style={{ textAlign: 'center', padding: '24px', color: 'var(--text-muted)' }}>
                  <RotateCw size={22} className="spin" color="var(--accent-primary)" style={{ margin: '0 auto 8px' }} />
                  <div style={{ fontWeight: 600, color: 'var(--text-white)' }}>
                    Reconstructing 3D Volume...
                  </div>
                  <div style={{ fontSize: '0.78rem', marginTop: 4 }}>
                    Analyzing anatomy for {selectedCaseId}
                  </div>
                </div>
              ) : metrics3D ? (
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
        </div>
        )
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

          {(health && !health.model_loaded) && (
            <div className="analysis-unavailable" role="status">
              <AlertTriangle size={18} />
              <div>
                <strong>Analysis is not configured on this machine.</strong>
                <span>Add a trained model checkpoint, restart the API, then upload a study.</span>
              </div>
            </div>
          )}

          {uploadError && (
            <div className="upload-error" role="alert">
              <AlertTriangle size={17} />
              <span>{uploadError}</span>
              <button onClick={() => setUploadError('')} aria-label="Dismiss message">×</button>
            </div>
          )}

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

          {/* When Image Is Uploaded: Prominent Loading Image Card */}
          {isUploading && (
            <div className="loading-image-card">
              <div className="loading-image-preview-frame">
                {uploadPreviewUrl ? (
                  <img src={uploadPreviewUrl} alt="Uploaded CT preview" className="loading-image-preview-img" />
                ) : (
                  <Layers size={56} color="var(--accent-primary)" style={{ opacity: 0.6 }} />
                )}
                <div className="scanner-line" />
              </div>

              <div className="loading-image-title">
                <RotateCw className="spin" size={24} color="var(--accent-primary)" />
                Loading Image...
              </div>

              <div className="loading-image-subtitle">
                {uploadProgressText || "Uploading CT scan & executing deep learning segmentation..."}
              </div>

              <div className="loading-pulse-bar">
                <div className="loading-pulse-fill" />
              </div>

              <span className="badge badge-cyan" style={{ fontSize: '0.75rem', padding: '4px 12px' }}>
                PyTorch Neural Core Inference Active
              </span>
            </div>
          )}

          {/* Drag & Drop Zone (Visible when not uploading and no results yet) */}
          {(!isUploading && !uploadedCaseResult && !singleSliceResult) && (
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
