import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  Activity,
  Layers,
  Play,
  Pause,
  ChevronLeft,
  ChevronRight,
  Upload,
  FileText,
  Cpu,
  Download,
  Copy,
  Check,
  AlertTriangle,
  RotateCw,
  Eye,
  Sliders,
  FolderPlus,
  Search
} from 'lucide-react';
import './App.css';
import SliceAnalysisPanel from './components/SliceAnalysisPanel';
import ScanWideOverview from './components/ScanWideOverview';

export default function App() {
  // Navigation & View State
  const [activeTab, setActiveTab] = useState('explorer'); // 'explorer' | 'upload'
  const [health, setHealth] = useState(null);
  const [cases, setCases] = useState([]);
  const [searchQuery, setSearchQuery] = useState('');
  // Wait for /api/cases before choosing a study. A hard-coded initial ID caused
  // two whole-volume /full requests to start concurrently on first load.
  const [selectedCaseId, setSelectedCaseId] = useState(null);
  
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
  const [maskOverlayUrl, setMaskOverlayUrl] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);

  // Filtered cases for study tray search
  const filteredCases = useMemo(() => {
    if (!searchQuery.trim()) return cases;
    const q = searchQuery.toLowerCase().trim();
    return cases.filter(c => c.case_id.toLowerCase().includes(q));
  }, [cases, searchQuery]);


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

  // 2. Fetch Entire Case Bundle at once (Batch-computed slices)
  useEffect(() => {
    if (!selectedCaseId) return;

    let active = true;
    const controller = new AbortController();

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
      setLoadingCase(false);
      setLoadingSlice(false);
      return;
    }

    setLoadingCase(true);
    setLoadingSlice(true);
    setCaseSlices([]);

    const caseMeta = cases.find(c => c.case_id === selectedCaseId);
    const estTotal = caseMeta?.slice_count || totalSlices || 50;
    setTotalSlices(estTotal);
    const initialIdx = Math.floor(estTotal / 2);
    setSliceIndex(initialIdx);

    // Fast Path 1: Immediately fetch initial single slice preview (resolves in ~150ms)
    fetch(`/api/case/${selectedCaseId}/slice/${initialIdx}`, { signal: controller.signal })
      .then(res => {
        if (!res.ok) throw new Error(`Status ${res.status}`);
        return res.json();
      })
      .then(initialSlice => {
        if (!active) return;
        setSliceData(prev => {
          if (caseCacheRef.current[selectedCaseId]) return prev;
          return initialSlice;
        });
        setLoadingSlice(false);
      })
      .catch(err => {
        if (err.name === 'AbortError') return;
        console.debug("Initial slice preview notice:", err);
      });

    // Fast Path 2: Concurrently fetch full case bundle for complete volume navigation
    fetch(`/api/case/${selectedCaseId}/full`, { signal: controller.signal })
      .then(res => {
        if (!res.ok) throw new Error(`Status ${res.status}`);
        return res.json();
      })
      .then(data => {
        if (!active) return;
        if (data && data.slices && data.slices.length > 0) {
          caseCacheRef.current[selectedCaseId] = data;
          setCaseSlices(data.slices);
          const total = data.total_slices || data.slices.length;
          setTotalSlices(total);

          // If on initial middle slice, jump to prominent tumor slice if detected
          setSliceIndex(prevIdx => {
            const tumorIdx = data.slices.findIndex(s => (s.tumor_area_cm2 > 0) || (s.pixel_counts && s.pixel_counts.tumor > 0));
            const targetIdx = (prevIdx === initialIdx && tumorIdx !== -1) ? tumorIdx : prevIdx;
            const targetSlice = data.slices[targetIdx] || data.slices.find(s => s.slice_index === targetIdx) || data.slices[0];
            if (targetSlice) {
              setSliceData(targetSlice);
            }
            return targetIdx;
          });
        }
        setLoadingCase(false);
        setLoadingSlice(false);
      })
      .catch(err => {
        if (err.name === 'AbortError') return;
        if (!active) return;
        console.warn("Full case fetch fallback:", err);
        fetch(`/api/case/${selectedCaseId}/slices`, { signal: controller.signal })
          .then(res => res.json())
          .then(data => {
            if (!active) return;
            if (data && data.slices) {
              setTotalSlices(data.total_slices);
            }
            setLoadingCase(false);
          })
          .catch(() => {
            if (active) setLoadingCase(false);
          });
      });

    return () => {
      active = false;
      controller.abort();
    };
  }, [selectedCaseId, cases]);

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
    let active = true;
    const controller = new AbortController();

    if (selectedCaseId && (!caseSlices || caseSlices.length === 0)) {
      setLoadingSlice(true);
      fetch(`/api/case/${selectedCaseId}/slice/${sliceIndex}`, { signal: controller.signal })
        .then(res => res.json())
        .then(data => {
          if (!active) return;
          setSliceData(data);
          setLoadingSlice(false);
        })
        .catch(err => {
          if (err.name === 'AbortError') return;
          if (!active) return;
          console.error("Error fetching slice fallback:", err);
          setLoadingSlice(false);
        });
    }

    return () => {
      active = false;
      controller.abort();
    };
  }, [selectedCaseId, sliceIndex, caseSlices]);

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

  // Build a transparent colour layer from the class-mask so each anatomy
  // control can be toggled independently over the unchanged CT image.
  useEffect(() => {
    const activeTargetSlice = (activeTab === 'explorer')
      ? sliceData
      : (singleSliceResult || uploadedSliceDetail);

    if (!activeTargetSlice || viewMode === 'pure_ct' || viewMode === 'split') {
      setMaskOverlayUrl(null);
      return undefined;
    }

    const maskSource = viewMode === 'gt_overlay'
      ? (activeTargetSlice.ground_truth_mask || activeTargetSlice.predicted_mask)
      : activeTargetSlice.predicted_mask;
    if (!maskSource) {
      setMaskOverlayUrl(null);
      return undefined;
    }

    let cancelled = false;
    const maskImage = new Image();
    maskImage.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = maskImage.naturalWidth;
      canvas.height = maskImage.naturalHeight;
      const context = canvas.getContext('2d');
      context.drawImage(maskImage, 0, 0);
      const pixels = context.getImageData(0, 0, canvas.width, canvas.height);
      const palette = {
        1: [16, 185, 129, 140],
        2: [239, 68, 68, 180],
        3: [59, 130, 246, 160]
      };
      const layerVisibility = [false, visibleLayers.kidney, visibleLayers.tumor, visibleLayers.cyst];

      for (let i = 0; i < pixels.data.length; i += 4) {
        const classId = Math.round(pixels.data[i] / 80);
        const color = palette[classId];
        if (color && layerVisibility[classId]) {
          pixels.data[i] = color[0];
          pixels.data[i + 1] = color[1];
          pixels.data[i + 2] = color[2];
          pixels.data[i + 3] = color[3];
        } else {
          pixels.data[i + 3] = 0;
        }
      }

      context.putImageData(pixels, 0, 0);
      if (!cancelled) setMaskOverlayUrl(canvas.toDataURL('image/png'));
    };
    maskImage.onerror = () => {
      if (!cancelled) setMaskOverlayUrl(null);
    };
    maskImage.src = maskSource;

    return () => { cancelled = true; };
  }, [sliceData, singleSliceResult, uploadedSliceDetail, activeTab, viewMode, visibleLayers]);

  // Keyboard navigation
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (activeTab !== 'explorer') return;
      const target = e.target;
      const isFormElement = target && (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.tagName === 'SELECT' ||
        target.isContentEditable
      );
      if (isFormElement) return;

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

        // Refresh cohort cases in background so uploaded case is available in studies tray,
        // but stay on upload tab so the user sees the result right in Drag & Drop view
        fetch('/api/cases')
          .then(r => r.json())
          .then(cList => {
            setCases(cList);
            setSelectedCaseId(data.case_id);
          })
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
        data.filename = file.name;
        const cid = file.name.replace(/\.[^/.]+$/, "");
        data.case_id = cid;
        data.slice_index = 0;
        setSingleSliceResult(data);
        setIsUploading(false);
        setUploadProgressText('');
        setUploadPreviewUrl(null);

        // Pre-populate client case cache for instant exploration
        caseCacheRef.current[cid] = {
          case_id: cid,
          total_slices: 1,
          slices: [data],
          slice_summary: data.slice_summary ? [data.slice_summary] : []
        };
        // Register in studies tray
        setCases(prev => {
          if (prev.some(c => c.case_id === cid)) return prev;
          return [
            {
              case_id: cid,
              slice_count: 1,
              has_tumor: Boolean(data.slice_summary?.tumor_detected || (data.pixel_counts?.tumor > 0)),
              has_cyst: Boolean(data.slice_summary?.cyst_detected || (data.pixel_counts?.cyst > 0)),
              split: 'custom_upload'
            },
            ...prev
          ];
        });

        // If user was on explorer tab when uploading, immediately activate this slice study
        if (activeTab === 'explorer') {
          setSelectedCaseId(cid);
          setSliceIndex(0);
          setTotalSlices(1);
          setSliceData(data);
          setCaseSlices([data]);
        }
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

  const openSingleSliceInExplorer = () => {
    if (!singleSliceResult) return;
    const cid = singleSliceResult.case_id || 'custom_slice';
    setSelectedCaseId(cid);
    setSliceIndex(0);
    setTotalSlices(1);
    setSliceData(singleSliceResult);
    setCaseSlices([singleSliceResult]);
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
    <div
      className="app-container"
      onDragOver={(e) => {
        e.preventDefault();
        setDragActive(true);
      }}
      onDragLeave={(e) => {
        if (!e.currentTarget.contains(e.relatedTarget)) {
          setDragActive(false);
        }
      }}
      onDrop={handleDrop}
    >
      {dragActive && (
        <div className="global-drop-overlay">
          <div className="global-drop-card glass-panel">
            <Upload size={48} color="var(--accent-primary)" className="bounce-subtle" />
            <h3>Drop CT Scan File or Study Folder Here</h3>
            <p>Instant Residual U-Net AI Segmentation & Slice Prediction Summary</p>
          </div>
        </div>
      )}
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
            <div className="tray-search-box">
              <Search size={13} className="tray-search-icon" />
              <input
                type="text"
                placeholder="Search studies…"
                value={searchQuery}
                onChange={e => setSearchQuery(e.target.value)}
                className="tray-search-input"
                aria-label="Search studies"
              />
              {searchQuery && (
                <button
                  type="button"
                  className="tray-search-clear"
                  onClick={() => setSearchQuery('')}
                  aria-label="Clear search"
                >
                  ×
                </button>
              )}
            </div>
            <div className="tray-section-label">
              AVAILABLE CASES {searchQuery && `(${filteredCases.length})`}
            </div>
            <div className="tray-case-list">
              {filteredCases.length === 0 ? (
                <div className="tray-no-results">
                  <span>No studies match &ldquo;{searchQuery}&rdquo;</span>
                </div>
              ) : (
                filteredCases.map((c, index) => (
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
                ))
              )}
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
                    viewMode === 'pure_ct' ? (
                      <img src={sliceData.original_image} alt="CT Slice" className="viewport-image" />
                    ) : (
                      <>
                        <img src={sliceData.original_image} alt="CT Slice" className="viewport-image" />
                        <img
                          src={maskOverlayUrl || (viewMode === 'gt_overlay'
                            ? (sliceData.ground_truth_overlay || sliceData.overlay_image)
                            : sliceData.overlay_image)}
                          alt="Segmentation overlay"
                          className="viewport-overlay-layer"
                          style={{ opacity }}
                        />
                      </>
                    )
                  )}
                  {!sliceData && !loadingCase && (
                    <div className="viewport-empty-state" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', minHeight: 320, color: 'var(--text-muted)' }}>
                      <Layers size={42} style={{ opacity: 0.35, marginBottom: 8 }} />
                      <span style={{ fontSize: '0.85rem' }}>No slice data available for this position</span>
                    </div>
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
                    aria-label="Segmentation overlay opacity"
                  />
                  <output>{Math.round(opacity * 100)}%</output>
                </div>
              </div>
            </div>
          </section>

          {/* RIGHT: INTEGRATED SLICE ANALYSIS PANEL */}
          <SliceAnalysisPanel
            selectedCaseId={selectedCaseId}
            sliceIndex={sliceIndex}
            totalSlices={totalSlices}
            sliceData={sliceData}
            loadingCase={loadingCase}
            loadingSlice={loadingSlice}
          />

          {/* BOTTOM: UNIFIED SCAN-WIDE ANALYSIS */}
          <ScanWideOverview
            selectedCaseId={selectedCaseId}
            caseSlices={caseSlices}
            sliceIndex={sliceIndex}
            setSliceIndex={setSliceIndex}
            totalSlices={totalSlices}
            loadingCase={loadingCase}
          />
          </main>
        </div>
        )
      ) : (
        /* --- DUAL MODE UPLOAD VIEW: WHOLE CASE FOLDER & SINGLE CT SLICE --- */
        <main className={`upload-container ${(uploadedCaseResult || singleSliceResult) ? 'has-result' : ''}`}>
          {/* Mode Switcher Tabs - Only visible during upload selection */}
          {(!uploadedCaseResult && !singleSliceResult) && (
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
          )}

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
          {/* WHOLE CASE UPLOAD RESULTS DASHBOARD: MATCHES COHORT CASE EXPLORER */}
          {/* ============================================================================== */}
          {uploadedCaseResult && (
            <div className="uploaded-case-card glass-panel">
              {/* Header with Case Metadata & Actions */}
              <div className="uploaded-case-header">
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                    <h2 style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--text-white)', margin: 0 }}>
                      Patient Case: {uploadedCaseResult.case_id}
                    </h2>
                    <span className="badge badge-emerald">3D Volume</span>
                    <span className="badge badge-cyan">{uploadedCaseResult.slices_analyzed} Slices</span>
                    <span className="badge badge-cyan">{uploadedCaseResult.inference_time_sec}s Latency</span>
                  </div>
                </div>

                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <button className="btn-secondary" style={{ padding: '5px 12px', fontSize: '0.78rem' }} onClick={() => { setUploadedCaseResult(null); setUploadedSliceDetail(null); }}>
                    <RotateCw size={13} /> Upload Another Case
                  </button>
                </div>
              </div>

              {/* Two-Column Clinical Workspace: Left Primary CT Viewer, Right Slice Prediction Summary */}
              <div className="main-layout" style={{ maxWidth: '100%', width: '100%', padding: 0, margin: 0 }}>
                {/* LEFT: CT VIEWER PANEL */}
                <section className="glass-panel viewer-panel">
                  <div className="viewer-header">
                    <div className="viewer-study-meta">
                      <div className="viewer-overline">AXIAL CT / WHOLE CASE SERIES</div>
                      <div className="viewer-case-name">{uploadedCaseResult.case_id}</div>
                      {loadingUploadedSlice && (
                        <span className="viewer-loading">
                          <RotateCw size={12} className="spin" /> Updating slice
                        </span>
                      )}
                    </div>
                  </div>

                  {/* Viewport Frame */}
                  <div className="viewport-frame">
                    {loadingUploadedSlice ? (
                      <div className="viewport-loading-overlay">
                        <RotateCw size={22} className="spin" color="var(--accent-primary)" />
                        <span style={{ marginTop: 8, fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                          Loading slice #{uploadedCaseSliceIndex + 1}…
                        </span>
                      </div>
                    ) : (
                      uploadedSliceDetail && (
                        <>
                          <img src={uploadedSliceDetail.original_image} alt="CT Slice" className="viewport-image" />
                          <img
                            src={maskOverlayUrl || uploadedSliceDetail.overlay_image}
                            alt="Segmentation overlay"
                            className="viewport-overlay-layer"
                            style={{ opacity }}
                          />
                        </>
                      )
                    )}
                  </div>

                  {/* Slice Control Bar */}
                  <div className="slice-control-bar">
                    <div className="slice-slider-row">
                      <button
                        className="playback-btn"
                        onClick={() => setIsUploadedPlaying(p => !p)}
                        title={isUploadedPlaying ? "Pause Cine Loop" : "Play Cine Loop"}
                      >
                        {isUploadedPlaying ? <Pause size={15} /> : <Play size={15} />}
                      </button>
                      <button
                        className="playback-btn"
                        onClick={() => setUploadedCaseSliceIndex(prev => Math.max(0, prev - 1))}
                        disabled={uploadedCaseSliceIndex === 0}
                        title="Previous Slice (←)"
                      >
                        <ChevronLeft size={16} />
                      </button>
                      <input
                        type="range"
                        min="0"
                        max={Math.max(0, uploadedCaseResult.slices_analyzed - 1)}
                        value={uploadedCaseSliceIndex}
                        onChange={e => setUploadedCaseSliceIndex(parseInt(e.target.value))}
                      />
                      <button
                        className="playback-btn"
                        onClick={() => setUploadedCaseSliceIndex(prev => Math.min(uploadedCaseResult.slices_analyzed - 1, prev + 1))}
                        disabled={uploadedCaseSliceIndex === uploadedCaseResult.slices_analyzed - 1}
                        title="Next Slice (→)"
                      >
                        <ChevronRight size={16} />
                      </button>
                      <div className="slice-counter">
                        {uploadedCaseSliceIndex + 1} / {uploadedCaseResult.slices_analyzed}
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
                          aria-label="Segmentation overlay opacity"
                        />
                        <output>{Math.round(opacity * 100)}%</output>
                      </div>
                    </div>
                  </div>
                </section>

                {/* RIGHT: INTEGRATED SLICE ANALYSIS PANEL */}
                <SliceAnalysisPanel
                  selectedCaseId={uploadedCaseResult.case_id}
                  sliceIndex={uploadedCaseSliceIndex}
                  totalSlices={uploadedCaseResult.slices_analyzed}
                  sliceData={uploadedSliceDetail}
                  loadingCase={false}
                  loadingSlice={loadingUploadedSlice}
                />
              </div>
            </div>
          )}

          {/* ============================================================================== */}
          {/* SINGLE CT SLICE UPLOAD RESULT: MATCHES COHORT CASE EXPLORER WORKSPACE */}
          {/* ============================================================================== */}
          {singleSliceResult && (
            <div className="uploaded-case-card glass-panel">
              {/* Header with Case Metadata & Actions */}
              <div className="uploaded-case-header">
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                    <h2 style={{ fontSize: '1.1rem', fontWeight: 800, color: 'var(--text-white)', margin: 0 }}>
                      Slice: {singleSliceResult.filename || "Uploaded Slice"}
                    </h2>
                    <span className="badge badge-emerald">2D Analysis</span>
                    <span className="badge badge-cyan">{singleSliceResult.inference_time_ms} ms Latency</span>
                  </div>
                </div>

                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                  <button
                    className="btn-secondary"
                    style={{ padding: '5px 12px', fontSize: '0.78rem' }}
                    onClick={() => {
                      setSingleSliceResult(null);
                    }}
                  >
                    <RotateCw size={13} /> Upload Another Scan
                  </button>
                </div>
              </div>

              {/* Two-Column Clinical Workspace: Left Primary CT Viewer, Right Slice Prediction Summary */}
              <div className="main-layout" style={{ maxWidth: '100%', width: '100%', padding: 0, margin: 0 }}>
                {/* LEFT: CT VIEWER PANEL */}
                <section className="glass-panel viewer-panel">
                  <div className="viewer-header">
                    <div className="viewer-study-meta">
                      <div className="viewer-overline">AXIAL CT / 2D INSTANT PREDICTION</div>
                      <div className="viewer-case-name">{singleSliceResult.filename || "Uploaded Slice"}</div>
                    </div>
                  </div>

                  {/* Viewport Frame */}
                  <div className="viewport-frame">
                    <img src={singleSliceResult.original_image} alt="CT Slice" className="viewport-image" />
                    <img
                      src={maskOverlayUrl || singleSliceResult.overlay_image}
                      alt="Segmentation overlay"
                      className="viewport-overlay-layer"
                      style={{ opacity }}
                    />
                  </div>

                  {/* Slice Control Bar (Single Slice Indicator & Layer Legend) */}
                  <div className="slice-control-bar">
                    <div className="slice-slider-row" style={{ justifyContent: 'space-between' }}>
                      <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 600 }}>
                        Single Axial Slice Study
                      </span>
                      <div className="slice-counter">1 / 1</div>
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
                          aria-label="Segmentation overlay opacity"
                        />
                        <output>{Math.round(opacity * 100)}%</output>
                      </div>
                    </div>
                  </div>
                </section>

                {/* RIGHT: INTEGRATED SLICE ANALYSIS PANEL */}
                <SliceAnalysisPanel
                  selectedCaseId={singleSliceResult.case_id || "single_slice"}
                  sliceIndex={0}
                  totalSlices={1}
                  sliceData={singleSliceResult}
                  loadingCase={false}
                  loadingSlice={false}
                />
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
