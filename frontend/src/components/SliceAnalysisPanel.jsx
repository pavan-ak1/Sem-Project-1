import React, { useMemo } from 'react';
import {
  FileText,
  Download,
  Info,
  CheckCircle2,
  HelpCircle,
  XCircle,
  Sparkles,
  AlertTriangle,
  RotateCw
} from 'lucide-react';
import './SliceAnalysisPanel.css';

/**
 * Standardized Anatomical & Class Palettes
 */
const CLASS_PALETTE = {
  background: {
    id: 0,
    name: 'Background',
    color: '#64748B',
    bgColor: 'rgba(100, 116, 139, 0.12)'
  },
  kidney: {
    id: 1,
    name: 'Kidney',
    color: '#10B981',
    bgColor: 'rgba(16, 185, 129, 0.12)'
  },
  tumor: {
    id: 2,
    name: 'Tumor',
    color: '#EF4444',
    bgColor: 'rgba(239, 68, 68, 0.12)'
  },
  cyst: {
    id: 3,
    name: 'Cyst',
    color: '#3B82F6',
    bgColor: 'rgba(59, 130, 246, 0.12)'
  }
};

export default function SliceAnalysisPanel({
  selectedCaseId,
  sliceIndex = 0,
  totalSlices = 0,
  sliceData = null,
  loadingCase = false,
  loadingSlice = false
}) {
  // Extract and calculate dynamic metrics from actual prediction mask
  const {
    percentages,
    pixelCounts,
    totalPixels,
    kidneyDetected,
    tumorDetected,
    cystDetected,
    kidneyRegions,
    tumorRegions,
    cystRegions,
    totalConnectedRegions,
    summaryText,
    qualityFlags,
    hasMetrics
  } = useMemo(() => {
    if (!sliceData) {
      return {
        percentages: { background: 100, kidney: 0, tumor: 0, cyst: 0 },
        pixelCounts: { background: 0, kidney: 0, tumor: 0, cyst: 0 },
        totalPixels: 0,
        kidneyDetected: false,
        tumorDetected: false,
        cystDetected: false,
        kidneyRegions: null,
        tumorRegions: null,
        cystRegions: null,
        totalConnectedRegions: null,
        summaryText: 'Waiting for slice data…',
        qualityFlags: '',
        hasMetrics: false
      };
    }

    const rawSummary = sliceData.slice_summary || {};
    const px = sliceData.pixel_counts || {};

    // Dynamic pixel sum from actual slice dimensions (never hardcoding 512x512)
    const sumPixels = (px.background || 0) + (px.kidney || 0) + (px.tumor || 0) + (px.cyst || 0);

    // Calculate exact percentages proportional to actual mask pixels
    const bgPct = sumPixels > 0
      ? (px.background / sumPixels) * 100
      : (rawSummary.background_percentage != null ? rawSummary.background_percentage : 100.0);

    const kPct = sumPixels > 0
      ? (px.kidney / sumPixels) * 100
      : (rawSummary.kidney_percentage != null ? rawSummary.kidney_percentage : 0.0);

    const tPct = sumPixels > 0
      ? (px.tumor / sumPixels) * 100
      : (rawSummary.tumor_percentage != null ? rawSummary.tumor_percentage : 0.0);

    const cPct = sumPixels > 0
      ? (px.cyst / sumPixels) * 100
      : (rawSummary.cyst_percentage != null ? rawSummary.cyst_percentage : 0.0);

    // Detection flags
    const kDet = rawSummary.kidney_detected ?? (px.kidney > 0);
    const tDet = rawSummary.tumor_detected ?? (px.tumor > 0);
    const cDet = rawSummary.cyst_detected ?? (px.cyst > 0);

    // Connected regions from backend CCA post-processing
    const kReg = rawSummary.kidney_connected_regions != null ? rawSummary.kidney_connected_regions : (kDet ? 1 : 0);
    const tReg = rawSummary.tumor_connected_regions != null ? rawSummary.tumor_connected_regions : (tDet ? 1 : 0);
    const cReg = rawSummary.cyst_connected_regions != null ? rawSummary.cyst_connected_regions : (cDet ? 1 : 0);
    const totReg = rawSummary.total_connected_regions != null ? rawSummary.total_connected_regions : (kReg + tReg + cReg);

    // Clinical model summary description
    let summary = rawSummary.summary;
    if (!summary) {
      if (!kDet && !tDet && !cDet) {
        summary = 'No qualifying foreground regions detected in this slice.';
      } else if (tDet && !kDet) {
        summary = 'Tumor-class region predicted without a qualifying kidney region.';
      } else if (cDet && !kDet) {
        summary = 'Cyst-class region predicted without a qualifying kidney region.';
      } else if (kDet && !tDet && !cDet) {
        summary = 'Kidney tissue detected. No tumor or cyst class retained after post-processing.';
      } else {
        const labels = [];
        if (kDet) labels.push('kidney');
        if (tDet) labels.push('tumor');
        if (cDet) labels.push('cyst');
        summary = `${labels.join(' and ').replace(/^./, str => str.toUpperCase())} predicted in this slice.`;
      }
    }

    return {
      percentages: {
        background: Number(bgPct),
        kidney: Number(kPct),
        tumor: Number(tPct),
        cyst: Number(cPct)
      },
      pixelCounts: {
        background: px.background || 0,
        kidney: px.kidney || 0,
        tumor: px.tumor || 0,
        cyst: px.cyst || 0
      },
      totalPixels: sumPixels,
      kidneyDetected: Boolean(kDet),
      tumorDetected: Boolean(tDet),
      cystDetected: Boolean(cDet),
      kidneyRegions: kReg,
      tumorRegions: tReg,
      cystRegions: cReg,
      totalConnectedRegions: totReg,
      summaryText: summary,
      qualityFlags: rawSummary.quality_flags || '',
      hasMetrics: true
    };
  }, [sliceData]);

  const activeSliceNum = sliceIndex + 1;

  const handleExportCSV = (e) => {
    if (selectedCaseId && (selectedCaseId.startsWith('case_') || selectedCaseId.startsWith('upload_'))) {
      return;
    }
    e.preventDefault();
    if (!sliceData) return;
    const summary = sliceData.slice_summary || {
      case_id: selectedCaseId || 'custom_slice',
      slice_index: sliceIndex,
      kidney_detected: kidneyDetected,
      tumor_detected: tumorDetected,
      cyst_detected: cystDetected,
      kidney_pixels: pixelCounts.kidney,
      tumor_pixels: pixelCounts.tumor,
      cyst_pixels: pixelCounts.cyst,
      background_pixels: pixelCounts.background,
      kidney_percentage: percentages.kidney,
      tumor_percentage: percentages.tumor,
      cyst_percentage: percentages.cyst,
      background_percentage: percentages.background,
      summary: summaryText
    };
    const headers = Object.keys(summary).join(',');
    const values = Object.values(summary).map(v => typeof v === 'string' ? `"${v.replace(/"/g, '""')}"` : v).join(',');
    const csvContent = "data:text/csv;charset=utf-8," + encodeURIComponent(`${headers}\n${values}\n`);
    const link = document.createElement("a");
    link.setAttribute("href", csvContent);
    link.setAttribute("download", `${selectedCaseId || 'slice'}_prediction_summary.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <section className="glass-panel slice-analysis-panel" aria-label="Integrated Slice Analysis">
      {/* PANEL HEADER */}
      <div className="sap-header">
        <div className="sap-title-block">
          <div className="sap-title-row">
            <FileText size={17} color="var(--accent-primary)" />
            <h3 className="sap-title">Integrated Slice Analysis</h3>
            <span className="sap-slice-badge">
              Slice #{activeSliceNum} {totalSlices > 0 && `of ${totalSlices}`}
            </span>
          </div>
          <p className="sap-disclaimer">
            Automated model predictions for active slice — not a clinical diagnosis.
          </p>
        </div>

        {/* CSV Export Action */}
        <a
          className="btn-secondary sap-export-btn"
          href={`/api/case/${selectedCaseId}/slice-summary.csv`}
          download={`${selectedCaseId}_slice_summary.csv`}
          onClick={handleExportCSV}
          title="Download complete case slice summary CSV"
        >
          <Download size={13} />
          <span>Export CSV</span>
        </a>
      </div>

      {/* LOADING OR CONTENT */}
      {loadingCase ? (
        <div className="sap-loading-state" role="status">
          <RotateCw size={20} className="spin" color="var(--accent-primary)" />
          <span>Analyzing study slices for {selectedCaseId}…</span>
        </div>
      ) : !hasMetrics ? (
        <div className="sap-empty-state" role="status">
          <AlertTriangle size={24} color="var(--accent-amber)" />
          <span>No prediction metrics available for slice #{activeSliceNum}.</span>
        </div>
      ) : (
        <div className="sap-content">
          {/* 1. SLICE STATUS SUMMARY BANNER */}
          <div className={`sap-status-banner ${!kidneyDetected && !tumorDetected && !cystDetected ? 'neutral' : ''}`}>
            <div className="sap-status-icon">
              {loadingSlice ? (
                <RotateCw size={15} className="spin" color="var(--accent-primary)" />
              ) : (
                <Sparkles size={15} color="var(--accent-primary)" />
              )}
            </div>
            <div className="sap-status-body">
              <div className="sap-status-headline">
                {loadingSlice ? 'Updating slice predictions…' : summaryText}
              </div>
              {qualityFlags && (
                <div className="sap-quality-flag">
                  Notice: {qualityFlags.replace(/_/g, ' ')}
                </div>
              )}
            </div>
          </div>

          {/* 2. COMPACT STATISTIC CARDS (2x2 GRID) */}
          <div className="sap-stats-grid">
            {/* Kidney Detected */}
            <div className={`sap-stat-card ${kidneyDetected ? 'detected-kidney' : 'not-detected'}`}>
              <div className="sap-stat-top">
                <span className="sap-stat-label">Kidney Detected</span>
                {kidneyDetected ? (
                  <CheckCircle2 size={15} color="var(--color-kidney)" />
                ) : (
                  <XCircle size={15} color="var(--text-muted)" />
                )}
              </div>
              <div className="sap-stat-val">
                {kidneyDetected ? 'Yes' : 'No'}
              </div>
              <div className="sap-stat-sub">
                {kidneyDetected ? 'Parenchyma in slice' : 'Not detected in slice'}
              </div>
            </div>

            {/* Tumor Predicted */}
            <div className={`sap-stat-card ${tumorDetected ? 'detected-tumor' : 'not-detected'}`}>
              <div className="sap-stat-top">
                <span className="sap-stat-label">Tumor Predicted</span>
                {tumorDetected ? (
                  <CheckCircle2 size={15} color="var(--color-tumor)" />
                ) : (
                  <HelpCircle size={15} color="var(--text-muted)" />
                )}
              </div>
              <div className="sap-stat-val">
                {tumorDetected ? 'Yes' : 'Not predicted'}
              </div>
              <div className="sap-stat-sub">
                {tumorDetected ? 'Renal neoplasm class' : 'Class not predicted'}
              </div>
            </div>

            {/* Cyst Predicted */}
            <div className={`sap-stat-card ${cystDetected ? 'detected-cyst' : 'not-detected'}`}>
              <div className="sap-stat-top">
                <span className="sap-stat-label">Cyst Predicted</span>
                {cystDetected ? (
                  <CheckCircle2 size={15} color="var(--color-cyst)" />
                ) : (
                  <HelpCircle size={15} color="var(--text-muted)" />
                )}
              </div>
              <div className="sap-stat-val">
                {cystDetected ? 'Yes' : 'Not predicted'}
              </div>
              <div className="sap-stat-sub">
                {cystDetected ? 'Fluid-attenuated cyst' : 'Class not predicted'}
              </div>
            </div>

            {/* Qualifying Connected Regions */}
            <div className="sap-stat-card sap-stat-regions">
              <div className="sap-stat-top">
                <span className="sap-stat-label">Qualifying Regions</span>
                <Info size={14} color="var(--accent-primary)" />
              </div>
              <div className="sap-stat-val">
                {totalConnectedRegions != null ? totalConnectedRegions : '—'}
              </div>
              <div className="sap-stat-sub">
                Connected 2D components
              </div>
            </div>
          </div>

          {/* 3. PIXEL CLASS DISTRIBUTION (HORIZONTAL BAR CHART) */}
          <div className="sap-box">
            <div className="sap-box-header">
              <span className="sap-box-title">Pixel Class Distribution</span>
              <span className="sap-box-meta">
                {totalPixels > 0 ? `${totalPixels.toLocaleString()} total pixels` : 'Normalized mask'}
              </span>
            </div>

            <div className="sap-bar-chart" role="region" aria-label="Pixel class percentage distribution">
              {/* Background */}
              <div
                className="sap-bar-row"
                title={`Background: ${pixelCounts.background.toLocaleString()} pixels (${percentages.background.toFixed(2)}%)`}
              >
                <div className="sap-bar-lead">
                  <span className="sap-swatch" style={{ backgroundColor: CLASS_PALETTE.background.color }} />
                  <span className="sap-class-label">Background</span>
                </div>
                <div className="sap-track">
                  <div
                    className="sap-fill"
                    style={{
                      width: `${Math.min(100, Math.max(0, percentages.background))}%`,
                      backgroundColor: CLASS_PALETTE.background.color
                    }}
                    role="progressbar"
                    aria-valuenow={percentages.background}
                    aria-valuemin="0"
                    aria-valuemax="100"
                  />
                </div>
                <div className="sap-pct-val">
                  {percentages.background.toFixed(2)}%
                </div>
              </div>

              {/* Kidney */}
              <div
                className="sap-bar-row"
                title={`Kidney: ${pixelCounts.kidney.toLocaleString()} pixels (${percentages.kidney.toFixed(2)}%)`}
              >
                <div className="sap-bar-lead">
                  <span className="sap-swatch" style={{ backgroundColor: CLASS_PALETTE.kidney.color }} />
                  <span className="sap-class-label">Kidney</span>
                </div>
                <div className="sap-track">
                  <div
                    className="sap-fill"
                    style={{
                      width: `${Math.min(100, Math.max(0, percentages.kidney))}%`,
                      backgroundColor: CLASS_PALETTE.kidney.color
                    }}
                    role="progressbar"
                    aria-valuenow={percentages.kidney}
                    aria-valuemin="0"
                    aria-valuemax="100"
                  />
                </div>
                <div
                  className="sap-pct-val"
                  style={{ color: percentages.kidney > 0 ? CLASS_PALETTE.kidney.color : 'inherit' }}
                >
                  {percentages.kidney.toFixed(2)}%
                </div>
              </div>

              {/* Tumor */}
              <div
                className="sap-bar-row"
                title={`Tumor: ${pixelCounts.tumor.toLocaleString()} pixels (${percentages.tumor.toFixed(2)}%)`}
              >
                <div className="sap-bar-lead">
                  <span className="sap-swatch" style={{ backgroundColor: CLASS_PALETTE.tumor.color }} />
                  <span className="sap-class-label">Tumor</span>
                </div>
                <div className="sap-track">
                  <div
                    className="sap-fill"
                    style={{
                      width: `${Math.min(100, Math.max(0, percentages.tumor))}%`,
                      backgroundColor: CLASS_PALETTE.tumor.color
                    }}
                    role="progressbar"
                    aria-valuenow={percentages.tumor}
                    aria-valuemin="0"
                    aria-valuemax="100"
                  />
                </div>
                <div
                  className="sap-pct-val"
                  style={{ color: percentages.tumor > 0 ? CLASS_PALETTE.tumor.color : 'inherit' }}
                >
                  {percentages.tumor.toFixed(2)}%
                </div>
              </div>

              {/* Cyst */}
              <div
                className="sap-bar-row"
                title={`Cyst: ${pixelCounts.cyst.toLocaleString()} pixels (${percentages.cyst.toFixed(2)}%)`}
              >
                <div className="sap-bar-lead">
                  <span className="sap-swatch" style={{ backgroundColor: CLASS_PALETTE.cyst.color }} />
                  <span className="sap-class-label">Cyst</span>
                </div>
                <div className="sap-track">
                  <div
                    className="sap-fill"
                    style={{
                      width: `${Math.min(100, Math.max(0, percentages.cyst))}%`,
                      backgroundColor: CLASS_PALETTE.cyst.color
                    }}
                    role="progressbar"
                    aria-valuenow={percentages.cyst}
                    aria-valuemin="0"
                    aria-valuemax="100"
                  />
                </div>
                <div
                  className="sap-pct-val"
                  style={{ color: percentages.cyst > 0 ? CLASS_PALETTE.cyst.color : 'inherit' }}
                >
                  {percentages.cyst.toFixed(2)}%
                </div>
              </div>
            </div>
          </div>

          {/* 4. CONNECTED QUALIFYING REGIONS */}
          <div className="sap-box sap-regions-box">
            <div className="sap-box-header">
              <div className="sap-title-with-info">
                <span className="sap-box-title">Connected Qualifying Regions</span>
                <div
                  className="sap-info-wrap"
                  title="Connected predicted regions retained after project morphological filtering (minimum component size threshold and anatomical kidney containment). Does not reflect total pixel count."
                >
                  <Info size={13} color="var(--text-muted)" />
                </div>
              </div>
              <span className="sap-box-meta">Post-CCA retained</span>
            </div>

            <div className="sap-region-grid">
              <div className="sap-region-tile kidney-tile">
                <span className="sap-region-name">Kidney</span>
                <span className="sap-region-count">
                  {kidneyRegions != null ? kidneyRegions : '—'}
                </span>
                <span className="sap-region-sub">
                  {kidneyDetected ? 'Parenchyma region' : 'None detected'}
                </span>
              </div>

              <div className="sap-region-tile tumor-tile">
                <span className="sap-region-name">Tumor</span>
                <span className="sap-region-count">
                  {tumorRegions != null ? tumorRegions : '—'}
                </span>
                <span className="sap-region-sub">
                  {tumorDetected ? 'Neoplasm lesion' : 'None predicted'}
                </span>
              </div>

              <div className="sap-region-tile cyst-tile">
                <span className="sap-region-name">Cyst</span>
                <span className="sap-region-count">
                  {cystRegions != null ? cystRegions : '—'}
                </span>
                <span className="sap-region-sub">
                  {cystDetected ? 'Benign fluid region' : 'None predicted'}
                </span>
              </div>
            </div>

            <p className="sap-regions-explainer">
              Connected components retained after size filtering and anatomical renal envelope containment.
            </p>
          </div>
        </div>
      )}
    </section>
  );
}
