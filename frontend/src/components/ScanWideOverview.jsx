import React, { useState, useMemo, useRef } from 'react';
import {
  Activity,
  Table as TableIcon,
  Download,
  RotateCcw,
  AlertTriangle
} from 'lucide-react';
import './ScanWideOverview.css';

const CLASS_COLORS = {
  kidney: '#10B981',
  tumor: '#EF4444',
  cyst: '#3B82F6',
  background: '#64748B'
};

export default function ScanWideOverview({
  selectedCaseId,
  caseSlices = [],
  sliceIndex = 0,
  setSliceIndex,
  totalSlices = 0,
  loadingCase = false
}) {
  const [viewMode, setViewMode] = useState('chart'); // 'chart' | 'table'
  const [sliceFilter, setSliceFilter] = useState('all'); // 'all' | 'kidney' | 'tumor' | 'cyst' | 'no_foreground'
  const [hoveredPoint, setHoveredPoint] = useState(null);
  const svgRef = useRef(null);

  // Normalize all rows from caseSlices
  const allRows = useMemo(() => {
    if (!caseSlices || caseSlices.length === 0) return [];

    return caseSlices.map((slice, position) => {
      const raw = slice.slice_summary || {};
      const px = slice.pixel_counts || {};
      const totalPx = (px.background || 0) + (px.kidney || 0) + (px.tumor || 0) + (px.cyst || 0);

      const bgPct = totalPx > 0
        ? (px.background / totalPx) * 100
        : (raw.background_percentage ?? 100);

      const kPct = totalPx > 0
        ? (px.kidney / totalPx) * 100
        : (raw.kidney_percentage ?? 0);

      const tPct = totalPx > 0
        ? (px.tumor / totalPx) * 100
        : (raw.tumor_percentage ?? 0);

      const cPct = totalPx > 0
        ? (px.cyst / totalPx) * 100
        : (raw.cyst_percentage ?? 0);

      const kDet = raw.kidney_detected ?? (px.kidney > 0);
      const tDet = raw.tumor_detected ?? (px.tumor > 0);
      const cDet = raw.cyst_detected ?? (px.cyst > 0);

      const kReg = raw.kidney_connected_regions ?? (kDet ? 1 : 0);
      const tReg = raw.tumor_connected_regions ?? (tDet ? 1 : 0);
      const cReg = raw.cyst_connected_regions ?? (cDet ? 1 : 0);
      const totReg = raw.total_connected_regions ?? (kReg + tReg + cReg);

      let summaryText = raw.summary;
      if (!summaryText) {
        if (!kDet && !tDet && !cDet) {
          summaryText = 'No foreground region retained.';
        } else if (tDet && !kDet) {
          summaryText = 'Tumor-class region predicted without a qualifying kidney region.';
        } else if (cDet && !kDet) {
          summaryText = 'Cyst-class region predicted without a qualifying kidney region.';
        } else if (kDet && !tDet && !cDet) {
          summaryText = 'Kidney tissue detected. No tumor or cyst class retained.';
        } else {
          const labels = [];
          if (kDet) labels.push('kidney');
          if (tDet) labels.push('tumor');
          if (cDet) labels.push('cyst');
          summaryText = `${labels.join(' and ').replace(/^./, str => str.toUpperCase())} predicted.`;
        }
      }

      return {
        case_id: raw.case_id || slice.case_id || selectedCaseId || 'case',
        slice_index: raw.slice_index ?? slice.slice_index ?? position,
        _position: position,
        kidney_detected: Boolean(kDet),
        tumor_detected: Boolean(tDet),
        cyst_detected: Boolean(cDet),
        background_percentage: Number(bgPct),
        kidney_percentage: Number(kPct),
        tumor_percentage: Number(tPct),
        cyst_percentage: Number(cPct),
        kidney_connected_regions: Number(kReg),
        tumor_connected_regions: Number(tReg),
        cyst_connected_regions: Number(cReg),
        total_connected_regions: Number(totReg),
        summary: summaryText,
        quality_flags: raw.quality_flags || ''
      };
    });
  }, [caseSlices, selectedCaseId]);

  // Filtered rows for table view
  const filteredRows = useMemo(() => {
    return allRows.filter(row => {
      if (sliceFilter === 'kidney') return row.kidney_detected;
      if (sliceFilter === 'tumor') return row.tumor_detected;
      if (sliceFilter === 'cyst') return row.cyst_detected;
      if (sliceFilter === 'no_foreground') {
        return !row.kidney_detected && !row.tumor_detected && !row.cyst_detected;
      }
      return true;
    });
  }, [allRows, sliceFilter]);

  // Chart series computation
  const chartData = useMemo(() => {
    if (!allRows.length) return null;

    const points = allRows.map((r, i) => ({
      index: r.slice_index ?? i,
      position: r._position ?? i,
      kidney: r.kidney_percentage || 0,
      tumor: r.tumor_percentage || 0,
      cyst: r.cyst_percentage || 0,
      summary: r
    }));

    // Find dynamic maximum percentage for optimal visual scale
    const maxVal = Math.max(
      10,
      ...points.map(p => Math.max(p.kidney, p.tumor, p.cyst))
    );
    const yMax = Math.ceil((maxVal * 1.15) / 5) * 5;

    return { points, yMax };
  }, [allRows]);

  // SVG dimensions
  const svgWidth = 860;
  const svgHeight = 160;
  const padLeft = 52;
  const padRight = 30;
  const padTop = 22;
  const padBottom = 28;
  const chartW = svgWidth - padLeft - padRight;
  const chartH = svgHeight - padTop - padBottom;

  // Build SVG path strings
  const linePaths = useMemo(() => {
    if (!chartData || !chartData.points.length) return null;
    const { points, yMax } = chartData;
    const total = points.length;

    const computeX = pos => {
      if (total <= 1) return padLeft + chartW / 2;
      return padLeft + (pos / (total - 1)) * chartW;
    };

    const computeY = val => {
      const clamped = Math.max(0, Math.min(yMax, val));
      return padTop + chartH - (clamped / yMax) * chartH;
    };

    const buildPath = key => {
      return points
        .map((p, i) => {
          const x = computeX(i);
          const y = computeY(p[key]);
          return `${i === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
        })
        .join(' ');
    };

    const buildArea = key => {
      const lineStr = buildPath(key);
      const firstX = computeX(0);
      const lastX = computeX(total - 1);
      const baseLineY = padTop + chartH;
      return `${lineStr} L ${lastX.toFixed(1)} ${baseLineY} L ${firstX.toFixed(1)} ${baseLineY} Z`;
    };

    return {
      kidney: buildPath('kidney'),
      tumor: buildPath('tumor'),
      cyst: buildPath('cyst'),
      kidneyArea: buildArea('kidney'),
      tumorArea: buildArea('tumor'),
      cystArea: buildArea('cyst')
    };
  }, [chartData, chartH, chartW]);

  // Click on chart to navigate
  const handleChartClick = e => {
    if (!chartData || !setSliceIndex || !svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const relativeX = (clickX / rect.width) * svgWidth - padLeft;
    const ratio = Math.max(0, Math.min(1, relativeX / chartW));
    const targetIdx = Math.round(ratio * (chartData.points.length - 1));
    if (targetIdx >= 0 && targetIdx < chartData.points.length) {
      setSliceIndex(targetIdx);
    }
  };

  // Hover on chart for tooltip
  const handleChartMouseMove = e => {
    if (!chartData || !svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const relativeX = (clickX / rect.width) * svgWidth - padLeft;
    const ratio = Math.max(0, Math.min(1, relativeX / chartW));
    const targetIdx = Math.round(ratio * (chartData.points.length - 1));
    if (targetIdx >= 0 && targetIdx < chartData.points.length) {
      setHoveredPoint(chartData.points[targetIdx]);
    }
  };

  const handleChartMouseLeave = () => {
    setHoveredPoint(null);
  };

  const getPointX = pos => {
    if (!chartData || chartData.points.length <= 1) return padLeft + chartW / 2;
    return padLeft + (pos / (chartData.points.length - 1)) * chartW;
  };

  const getPointY = val => {
    if (!chartData) return padTop + chartH;
    const clamped = Math.max(0, Math.min(chartData.yMax, val));
    return padTop + chartH - (clamped / chartData.yMax) * chartH;
  };

  const activePoint = chartData?.points[sliceIndex] || null;

  return (
    <section className="glass-panel scanwide-panel" aria-label="Unified Scan-Wide Analysis">
      {/* HEADER SECTION */}
      <div className="swo-header">
        <div className="swo-title-group">
          <div className="swo-title-row">
            <Activity size={17} color="var(--accent-primary)" />
            <h3 className="swo-title">Scan-Wide Class Percentage Profile</h3>
          </div>
          <p className="swo-subtitle">
            Cross-slice pixel distribution across study {selectedCaseId} ({totalSlices || allRows.length} slices). Click any point or navigate via controls to inspect slices.
          </p>
        </div>

        <div className="swo-controls-row">
          {/* View Mode Toggle: Chart vs Table */}
          <div className="swo-view-toggle" role="tablist" aria-label="Scan-wide view mode">
            <button
              type="button"
              role="tab"
              aria-selected={viewMode === 'chart'}
              className={`swo-toggle-btn ${viewMode === 'chart' ? 'active' : ''}`}
              onClick={() => setViewMode('chart')}
              title="Continuous cross-slice trend profile"
            >
              <Activity size={13} />
              <span>Profile Chart</span>
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={viewMode === 'table'}
              className={`swo-toggle-btn ${viewMode === 'table' ? 'active' : ''}`}
              onClick={() => setViewMode('table')}
              title="Full tabular study overview"
            >
              <TableIcon size={13} />
              <span>Table View</span>
            </button>
          </div>

          {/* Table filter dropdown (visible when in table mode) */}
          {viewMode === 'table' && (
            <select
              className="select-input swo-filter-select"
              value={sliceFilter}
              onChange={e => setSliceFilter(e.target.value)}
              aria-label="Filter slices by prediction class"
            >
              <option value="all">All slices ({allRows.length})</option>
              <option value="kidney">Kidney detected</option>
              <option value="tumor">Tumor predicted</option>
              <option value="cyst">Cyst predicted</option>
              <option value="no_foreground">No foreground</option>
            </select>
          )}

          {/* Export CSV Button */}
          <a
            className="btn-secondary swo-export-btn"
            href={`/api/case/${selectedCaseId}/slice-summary.csv`}
            download={`${selectedCaseId}_slice_summary.csv`}
            title="Download full slice summary CSV"
          >
            <Download size={13} />
            <span>Export CSV</span>
          </a>
        </div>
      </div>

      {/* CONTENT: CHART OR COMPACT TABLE */}
      {loadingCase ? (
        <div className="swo-loading-skeleton" role="status">
          <span>Preparing whole-case prediction profile…</span>
        </div>
      ) : !allRows.length ? (
        <div className="swo-empty-state" role="status">
          <AlertTriangle size={24} color="var(--accent-amber)" />
          <span>No scan-wide prediction data available for {selectedCaseId}.</span>
        </div>
      ) : viewMode === 'chart' ? (
        /* 1. CONTINUOUS SVG LINE CHART VIEW */
        <div className="swo-chart-container">
          {/* Chart Header Legend */}
          <div className="swo-legend-row">
            <div className="swo-legend-item">
              <span className="swo-legend-line" style={{ backgroundColor: CLASS_COLORS.kidney }} />
              <span className="swo-legend-name">Kidney Parenchyma</span>
            </div>
            <div className="swo-legend-item">
              <span className="swo-legend-line" style={{ backgroundColor: CLASS_COLORS.tumor }} />
              <span className="swo-legend-name">Renal Neoplasm (Tumor)</span>
            </div>
            <div className="swo-legend-item">
              <span className="swo-legend-line" style={{ backgroundColor: CLASS_COLORS.cyst }} />
              <span className="swo-legend-name">Renal Cyst</span>
            </div>
            <div className="swo-legend-item">
              <span className="swo-legend-cursor-sample" />
              <span className="swo-legend-name">Selected Slice (#{sliceIndex + 1})</span>
            </div>
          </div>

          {/* SVG Chart */}
          <div className="swo-svg-frame">
            <svg
              ref={svgRef}
              viewBox={`0 0 ${svgWidth} ${svgHeight}`}
              className="swo-svg-canvas"
              onClick={handleChartClick}
              onMouseMove={handleChartMouseMove}
              onMouseLeave={handleChartMouseLeave}
              preserveAspectRatio="none"
              aria-label="Interactive CT scan-wide percentage profile chart"
            >
              <defs>
                <linearGradient id="swoKidneyGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={CLASS_COLORS.kidney} stopOpacity="0.22" />
                  <stop offset="100%" stopColor={CLASS_COLORS.kidney} stopOpacity="0.01" />
                </linearGradient>
                <linearGradient id="swoTumorGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={CLASS_COLORS.tumor} stopOpacity="0.22" />
                  <stop offset="100%" stopColor={CLASS_COLORS.tumor} stopOpacity="0.01" />
                </linearGradient>
                <linearGradient id="swoCystGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={CLASS_COLORS.cyst} stopOpacity="0.22" />
                  <stop offset="100%" stopColor={CLASS_COLORS.cyst} stopOpacity="0.01" />
                </linearGradient>
              </defs>

              {/* Horizontal Percentage Gridlines */}
              {[0, 0.25, 0.5, 0.75, 1.0].map((frac, idx) => {
                const y = padTop + chartH * (1 - frac);
                const labelVal = (chartData.yMax * frac).toFixed(0);
                return (
                  <g key={idx}>
                    <line
                      x1={padLeft}
                      y1={y}
                      x2={padLeft + chartW}
                      y2={y}
                      stroke="#E2E8F0"
                      strokeDasharray={idx === 0 ? 'none' : '3,3'}
                      strokeWidth="1"
                    />
                    <text
                      x={padLeft - 8}
                      y={y + 4}
                      textAnchor="end"
                      fontSize="9"
                      fill="#718096"
                      fontFamily="var(--font-mono, monospace)"
                    >
                      {labelVal}%
                    </text>
                  </g>
                );
              })}

              {/* X-axis Slice Index Ticks */}
              {(() => {
                const total = chartData.points.length;
                const step = Math.max(1, Math.floor(total / 8));
                const ticks = [];
                for (let i = 0; i < total; i += step) {
                  ticks.push(i);
                }
                if (ticks[ticks.length - 1] !== total - 1) {
                  ticks.push(total - 1);
                }
                return ticks.map(idx => {
                  const x = getPointX(idx);
                  const sliceNum = chartData.points[idx].index;
                  return (
                    <g key={idx}>
                      <line
                        x1={x}
                        y1={padTop + chartH}
                        x2={x}
                        y2={padTop + chartH + 4}
                        stroke="#CBD5E1"
                        strokeWidth="1"
                      />
                      <text
                        x={x}
                        y={padTop + chartH + 16}
                        textAnchor="middle"
                        fontSize="9"
                        fill="#718096"
                        fontFamily="var(--font-mono, monospace)"
                      >
                        #{sliceNum}
                      </text>
                    </g>
                  );
                });
              })()}

              {/* Shaded Areas */}
              {linePaths?.kidneyArea && (
                <path d={linePaths.kidneyArea} fill="url(#swoKidneyGrad)" />
              )}
              {linePaths?.cystArea && (
                <path d={linePaths.cystArea} fill="url(#swoCystGrad)" />
              )}
              {linePaths?.tumorArea && (
                <path d={linePaths.tumorArea} fill="url(#swoTumorGrad)" />
              )}

              {/* Series Stroke Lines */}
              {linePaths?.kidney && (
                <path
                  d={linePaths.kidney}
                  fill="none"
                  stroke={CLASS_COLORS.kidney}
                  strokeWidth="2.2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              )}
              {linePaths?.cyst && (
                <path
                  d={linePaths.cyst}
                  fill="none"
                  stroke={CLASS_COLORS.cyst}
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              )}
              {linePaths?.tumor && (
                <path
                  d={linePaths.tumor}
                  fill="none"
                  stroke={CLASS_COLORS.tumor}
                  strokeWidth="2.2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              )}

              {/* Active Slice Vertical Guide Line */}
              {activePoint && (
                <g className="swo-active-indicator">
                  <line
                    x1={getPointX(activePoint.position)}
                    y1={padTop}
                    x2={getPointX(activePoint.position)}
                    y2={padTop + chartH}
                    stroke="var(--accent-primary, #2563EB)"
                    strokeWidth="2"
                    strokeDasharray="4,3"
                  />
                  {/* Selected slice badge tag */}
                  <rect
                    x={getPointX(activePoint.position) - 22}
                    y={padTop - 18}
                    width="44"
                    height="15"
                    rx="4"
                    fill="var(--accent-primary, #2563EB)"
                  />
                  <text
                    x={getPointX(activePoint.position)}
                    y={padTop - 7}
                    textAnchor="middle"
                    fill="#FFFFFF"
                    fontSize="9"
                    fontWeight="700"
                    fontFamily="var(--font-mono, monospace)"
                  >
                    #{activePoint.index}
                  </text>

                  {/* Marker dots on series curves */}
                  <circle
                    cx={getPointX(activePoint.position)}
                    cy={getPointY(activePoint.kidney)}
                    r="3.5"
                    fill={CLASS_COLORS.kidney}
                    stroke="#FFFFFF"
                    strokeWidth="1.5"
                  />
                  {activePoint.tumor > 0 && (
                    <circle
                      cx={getPointX(activePoint.position)}
                      cy={getPointY(activePoint.tumor)}
                      r="3.5"
                      fill={CLASS_COLORS.tumor}
                      stroke="#FFFFFF"
                      strokeWidth="1.5"
                    />
                  )}
                  {activePoint.cyst > 0 && (
                    <circle
                      cx={getPointX(activePoint.position)}
                      cy={getPointY(activePoint.cyst)}
                      r="3.5"
                      fill={CLASS_COLORS.cyst}
                      stroke="#FFFFFF"
                      strokeWidth="1.5"
                    />
                  )}
                </g>
              )}

              {/* Hover Cursor */}
              {hoveredPoint && (
                <line
                  x1={getPointX(hoveredPoint.position)}
                  y1={padTop}
                  x2={getPointX(hoveredPoint.position)}
                  y2={padTop + chartH}
                  stroke="#94A3B8"
                  strokeWidth="1"
                  strokeDasharray="2,2"
                />
              )}
            </svg>

            {/* Hover Floating Tooltip */}
            {hoveredPoint && (
              <div
                className="swo-tooltip"
                style={{
                  left: `${((getPointX(hoveredPoint.position) / svgWidth) * 100).toFixed(1)}%`
                }}
              >
                <div className="swo-tooltip-head">Slice #{hoveredPoint.index}</div>
                <div className="swo-tooltip-stat" style={{ color: CLASS_COLORS.kidney }}>
                  <span>Kidney:</span>
                  <strong>{hoveredPoint.kidney.toFixed(2)}%</strong>
                </div>
                <div className="swo-tooltip-stat" style={{ color: CLASS_COLORS.tumor }}>
                  <span>Tumor:</span>
                  <strong>{hoveredPoint.tumor.toFixed(2)}%</strong>
                </div>
                <div className="swo-tooltip-stat" style={{ color: CLASS_COLORS.cyst }}>
                  <span>Cyst:</span>
                  <strong>{hoveredPoint.cyst.toFixed(2)}%</strong>
                </div>
                <div className="swo-tooltip-hint">Click point to jump</div>
              </div>
            )}
          </div>
        </div>
      ) : (
        /* 2. COMPACT TABLE VIEW */
        <div className="swo-table-container">
          <div className="swo-table-bar">
            <span>
              Showing <strong>{filteredRows.length}</strong> of <strong>{allRows.length}</strong> study slices
              {sliceFilter !== 'all' && ` (Filtered: ${sliceFilter.replace(/_/g, ' ')})`}
            </span>
            {filteredRows.length !== allRows.length && (
              <button
                type="button"
                className="swo-reset-btn"
                onClick={() => setSliceFilter('all')}
              >
                <RotateCcw size={11} />
                <span>Show all slices</span>
              </button>
            )}
          </div>

          {filteredRows.length ? (
            <div className="swo-table-wrap">
              <table className="swo-table">
                <thead>
                  <tr>
                    <th>Slice</th>
                    <th>Predicted classes</th>
                    <th>Class percentage (B/K/T/C)</th>
                    <th>Qualifying regions (K/T/C)</th>
                    <th>Summary</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredRows.map(row => {
                    const isSelected = row._position === sliceIndex;
                    return (
                      <tr
                        key={`${row.case_id}-${row.slice_index}`}
                        className={isSelected ? 'selected' : ''}
                        onClick={() => {
                          if (setSliceIndex) setSliceIndex(row._position);
                        }}
                        tabIndex={0}
                        role="button"
                        aria-label={`Select slice ${row.slice_index}`}
                        onKeyDown={e => {
                          if ((e.key === 'Enter' || e.key === ' ') && setSliceIndex) {
                            setSliceIndex(row._position);
                          }
                        }}
                      >
                        <td className="swo-slice-col">
                          <strong>#{row.slice_index}</strong>
                          {isSelected && <span className="swo-active-badge">SELECTED</span>}
                        </td>
                        <td>
                          <span className={row.kidney_detected ? 'summary-flag kidney' : 'summary-flag'}>
                            K {row.kidney_detected ? 'yes' : '—'}
                          </span>
                          <span className={row.tumor_detected ? 'summary-flag tumor' : 'summary-flag'}>
                            T {row.tumor_detected ? 'yes' : '—'}
                          </span>
                          <span className={row.cyst_detected ? 'summary-flag cyst' : 'summary-flag'}>
                            C {row.cyst_detected ? 'yes' : '—'}
                          </span>
                        </td>
                        <td className="swo-mono-col">
                          {row.background_percentage.toFixed(1)} / {row.kidney_percentage.toFixed(1)} / {row.tumor_percentage.toFixed(1)} / {row.cyst_percentage.toFixed(1)}%
                        </td>
                        <td className="swo-mono-col">
                          {row.kidney_connected_regions} / {row.tumor_connected_regions} / {row.cyst_connected_regions}
                        </td>
                        <td>
                          <span>{row.summary}</span>
                          {row.quality_flags && (
                            <small className="summary-quality"> · {row.quality_flags}</small>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="swo-empty-filter">
              No slices match the "{sliceFilter.replace(/_/g, ' ')}" filter.
            </div>
          )}
        </div>
      )}
    </section>
  );
}
