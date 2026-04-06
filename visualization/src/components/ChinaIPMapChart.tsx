import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import * as echarts from 'echarts/core'
import type { EChartsCoreOption } from 'echarts/core'
import { MapChart } from 'echarts/charts'
import { TooltipComponent, VisualMapComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { registerMap } from 'echarts/core'
import { Loader2 } from 'lucide-react'
import { toMapRegionName } from '@/utils/ipLocationMap'

echarts.use([MapChart, TooltipComponent, VisualMapComponent, CanvasRenderer])

const CHINA_GEO_URL = 'https://fastly.jsdelivr.net/npm/echarts@4.9.0/map/json/china.json'

let chinaMapRegistered = false

export interface IPMapDatum {
  rawLocation: string
  count: number
}

interface ChinaIPMapChartProps {
  data: IPMapDatum[]
  maxCount: number
  selectedRaw: string | null
  onRegionClick: (rawLocation: string | null) => void
  className?: string
  height?: number
}

export function ChinaIPMapChart({
  data,
  maxCount,
  selectedRaw,
  onRegionClick,
  className = '',
  height = 380,
}: ChinaIPMapChartProps) {
  const divRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<echarts.ECharts | null>(null)
  const onRegionClickRef = useRef(onRegionClick)
  onRegionClickRef.current = onRegionClick

  const [loadError, setLoadError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  const seriesData = useMemo(() => {
    const m = new Map<string, number>()
    for (const d of data) {
      const k = toMapRegionName(d.rawLocation)
      if (!k) continue
      m.set(k, (m.get(k) || 0) + d.count)
    }
    return Array.from(m.entries()).map(([name, value]) => ({ name, value }))
  }, [data])

  const mapMax = useMemo(
    () => Math.max(maxCount, ...seriesData.map((d) => d.value), 1),
    [seriesData, maxCount],
  )

  const selectedMapName =
    selectedRaw && selectedRaw !== '未知IP' ? toMapRegionName(selectedRaw) : null

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      setLoading(true)
      setLoadError(null)
      try {
        const res = await fetch(CHINA_GEO_URL)
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const geo = await res.json()
        if (cancelled) return
        if (!chinaMapRegistered) {
          registerMap('china', geo as Parameters<typeof registerMap>[1])
          chinaMapRegistered = true
        }
      } catch (e) {
        if (!cancelled) {
          setLoadError(e instanceof Error ? e.message : '地图数据加载失败')
        }
        setLoading(false)
        return
      }
      if (cancelled) return
      setLoading(false)
    })()
    return () => {
      cancelled = true
    }
  }, [])

  const applyChartOption = useCallback(() => {
    if (!chartRef.current || loading || loadError) return
    const option: EChartsCoreOption = {
      tooltip: {
        trigger: 'item',
        formatter: (p: unknown) => {
          const item = p as { name?: string; value?: number }
          const v = item.value ?? 0
          return `${item.name ?? ''}<br/>笔记数 <b>${v}</b>`
        },
      },
      visualMap: {
        min: 0,
        max: mapMax,
        text: ['高', '低'],
        realtime: true,
        calculable: true,
        inRange: {
          color: ['#fce7f3', '#f9a8d4', '#f43f5e', '#9f1239'],
        },
        left: 16,
        bottom: 24,
        textStyle: { fontSize: 11, color: '#6b7280' },
      },
      series: [
        {
          name: '笔记数',
          type: 'map',
          map: 'china',
          roam: true,
          scaleLimit: { min: 0.85, max: 4 },
          label: { show: true, fontSize: 11, color: '#4b5563' },
          emphasis: {
            label: { show: true, color: '#111827' },
            itemStyle: { areaColor: '#fda4af' },
          },
          selectedMode: false,
          data: seriesData,
        },
      ],
    }
    chartRef.current.setOption(option, true)
  }, [loading, loadError, mapMax, seriesData])

  useEffect(() => {
    if (loading || loadError || !divRef.current) return

    const el = divRef.current
    if (!chartRef.current) {
      chartRef.current = echarts.init(el, undefined, { renderer: 'canvas' })
    }
    const chart = chartRef.current

    applyChartOption()

    const onClick = (params: { componentType?: string; name?: string }) => {
      if (params.componentType !== 'series' || !params.name) return
      const hit = data.find((d) => toMapRegionName(d.rawLocation) === params.name)
      onRegionClickRef.current(hit ? hit.rawLocation : null)
    }
    chart.on('click', onClick)

    const ro = new ResizeObserver(() => chart.resize())
    ro.observe(el)

    return () => {
      chart.off('click', onClick)
      ro.disconnect()
    }
  }, [loading, loadError, data, applyChartOption])

  useEffect(() => {
    if (!chartRef.current || loading || loadError) return
    try {
      chartRef.current.dispatchAction({ type: 'downplay', seriesIndex: 0 })
      if (selectedMapName) {
        chartRef.current.dispatchAction({
          type: 'highlight',
          seriesIndex: 0,
          name: selectedMapName,
        })
      }
    } catch {
      /* 区域名不存在于地图时忽略 */
    }
  }, [selectedMapName, loading, loadError, seriesData])

  useEffect(() => {
    return () => {
      chartRef.current?.dispose()
      chartRef.current = null
    }
  }, [])

  if (loadError) {
    return (
      <div
        className={`flex flex-col items-center justify-center rounded-xl border border-amber-100 bg-amber-50/80 text-amber-900 text-sm p-6 ${className}`}
        style={{ minHeight: height }}
      >
        <p className="font-medium mb-1">地图暂不可用</p>
        <p className="text-xs text-amber-800/90 text-center">{loadError}</p>
        <p className="text-xs text-gray-500 mt-2 text-center">请检查网络，或使用下方列表查看分布</p>
      </div>
    )
  }

  if (loading) {
    return (
      <div
        className={`flex items-center justify-center gap-2 text-gray-400 text-sm ${className}`}
        style={{ minHeight: height }}
      >
        <Loader2 className="w-5 h-5 animate-spin" />
        加载地图…
      </div>
    )
  }

  return <div ref={divRef} className={className} style={{ width: '100%', height }} />
}
