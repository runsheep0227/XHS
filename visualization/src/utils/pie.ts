// 饼图计算工具函数（从 PieChart.tsx 迁出，解决 react-refresh 冲突）

export interface PieSlice {
  id: string | number
  name: string
  d: string
  fill: string
}

export function computePieSlices(
  data: { id: string | number; name: string; value: number; color?: string }[],
  cx = 200,
  cy = 200,
  r = 120
): (PieSlice & { value: number })[] {
  const total = data.reduce((s, d) => s + d.value, 0)
  let curAngle = 0
  return data.map((d) => {
    const angle = (d.value / total) * 360
    const sRad = ((curAngle - 90) * Math.PI) / 180
    const eRad = ((curAngle + angle - 90) * Math.PI) / 180
    const x1 = cx + r * Math.cos(sRad)
    const y1 = cy + r * Math.sin(sRad)
    const x2 = cx + r * Math.cos(eRad)
    const y2 = cy + r * Math.sin(eRad)
    const largeArc = angle > 180 ? 1 : 0
    const dPath = `M ${cx} ${cy} L ${x1} ${y1} A ${r} ${r} 0 ${largeArc} 1 ${x2} ${y2} Z`
    curAngle += angle
    return { id: d.id, name: d.name, d: dPath, fill: d.color || '#888', value: d.value }
  })
}
