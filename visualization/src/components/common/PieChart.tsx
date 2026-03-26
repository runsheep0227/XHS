import type { FC } from 'react'
import type { PieSlice } from '../../utils/pie'

interface PieChartProps {
  slices: PieSlice[]
}

const PieChart: FC<PieChartProps> = ({ slices }) => {
  return (
    <svg viewBox="0 0 400 400" className="w-full h-auto max-w-[280px] mx-auto">
      {slices.map((slice) => {
        const attrs = {
          d: slice.d,
          fill: slice.fill,
          className: 'cursor-pointer transition-all hover:opacity-80',
          title: slice.name,
        }
        return (
          <path
            {...(attrs as any)}
            key={String(slice.id)}
          />
        )
      })}
    </svg>
  )
}

export default PieChart
