import React, { ReactNode } from 'react';

interface PageContainerProps {
  children: ReactNode;
  className?: string;
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl' | '2xl' | 'full';
}

const maxWidthClasses = {
  sm: 'max-w-3xl',
  md: 'max-w-4xl',
  lg: 'max-w-5xl',
  xl: 'max-w-6xl',
  '2xl': 'max-w-7xl',
  full: 'max-w-full'
};

export function PageContainer({ 
  children, 
  className = '',
  maxWidth = '2xl'
}: PageContainerProps) {
  return (
    <div className={`w-full mx-auto px-2 sm:px-4 lg:px-6 ${maxWidthClasses[maxWidth]} ${className}`}>
      {children}
    </div>
  );
}

interface PageHeaderProps {
  title: string;
  subtitle?: string;
  icon?: ReactNode;
  actions?: ReactNode;
  className?: string;
}

export function PageHeader({ 
  title, 
  subtitle, 
  icon, 
  actions,
  className = '' 
}: PageHeaderProps) {
  return (
    <div className={`flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-4 sm:mb-6 ${className}`}>
      <div className="flex items-start gap-3">
        {icon && (
          <div className="hidden sm:flex w-10 h-10 sm:w-12 sm:h-12 bg-gradient-to-br from-rose-400 to-pink-500 rounded-xl items-center justify-center shadow-lg shrink-0">
            {icon}
          </div>
        )}
        <div>
          <h1 className="text-xl sm:text-2xl font-bold text-gray-800">
            {title}
          </h1>
          {subtitle && (
            <p className="text-sm text-gray-500 mt-1 hidden sm:block">
              {subtitle}
            </p>
          )}
        </div>
      </div>
      {actions && (
        <div className="flex items-center gap-2 sm:gap-3">
          {actions}
        </div>
      )}
    </div>
  );
}

interface StatsGridProps {
  children: ReactNode;
  className?: string;
  cols?: 2 | 3 | 4 | 5;
}

export function StatsGrid({ 
  children, 
  className = '',
  cols = 5 
}: StatsGridProps) {
  const gridCols = {
    2: 'grid-cols-2',
    3: 'grid-cols-2 sm:grid-cols-3',
    4: 'grid-cols-2 sm:grid-cols-4',
    5: 'grid-cols-2 sm:grid-cols-3 lg:grid-cols-5'
  };
  
  return (
    <div className={`grid ${gridCols[cols]} gap-3 sm:gap-4 ${className}`}>
      {children}
    </div>
  );
}

interface StatCardProps {
  label: string;
  value: string | number;
  icon?: ReactNode;
  trend?: {
    value: number;
    isPositive: boolean;
  };
  color?: 'rose' | 'pink' | 'blue' | 'green' | 'gray';
  className?: string;
}

const colorClasses = {
  rose: {
    bg: 'bg-rose-50',
    text: 'text-rose-600',
    light: 'bg-rose-100'
  },
  pink: {
    bg: 'bg-pink-50',
    text: 'text-pink-600',
    light: 'bg-pink-100'
  },
  blue: {
    bg: 'bg-blue-50',
    text: 'text-blue-600',
    light: 'bg-blue-100'
  },
  green: {
    bg: 'bg-green-50',
    text: 'text-green-600',
    light: 'bg-green-100'
  },
  gray: {
    bg: 'bg-gray-50',
    text: 'text-gray-600',
    light: 'bg-gray-100'
  }
};

export function StatCard({ 
  label, 
  value, 
  icon,
  trend,
  color = 'rose',
  className = '' 
}: StatCardProps) {
  const colors = colorClasses[color];
  
  return (
    <div className={`${colors.bg} rounded-xl p-3 sm:p-4 ${className}`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs sm:text-sm text-gray-500 truncate">{label}</span>
        {icon && <div className={`${colors.text} shrink-0`}>{icon}</div>}
      </div>
      <div className="flex items-baseline gap-2">
        <span className={`text-xl sm:text-2xl font-bold ${colors.text}`}>
          {value}
        </span>
        {trend && (
          <span className={`text-xs ${trend.isPositive ? 'text-green-600' : 'text-red-600'}`}>
            {trend.isPositive ? '↑' : '↓'} {Math.abs(trend.value)}%
          </span>
        )}
      </div>
    </div>
  );
}

interface CardProps {
  children: ReactNode;
  className?: string;
  padding?: 'none' | 'sm' | 'md' | 'lg';
  hover?: boolean;
  onClick?: () => void;
}

const paddingClasses = {
  none: '',
  sm: 'p-3',
  md: 'p-4 sm:p-6',
  lg: 'p-6 sm:p-8'
};

export function Card({ 
  children, 
  className = '',
  padding = 'md',
  hover = false,
  onClick
}: CardProps) {
  const Component = onClick ? 'button' : 'div';
  
  return (
    <Component
      onClick={onClick}
      className={`
        bg-white/80 backdrop-blur-sm rounded-xl sm:rounded-2xl shadow-sm
        ${paddingClasses[padding]}
        ${hover ? 'hover:shadow-md transition-shadow cursor-pointer' : ''}
        ${className}
      `}
    >
      {children}
    </Component>
  );
}

interface TabsProps {
  tabs: {
    id: string;
    label: string;
    icon?: ReactNode;
    badge?: string | number;
  }[];
  activeTab: string;
  onChange: (id: string) => void;
  className?: string;
}

export function Tabs({ tabs, activeTab, onChange, className = '' }: TabsProps) {
  return (
    <div className={`flex flex-wrap gap-1 sm:gap-2 p-1 bg-gray-100 rounded-lg ${className}`}>
      {tabs.map(tab => (
        <button
          key={tab.id}
          onClick={() => onChange(tab.id)}
          className={`
            flex items-center gap-1.5 sm:gap-2 px-3 sm:px-4 py-2 rounded-lg text-sm font-medium
            transition-all duration-200 whitespace-nowrap
            ${activeTab === tab.id
              ? 'bg-white text-rose-600 shadow-sm'
              : 'text-gray-600 hover:text-gray-800 hover:bg-white/50'
            }
          `}
        >
          {tab.icon}
          <span className="hidden xs:inline">{tab.label}</span>
          {tab.badge !== undefined && (
            <span className={`
              px-1.5 py-0.5 rounded-full text-xs
              ${activeTab === tab.id 
                ? 'bg-rose-100 text-rose-600' 
                : 'bg-gray-200 text-gray-600'
              }
            `}>
              {tab.badge}
            </span>
          )}
        </button>
      ))}
    </div>
  );
}

interface SearchInputProps {
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  className?: string;
}

export function SearchInput({ 
  value, 
  onChange, 
  placeholder = '搜索...',
  className = '' 
}: SearchInputProps) {
  return (
    <div className={`relative ${className}`}>
      <svg 
        className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" 
        fill="none" 
        stroke="currentColor" 
        viewBox="0 0 24 24"
      >
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full pl-10 pr-4 py-2 bg-gray-50 border border-gray-200 rounded-lg sm:rounded-full text-sm sm:text-base
          focus:outline-none focus:ring-2 focus:ring-rose-300 focus:border-transparent
          transition-all"
      />
      {value && (
        <button
          onClick={() => onChange('')}
          className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      )}
    </div>
  );
}

interface SelectInputProps {
  value: string;
  onChange: (value: string) => void;
  options: { value: string; label: string }[];
  className?: string;
}

export function SelectInput({ 
  value, 
  onChange, 
  options,
  className = '' 
}: SelectInputProps) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={`
        px-3 sm:px-4 py-2 bg-gray-50 border border-gray-200 rounded-lg sm:rounded-full
        text-sm sm:text-base focus:outline-none focus:ring-2 focus:ring-rose-300
        transition-all ${className}
      `}
    >
      {options.map(opt => (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  );
}
