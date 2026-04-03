// 响应式布局工具函数
export const breakpoints = {
  sm: '640px',
  md: '768px',
  lg: '1024px',
  xl: '1280px',
  '2xl': '1536px'
};

// 响应式类名映射
export const responsiveClasses = {
  // 布局
  container: 'w-full max-w-7xl mx-auto px-2 sm:px-4 lg:px-6',
  
  // 侧边栏
  sidebar: {
    left: 'w-full lg:w-64 xl:w-72',
    right: 'w-full lg:w-80 xl:w-96'
  },
  
  // 主内容区
  main: 'flex-1 min-w-0',
  
  // 卡片
  card: 'bg-white/80 backdrop-blur-sm rounded-xl sm:rounded-2xl',
  
  // 按钮
  button: {
    base: 'px-3 sm:px-4 py-2 rounded-lg sm:rounded-xl text-sm font-medium transition-all',
    group: 'flex flex-wrap gap-2'
  },
  
  // 输入框
  input: 'w-full px-3 sm:px-4 py-2 text-sm sm:text-base',
  
  // 标题
  title: {
    h1: 'text-xl sm:text-2xl font-bold',
    h2: 'text-lg sm:text-xl font-semibold',
    h3: 'text-base sm:text-lg font-medium'
  },
  
  // 统计数字
  stat: 'text-2xl sm:text-3xl font-bold'
};

// 判断是否为移动设备的简单方法
export const isMobile = () => typeof window !== 'undefined' && window.innerWidth < 1024;

// 格式化数字
export const formatNumber = (num: number, options?: { compact?: boolean; decimals?: number }): string => {
  const { compact = true, decimals = 1 } = options || {};
  
  if (compact) {
    if (num >= 10000) {
      return (num / 10000).toFixed(decimals) + 'w';
    }
    if (num >= 1000) {
      return (num / 1000).toFixed(decimals) + 'k';
    }
  }
  
  return num.toLocaleString();
};

// 生成唯一ID
export const generateId = (prefix: string = 'id'): string => {
  return `${prefix}-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};

// 防抖函数
export const debounce = <T extends (...args: unknown[]) => unknown>(
  func: T,
  wait: number
): ((...args: Parameters<T>) => void) => {
  let timeout: ReturnType<typeof setTimeout>;
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
};

// 节流函数
export const throttle = <T extends (...args: unknown[]) => unknown>(
  func: T,
  limit: number
): ((...args: Parameters<T>) => void) => {
  let inThrottle: boolean;
  return (...args: Parameters<T>) => {
    if (!inThrottle) {
      func(...args);
      inThrottle = true;
      setTimeout(() => (inThrottle = false), limit);
    }
  };
};
