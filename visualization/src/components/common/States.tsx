import React from 'react';
import { Search, Inbox, FileX, Users, MessageCircle } from 'lucide-react';

interface EmptyStateProps {
  type?: 'search' | 'data' | 'comments' | 'users' | 'custom';
  title?: string;
  description?: string;
  action?: {
    label: string;
    onClick: () => void;
  };
  className?: string;
}

const iconMap = {
  search: Search,
  data: Inbox,
  comments: MessageCircle,
  users: Users,
  custom: FileX
};

const defaultMessages = {
  search: {
    title: '未找到匹配结果',
    description: '请尝试其他搜索关键词'
  },
  data: {
    title: '暂无数据',
    description: '当前没有可展示的数据'
  },
  comments: {
    title: '暂无评论',
    description: '该笔记暂无评论内容'
  },
  users: {
    title: '暂无用户',
    description: '没有找到相关用户'
  },
  custom: {
    title: '内容为空',
    description: '暂无内容'
  }
};

export function EmptyState({ 
  type = 'custom', 
  title, 
  description, 
  action,
  className = '' 
}: EmptyStateProps) {
  const Icon = iconMap[type];
  const defaults = defaultMessages[type];
  
  return (
    <div className={`flex flex-col items-center justify-center py-12 px-4 ${className}`}>
      <div className="w-16 h-16 sm:w-20 sm:h-20 rounded-full bg-gray-100 flex items-center justify-center mb-4">
        <Icon className="w-8 h-8 sm:w-10 sm:h-10 text-gray-400" />
      </div>
      <h3 className="text-lg font-medium text-gray-700 mb-2">
        {title || defaults.title}
      </h3>
      <p className="text-sm text-gray-500 text-center max-w-md mb-6">
        {description || defaults.description}
      </p>
      {action && (
        <button
          onClick={action.onClick}
          className="px-4 py-2 bg-rose-500 text-white rounded-lg hover:bg-rose-600 transition-colors text-sm font-medium"
        >
          {action.label}
        </button>
      )}
    </div>
  );
}

interface LoadingStateProps {
  type?: 'spinner' | 'skeleton' | 'pulse';
  rows?: number;
  className?: string;
}

export function LoadingState({ 
  type = 'spinner', 
  rows = 3,
  className = '' 
}: LoadingStateProps) {
  if (type === 'skeleton') {
    return (
      <div className={`space-y-3 ${className}`}>
        {Array.from({ length: rows }).map((_, i) => (
          <div key={i} className="animate-pulse">
            <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
            <div className="h-3 bg-gray-100 rounded w-1/2"></div>
          </div>
        ))}
      </div>
    );
  }
  
  if (type === 'pulse') {
    return (
      <div className={`flex items-center justify-center py-12 ${className}`}>
        <div className="flex space-x-2">
          <div className="w-3 h-3 bg-rose-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }}></div>
          <div className="w-3 h-3 bg-rose-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }}></div>
          <div className="w-3 h-3 bg-rose-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }}></div>
        </div>
      </div>
    );
  }
  
  // 默认 spinner
  return (
    <div className={`flex items-center justify-center py-12 ${className}`}>
      <div className="w-8 h-8 border-4 border-rose-200 border-t-rose-500 rounded-full animate-spin"></div>
    </div>
  );
}

interface SkeletonCardProps {
  className?: string;
}

export function SkeletonCard({ className = '' }: SkeletonCardProps) {
  return (
    <div className={`bg-white rounded-xl p-4 animate-pulse ${className}`}>
      <div className="flex items-center gap-3 mb-4">
        <div className="w-10 h-10 bg-gray-200 rounded-lg"></div>
        <div className="flex-1">
          <div className="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
          <div className="h-3 bg-gray-100 rounded w-1/2"></div>
        </div>
      </div>
      <div className="space-y-2">
        <div className="h-3 bg-gray-100 rounded"></div>
        <div className="h-3 bg-gray-100 rounded w-5/6"></div>
      </div>
    </div>
  );
}
