import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom';
import TopicAnalysis from './pages/TopicAnalysis';
import CommentAnalysis from './pages/CommentAnalysis';
import ContentJudge from './pages/ContentJudge';

function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-50">
        {/* 响应式顶部模块切换导航 */}
        <nav className="bg-white/95 backdrop-blur-md border-b border-gray-200 px-2 sm:px-4 py-2 sm:py-3 sticky top-0 z-50">
          <div className="max-w-7xl mx-auto flex items-center justify-center gap-1 sm:gap-2">
            <NavLink
              to="/"
              className={({ isActive }) =>
                `flex items-center gap-1.5 sm:gap-2 px-3 sm:px-5 py-2 rounded-full font-medium text-sm sm:text-base transition-all ${
                  isActive
                    ? 'bg-gradient-to-r from-rose-500 to-pink-500 text-white shadow-lg'
                    : 'text-gray-600 hover:bg-gray-100'
                }`
              }
            >
              <span className="hidden xs:inline">📝</span>
              <span className="truncate">笔记主题</span>
            </NavLink>
            <NavLink
              to="/comments"
              className={({ isActive }) =>
                `flex items-center gap-1.5 sm:gap-2 px-3 sm:px-5 py-2 rounded-full font-medium text-sm sm:text-base transition-all ${
                  isActive
                    ? 'bg-gradient-to-r from-blue-500 to-cyan-500 text-white shadow-lg'
                    : 'text-gray-600 hover:bg-gray-100'
                }`
              }
            >
              <span className="hidden xs:inline">💬</span>
              <span className="truncate">评论分析</span>
            </NavLink>
            <NavLink
              to="/judge"
              className={({ isActive }) =>
                `flex items-center gap-1.5 sm:gap-2 px-3 sm:px-5 py-2 rounded-full font-medium text-sm sm:text-base transition-all ${
                  isActive
                    ? 'bg-gradient-to-r from-violet-500 to-purple-500 text-white shadow-lg'
                    : 'text-gray-600 hover:bg-gray-100'
                }`
              }
            >
              <span className="hidden xs:inline">✨</span>
              <span className="truncate">AI判断</span>
            </NavLink>
          </div>
        </nav>

        {/* 页面内容 */}
        <Routes>
          <Route path="/" element={<TopicAnalysis />} />
          <Route path="/comments" element={<CommentAnalysis />} />
          <Route path="/judge" element={<ContentJudge />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
