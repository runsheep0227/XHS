import { BrowserRouter, Routes, Route, NavLink, useLocation } from 'react-router-dom';
import TopicAnalysis from './pages/TopicAnalysis';
import CommentAnalysis from './pages/CommentAnalysis';
import CommentGeoInsight from './pages/CommentGeoInsight';
import ContentJudge from './pages/ContentJudge';

type AppScrollTheme = 'topic' | 'comments' | 'judge';

function scrollThemeForPath(pathname: string): AppScrollTheme {
  if (pathname.startsWith('/comments')) return 'comments';
  if (pathname.startsWith('/judge')) return 'judge';
  return 'topic';
}

function AppShell() {
  const { pathname } = useLocation();
  const theme = scrollThemeForPath(pathname);

  /* 只在外壳铺渐变；子页勿再叠一层，否则长页滚动易出现色带/分界突兀 */
  const shellBg =
    theme === 'comments'
      ? 'min-h-screen bg-gradient-to-b from-slate-50 via-cyan-50/25 to-indigo-50/55'
      : theme === 'judge'
        ? 'min-h-screen bg-gradient-to-b from-violet-50 via-purple-50/35 to-indigo-50/45'
        : 'min-h-screen bg-gradient-to-b from-rose-50/90 via-pink-50/30 to-rose-100/70';

  return (
    <div className={shellBg}>
      {/* 响应式顶部模块切换导航 */}
      <nav
        className="fixed top-0 left-0 right-0 z-40 bg-white/95 backdrop-blur-md border-b border-gray-200 px-2 sm:px-4 py-2 sm:py-3 shadow-sm"
        aria-label="站内导航"
      >
        <div className="max-w-7xl mx-auto flex items-center justify-center gap-1 sm:gap-2">
          <NavLink
            to="/"
            end
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
            <span className="truncate">在线交互</span>
          </NavLink>
        </div>
      </nav>

      <main className="pt-[65px]">
        <Routes>
          <Route path="/" element={<TopicAnalysis />} />
          <Route path="/comments" element={<CommentAnalysis />} />
          <Route path="/comments/geo" element={<CommentGeoInsight />} />
          <Route path="/judge" element={<ContentJudge />} />
        </Routes>
      </main>
    </div>
  );
}

function App() {
  return (
    <BrowserRouter>
      <AppShell />
    </BrowserRouter>
  );
}

export default App;
