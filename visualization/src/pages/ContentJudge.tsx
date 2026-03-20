import { useState } from 'react';
import { Sparkles, MessageCircle, Send, Loader2, Copy, Check } from 'lucide-react';

type SentimentType = 'positive' | 'negative' | 'neutral';

interface JudgeResult {
  topic: string;
  topicConfidence: number;
  sentiment: SentimentType;
  sentimentScore: number;
  keywords: string[];
}

export default function ContentJudge() {
  const [noteContent, setNoteContent] = useState('');
  const [commentContent, setCommentContent] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<JudgeResult | null>(null);
  const [copied, setCopied] = useState(false);

  const handleJudge = async () => {
    if (!noteContent.trim() && !commentContent.trim()) return;
    
    setIsLoading(true);
    setResult(null);

    // 模拟 API 调用
    await new Promise(resolve => setTimeout(resolve, 1500));

    // Mock 结果
    const mockResult: JudgeResult = {
      topic: noteContent.includes('美妆') || noteContent.includes('护肤') ? '美妆护肤' :
             noteContent.includes('美食') ? '美食分享' :
             noteContent.includes('旅游') || noteContent.includes('旅行') ? '旅行攻略' :
             noteContent.includes('穿搭') || noteContent.includes('衣服') ? '穿搭分享' :
             noteContent.includes('健身') || noteContent.includes('运动') ? '健身运动' :
             noteContent.includes('数码') || noteContent.includes('手机') ? '数码科技' :
             noteContent.includes('母婴') || noteContent.includes('宝宝') ? '母婴育儿' : '生活日常',
      topicConfidence: 0.75 + Math.random() * 0.2,
      sentiment: commentContent.includes('好') || commentContent.includes('喜欢') || commentContent.includes('棒') || commentContent.includes('赞') ? 'positive' :
                 commentContent.includes('差') || commentContent.includes('不好') || commentContent.includes('坑') ? 'negative' : 'neutral',
      sentimentScore: 0.5 + Math.random() * 0.4,
      keywords: ['种草', '推荐', '必看'].filter(() => Math.random() > 0.5)
    };

    setResult(mockResult);
    setIsLoading(false);
  };

  const handleCopy = () => {
    if (!result) return;
    const text = `主题判断: ${result.topic} (${(result.topicConfidence * 100).toFixed(1)}%)
情感判断: ${result.sentiment === 'positive' ? '正面' : result.sentiment === 'negative' ? '负面' : '中性'} (${(result.sentimentScore * 100).toFixed(1)}%)`;
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const sentimentConfig = {
    positive: { label: '正面', color: 'from-green-400 to-emerald-500', icon: '😊' },
    negative: { label: '负面', color: 'from-red-400 to-rose-500', icon: '😔' },
    neutral: { label: '中性', color: 'from-gray-400 to-slate-500', icon: '😐' }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-violet-50 via-purple-50 to-indigo-100">
      {/* 顶部导航栏 */}
      <header className="bg-white/80 backdrop-blur-md border-b border-violet-100 px-6 py-4 sticky top-[65px] z-40">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-violet-400 to-purple-500 rounded-xl flex items-center justify-center shadow-lg">
              <Sparkles className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-violet-500 to-purple-600 bg-clip-text text-transparent">
                AI 内容判断
              </h1>
              <p className="text-xs text-gray-400">智能判断笔记主题与评论情感</p>
            </div>
          </div>
        </div>
      </header>

      <main className="p-6 max-w-4xl mx-auto">
        {/* 输入区域 */}
        <div className="grid gap-6 mb-8">
          {/* 笔记内容输入 */}
          <div className="bg-white/70 backdrop-blur-sm rounded-2xl p-6 shadow-xl border border-violet-100">
            <div className="flex items-center gap-2 mb-4">
              <span className="text-2xl">📝</span>
              <h2 className="text-lg font-semibold text-gray-700">笔记内容</h2>
            </div>
            <textarea
              value={noteContent}
              onChange={(e) => setNoteContent(e.target.value)}
              placeholder="请输入小红书笔记内容..."
              className="w-full h-32 p-4 bg-gray-50/50 border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-violet-300 focus:border-transparent resize-none transition-all"
            />
            <div className="flex justify-between items-center mt-2">
              <span className="text-xs text-gray-400">{noteContent.length} 字</span>
              <span className="text-xs text-gray-400">用于判断主题</span>
            </div>
          </div>

          {/* 评论内容输入 */}
          <div className="bg-white/70 backdrop-blur-sm rounded-2xl p-6 shadow-xl border border-violet-100">
            <div className="flex items-center gap-2 mb-4">
              <MessageCircle className="w-6 h-6 text-purple-500" />
              <h2 className="text-lg font-semibold text-gray-700">评论内容</h2>
            </div>
            <textarea
              value={commentContent}
              onChange={(e) => setCommentContent(e.target.value)}
              placeholder="请输入评论内容（可选）..."
              className="w-full h-24 p-4 bg-gray-50/50 border border-gray-200 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-purple-300 focus:border-transparent resize-none transition-all"
            />
            <div className="flex justify-between items-center mt-2">
              <span className="text-xs text-gray-400">{commentContent.length} 字</span>
              <span className="text-xs text-gray-400">用于判断情感（可选）</span>
            </div>
          </div>

          {/* 提交按钮 */}
          <button
            onClick={handleJudge}
            disabled={isLoading || (!noteContent.trim() && !commentContent.trim())}
            className="w-full py-4 bg-gradient-to-r from-violet-500 to-purple-500 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl hover:scale-[1.02] active:scale-[0.98] transition-all disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:scale-100 flex items-center justify-center gap-2"
          >
            {isLoading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                AI 正在分析中...
              </>
            ) : (
              <>
                <Send className="w-5 h-5" />
                开始判断
              </>
            )}
          </button>
        </div>

        {/* 结果展示 */}
        {result && (
          <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-xl border border-violet-100 animate-in fade-in slide-in-from-bottom-4 duration-300">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-lg font-semibold text-gray-700">判断结果</h2>
              <button
                onClick={handleCopy}
                className="flex items-center gap-1 px-3 py-1.5 text-sm text-gray-500 hover:text-violet-600 hover:bg-violet-50 rounded-lg transition-all"
              >
                {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                {copied ? '已复制' : '复制结果'}
              </button>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              {/* 主题判断 */}
              <div className="bg-gradient-to-br from-rose-50 to-pink-50 rounded-xl p-5 border border-rose-100">
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-xl">🏷️</span>
                  <span className="font-medium text-gray-700">主题判断</span>
                </div>
                <div className="flex items-end gap-3 mb-3">
                  <span className="text-3xl font-bold text-rose-600">{result.topic}</span>
                  <span className="text-sm text-gray-400 mb-1">主题</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-gradient-to-r from-rose-400 to-pink-500 h-2 rounded-full transition-all duration-500"
                    style={{ width: `${result.topicConfidence * 100}%` }}
                  />
                </div>
                <div className="flex justify-between mt-1">
                  <span className="text-xs text-gray-400">置信度</span>
                  <span className="text-xs font-medium text-rose-500">{(result.topicConfidence * 100).toFixed(1)}%</span>
                </div>
              </div>

              {/* 情感判断 */}
              <div className={`bg-gradient-to-br rounded-xl p-5 border ${
                result.sentiment === 'positive' ? 'from-green-50 to-emerald-50 border-green-100' :
                result.sentiment === 'negative' ? 'from-red-50 to-rose-50 border-red-100' :
                'from-gray-50 to-slate-50 border-gray-100'
              }`}>
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-xl">{sentimentConfig[result.sentiment].icon}</span>
                  <span className="font-medium text-gray-700">情感判断</span>
                </div>
                <div className="flex items-end gap-3 mb-3">
                  <span className={`text-3xl font-bold ${
                    result.sentiment === 'positive' ? 'text-green-600' :
                    result.sentiment === 'negative' ? 'text-red-600' : 'text-gray-600'
                  }`}>
                    {sentimentConfig[result.sentiment].label}
                  </span>
                  <span className="text-sm text-gray-400 mb-1">情感</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className={`bg-gradient-to-r h-2 rounded-full transition-all duration-500 ${
                      result.sentiment === 'positive' ? 'from-green-400 to-emerald-500' :
                      result.sentiment === 'negative' ? 'from-red-400 to-rose-500' :
                      'from-gray-400 to-slate-500'
                    }`}
                    style={{ width: `${result.sentimentScore * 100}%` }}
                  />
                </div>
                <div className="flex justify-between mt-1">
                  <span className="text-xs text-gray-400">情感得分</span>
                  <span className={`text-xs font-medium ${
                    result.sentiment === 'positive' ? 'text-green-500' :
                    result.sentiment === 'negative' ? 'text-red-500' : 'text-gray-500'
                  }`}>
                    {(result.sentimentScore * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>

            {/* 关键词标签 */}
            {result.keywords.length > 0 && (
              <div className="mt-6 pt-6 border-t border-violet-100">
                <span className="text-sm text-gray-500 mr-2">关键词:</span>
                <div className="inline-flex flex-wrap gap-2 mt-2">
                  {result.keywords.map((keyword, index) => (
                    <span 
                      key={index}
                      className="px-3 py-1 bg-violet-100 text-violet-600 text-sm rounded-full"
                    >
                      {keyword}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* 空状态提示 */}
        {!result && !isLoading && (
          <div className="text-center py-12 text-gray-400">
            <Sparkles className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>填写内容后点击"开始判断"获取AI分析结果</p>
          </div>
        )}
      </main>
    </div>
  );
}
