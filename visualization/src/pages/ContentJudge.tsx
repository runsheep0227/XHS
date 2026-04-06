import { useState } from 'react';
import { MessageCircle, Send, Loader2, Copy, Check } from 'lucide-react';

type SentimentType = 'positive' | 'negative' | 'neutral';

interface JudgeResult {
  topic: string;
  topicConfidence: number;
  sentiment: SentimentType;
  sentimentScore: number;
  keywords: string[];
  /** 主题模型不可用或推理失败时的说明 */
  topicError?: string;
  /** 情感模型不可用或推理失败时的说明（评论非空时） */
  sentimentError?: string;
  /** 开发排查：来源与跳过原因 */
  debug?: string[];
}

export default function ContentJudge() {
  const [noteContent, setNoteContent] = useState('');
  const [commentContent, setCommentContent] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<JudgeResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  const handleJudge = async () => {
    if (!noteContent.trim() && !commentContent.trim()) return;

    setIsLoading(true);
    setResult(null);
    setError(null);

    const ctrl = new AbortController();
    const t = window.setTimeout(() => ctrl.abort(), 180_000);

    try {
      const res = await fetch('/api/judge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          note: noteContent.trim(),
          comment: commentContent.trim(),
        }),
        signal: ctrl.signal,
      });
      const data: unknown = await res.json().catch(() => ({}));
      if (!res.ok) {
        const msg =
          typeof data === 'object' &&
          data !== null &&
          'error' in data &&
          typeof (data as { error: unknown }).error === 'string'
            ? (data as { error: string }).error
            : `请求失败（${res.status}）`;
        setError(msg);
        return;
      }
      const o = data as Record<string, unknown>;
      if (typeof o.error === 'string') {
        setError(o.error);
        return;
      }
      const sentimentRaw = o.sentiment;
      const sentiment: SentimentType =
        sentimentRaw === 'positive' || sentimentRaw === 'negative' || sentimentRaw === 'neutral'
          ? sentimentRaw
          : 'neutral';
      const debug = Array.isArray(o.debug)
        ? o.debug.filter((x): x is string => typeof x === 'string')
        : undefined;
      const topicError = typeof o.topicError === 'string' ? o.topicError : undefined;
      const sentimentError = typeof o.sentimentError === 'string' ? o.sentimentError : undefined;
      setResult({
        topic: typeof o.topic === 'string' ? o.topic : '（无主题）',
        topicConfidence: typeof o.topicConfidence === 'number' ? o.topicConfidence : 0,
        sentiment,
        sentimentScore: typeof o.sentimentScore === 'number' ? o.sentimentScore : 0,
        keywords: Array.isArray(o.keywords)
          ? o.keywords.filter((k): k is string => typeof k === 'string')
          : [],
        ...(topicError ? { topicError } : {}),
        ...(sentimentError ? { sentimentError } : {}),
        ...(debug?.length ? { debug } : {}),
      });
    } catch (e) {
      if (e instanceof Error && e.name === 'AbortError') {
        setError('分析超时（3 分钟），请缩短文本或检查本机 Python / 模型是否过慢。');
      } else {
        setError(e instanceof Error ? e.message : '网络或服务器错误');
      }
    } finally {
      window.clearTimeout(t);
      setIsLoading(false);
    }
  };

  const handleCopy = () => {
    if (!result) return;
    const topicLine = result.topicError
      ? `主题判断: 失败 — ${result.topicError}`
      : `主题判断: ${result.topic} (${(result.topicConfidence * 100).toFixed(1)}%)`;
    const sentLine = result.sentimentError
      ? `情感判断: 失败 — ${result.sentimentError}`
      : `情感判断: ${result.sentiment === 'positive' ? '正面' : result.sentiment === 'negative' ? '负面' : '中性'} (${(result.sentimentScore * 100).toFixed(1)}%)`;
    const text = `${topicLine}\n${sentLine}`;
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
    <div className="min-h-screen pb-10 md:pb-14">
      {/* 顶部导航栏 */}
      <header className="bg-white/80 backdrop-blur-md border-b border-violet-100 px-6 py-4 sticky top-[65px] z-40">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-violet-400 to-purple-500 rounded-xl flex items-center justify-center shadow-lg">
              <span className="text-lg leading-none" aria-hidden>
                ✨
              </span>
            </div>
            <div>
              <h1 className="text-xl font-bold bg-gradient-to-r from-violet-500 to-purple-600 bg-clip-text text-transparent">
                AI 内容判断
              </h1>
              <p className="text-xs text-gray-400">
                主题与情感均从已导出模型加载（BERTopic / comment 分类模型）
              </p>
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

          {error && (
            <div
              className="rounded-xl border border-amber-200 bg-amber-50/90 px-4 py-3 text-sm text-amber-900"
              role="alert"
            >
              {error}
            </div>
          )}
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
                <div className="flex items-end gap-3 mb-3 flex-wrap">
                  <span
                    className={`text-3xl font-bold ${result.topicError ? 'text-rose-400' : 'text-rose-600'}`}
                  >
                    {result.topic}
                  </span>
                  <span className="text-sm text-gray-400 mb-1">主题</span>
                </div>
                {result.topicError ? (
                  <p className="text-sm text-amber-800 bg-amber-50 border border-amber-100 rounded-lg px-3 py-2 mb-3">
                    {result.topicError}
                  </p>
                ) : null}
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className="bg-gradient-to-r from-rose-400 to-pink-500 h-2 rounded-full transition-all duration-500"
                    style={{
                      width: `${result.topicError ? 0 : result.topicConfidence * 100}%`,
                    }}
                  />
                </div>
                <div className="flex justify-between mt-1">
                  <span className="text-xs text-gray-400">置信度</span>
                  <span className="text-xs font-medium text-rose-500">
                    {result.topicError ? '—' : `${(result.topicConfidence * 100).toFixed(1)}%`}
                  </span>
                </div>
              </div>

              {/* 情感判断 */}
              <div
                className={`bg-gradient-to-br rounded-xl p-5 border ${
                  result.sentimentError
                    ? 'from-gray-50 to-slate-50 border-gray-100'
                    : result.sentiment === 'positive'
                      ? 'from-green-50 to-emerald-50 border-green-100'
                      : result.sentiment === 'negative'
                        ? 'from-red-50 to-rose-50 border-red-100'
                        : 'from-gray-50 to-slate-50 border-gray-100'
                }`}
              >
                <div className="flex items-center gap-2 mb-3">
                  <span className="text-xl">{sentimentConfig[result.sentiment].icon}</span>
                  <span className="font-medium text-gray-700">情感判断</span>
                </div>
                <div className="flex items-end gap-3 mb-3">
                  <span
                    className={`text-3xl font-bold ${
                      result.sentimentError
                        ? 'text-gray-500'
                        : result.sentiment === 'positive'
                          ? 'text-green-600'
                          : result.sentiment === 'negative'
                            ? 'text-red-600'
                            : 'text-gray-600'
                    }`}
                  >
                    {result.sentimentError ? '—' : sentimentConfig[result.sentiment].label}
                  </span>
                  <span className="text-sm text-gray-400 mb-1">情感</span>
                </div>
                {result.sentimentError ? (
                  <p className="text-sm text-amber-800 bg-amber-50 border border-amber-100 rounded-lg px-3 py-2 mb-3">
                    {result.sentimentError}
                  </p>
                ) : null}
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div
                    className={`bg-gradient-to-r h-2 rounded-full transition-all duration-500 ${
                      result.sentiment === 'positive' ? 'from-green-400 to-emerald-500' :
                      result.sentiment === 'negative' ? 'from-red-400 to-rose-500' :
                      'from-gray-400 to-slate-500'
                    }`}
                    style={{
                      width: `${result.sentimentError ? 0 : result.sentimentScore * 100}%`,
                    }}
                  />
                </div>
                <div className="flex justify-between mt-1">
                  <span className="text-xs text-gray-400">置信度（预测类概率）</span>
                  <span
                    className={`text-xs font-medium ${
                      result.sentimentError
                        ? 'text-gray-400'
                        : result.sentiment === 'positive'
                          ? 'text-green-500'
                          : result.sentiment === 'negative'
                            ? 'text-red-500'
                            : 'text-gray-500'
                    }`}
                  >
                    {result.sentimentError ? '—' : `${(result.sentimentScore * 100).toFixed(1)}%`}
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

            {import.meta.env.DEV && result.debug && result.debug.length > 0 && (
              <p className="mt-4 text-xs text-gray-400 font-mono break-all">
                {result.debug.join(' · ')}
              </p>
            )}
          </div>
        )}

        {/* 空状态提示 */}
        {!result && !isLoading && (
          <div className="text-center py-12 text-gray-400">
            <div className="text-5xl mx-auto mb-4 opacity-60 leading-none" aria-hidden>
              ✨
            </div>
            <p>填写内容后点击&quot;开始判断&quot;获取AI分析结果</p>
          </div>
        )}
      </main>
    </div>
  );
}
