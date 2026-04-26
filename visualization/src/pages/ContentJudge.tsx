import { useMemo, useState } from 'react';
import { MessageCircle, Send, Loader2, Copy, Check, Trash2, RotateCcw } from 'lucide-react';

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

type JudgeHistoryStatus = 'loading' | 'done' | 'error';

interface JudgeHistoryItem {
  id: string;
  createdAt: number;
  input: {
    note: string;
    comment: string;
  };
  status: JudgeHistoryStatus;
  result?: JudgeResult;
  error?: string;
}

export default function ContentJudge() {
  const [noteContent, setNoteContent] = useState('');
  const [commentContent, setCommentContent] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [history, setHistory] = useState<JudgeHistoryItem[]>([]);
  const [copiedId, setCopiedId] = useState<string | null>(null);

  const latest = history[0] ?? null;

  const canSubmit = Boolean(noteContent.trim() || commentContent.trim());
  const trimmedInput = useMemo(
    () => ({ note: noteContent.trim(), comment: commentContent.trim() }),
    [noteContent, commentContent]
  );

  const handleJudge = async () => {
    if (!trimmedInput.note && !trimmedInput.comment) return;

    const id = `judge_${Date.now()}_${Math.random().toString(16).slice(2)}`;
    const createdAt = Date.now();
    const inputSnap = { ...trimmedInput };

    setHistory((prev) => [
      { id, createdAt, input: inputSnap, status: 'loading' },
      ...prev,
    ]);

    setIsLoading(true);

    const ctrl = new AbortController();
    const t = window.setTimeout(() => ctrl.abort(), 180_000);

    try {
      const res = await fetch('/api/judge', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          note: inputSnap.note,
          comment: inputSnap.comment,
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
        setHistory((prev) =>
          prev.map((it) => (it.id === id ? { ...it, status: 'error', error: msg } : it))
        );
        return;
      }
      const o = data as Record<string, unknown>;
      if (typeof o.error === 'string') {
        setHistory((prev) =>
          prev.map((it) => (it.id === id ? { ...it, status: 'error', error: o.error as string } : it))
        );
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
      const result: JudgeResult = {
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
      };
      setHistory((prev) =>
        prev.map((it) => (it.id === id ? { ...it, status: 'done', result } : it))
      );
    } catch (e) {
      if (e instanceof Error && e.name === 'AbortError') {
        const msg = '分析超时（3 分钟），请缩短文本或检查本机 Python / 模型是否过慢。';
        setHistory((prev) =>
          prev.map((it) => (it.id === id ? { ...it, status: 'error', error: msg } : it))
        );
      } else {
        const msg = e instanceof Error ? e.message : '网络或服务器错误';
        setHistory((prev) =>
          prev.map((it) => (it.id === id ? { ...it, status: 'error', error: msg } : it))
        );
      }
    } finally {
      window.clearTimeout(t);
      setIsLoading(false);
    }
  };

  // 如果最新一条还在跑，让用户知道“本次”的关联
  const latestLoadingHint =
    latest?.status === 'loading'
      ? '正在分析最新一次提交…（历史记录仍可浏览/回填）'
      : null;

  const resultToCopyText = (r: JudgeResult) => {
    const topicLine = r.topicError
      ? `主题判断: 失败 — ${r.topicError}`
      : `主题判断: ${r.topic} (${(r.topicConfidence * 100).toFixed(1)}%)`;
    const sentLine = r.sentimentError
      ? `情感判断: 失败 — ${r.sentimentError}`
      : `情感判断: ${r.sentiment === 'positive' ? '正面' : r.sentiment === 'negative' ? '负面' : '中性'} (${(r.sentimentScore * 100).toFixed(1)}%)`;
    return `${topicLine}\n${sentLine}`;
  };

  const handleCopy = (item: JudgeHistoryItem) => {
    if (!item.result) return;
    navigator.clipboard.writeText(resultToCopyText(item.result));
    setCopiedId(item.id);
    setTimeout(() => setCopiedId((cur) => (cur === item.id ? null : cur)), 2000);
  };

  const handleRehydrate = (item: JudgeHistoryItem) => {
    setNoteContent(item.input.note);
    setCommentContent(item.input.comment);
  };

  const handleDelete = (id: string) => {
    setHistory((prev) => prev.filter((it) => it.id !== id));
    setCopiedId((cur) => (cur === id ? null : cur));
  };

  const handleClear = () => {
    setHistory([]);
    setCopiedId(null);
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
                在线交互
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
            disabled={isLoading || !canSubmit}
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

          {latestLoadingHint ? (
            <div className="rounded-xl border border-violet-100 bg-violet-50/60 px-4 py-3 text-sm text-violet-900">
              {latestLoadingHint}
            </div>
          ) : null}
        </div>

        {/* 历史结果 */}
        {history.length > 0 ? (
          <div className="bg-white/80 backdrop-blur-sm rounded-2xl p-6 shadow-xl border border-violet-100">
            <div className="flex flex-wrap items-center justify-between gap-3 mb-5">
              <h2 className="text-lg font-semibold text-gray-700">判断记录</h2>
              <button
                type="button"
                onClick={handleClear}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 bg-white text-sm text-slate-600 hover:bg-slate-50 transition-colors"
              >
                <Trash2 className="w-4 h-4" />
                清空记录
              </button>
            </div>

            <div className="space-y-4">
              {history.map((item, idx) => {
                const r = item.result;
                const isNewest = idx === 0;
                const date = new Date(item.createdAt);
                const timeStr = `${date.toLocaleDateString('zh-CN')} ${date.toLocaleTimeString('zh-CN')}`;
                const noteLen = item.input.note.length;
                const commentLen = item.input.comment.length;

                return (
                  <div
                    key={item.id}
                    className={`rounded-2xl border p-4 md:p-5 ${
                      isNewest ? 'border-violet-200 bg-violet-50/30' : 'border-slate-200 bg-white'
                    }`}
                  >
                    <div className="flex flex-wrap items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="text-xs text-slate-500">
                          {isNewest ? (
                            <span className="font-semibold text-violet-600">最新</span>
                          ) : (
                            <span className="font-medium text-slate-500">历史</span>
                          )}
                          <span className="mx-2 text-slate-300">·</span>
                          <span className="tabular-nums">{timeStr}</span>
                          <span className="mx-2 text-slate-300">·</span>
                          <span className="tabular-nums">笔记 {noteLen} 字</span>
                          <span className="mx-2 text-slate-300">·</span>
                          <span className="tabular-nums">评论 {commentLen} 字</span>
                        </div>
                        <div className="mt-2 grid grid-cols-1 md:grid-cols-2 gap-3">
                          <div className="rounded-xl border border-slate-100 bg-slate-50/60 px-3 py-2">
                            <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-1">笔记</div>
                            <div className="text-sm text-slate-700 whitespace-pre-wrap break-words">
                              {item.input.note ? (item.input.note.length > 160 ? `${item.input.note.slice(0, 160)}…` : item.input.note) : '（空）'}
                            </div>
                          </div>
                          <div className="rounded-xl border border-slate-100 bg-slate-50/60 px-3 py-2">
                            <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-1">评论</div>
                            <div className="text-sm text-slate-700 whitespace-pre-wrap break-words">
                              {item.input.comment
                                ? item.input.comment.length > 160
                                  ? `${item.input.comment.slice(0, 160)}…`
                                  : item.input.comment
                                : '（空）'}
                            </div>
                          </div>
                        </div>
                      </div>

                      <div className="flex flex-wrap items-center justify-end gap-2 shrink-0">
                        <button
                          type="button"
                          onClick={() => handleRehydrate(item)}
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 bg-white text-sm text-slate-600 hover:bg-slate-50 transition-colors"
                          title="填回输入框"
                        >
                          <RotateCcw className="w-4 h-4" />
                          回填
                        </button>
                        <button
                          type="button"
                          onClick={() => handleDelete(item.id)}
                          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 bg-white text-sm text-slate-600 hover:bg-slate-50 transition-colors"
                          title="删除该条"
                        >
                          <Trash2 className="w-4 h-4" />
                          删除
                        </button>
                      </div>
                    </div>

                    <div className="mt-4">
                      {item.status === 'loading' ? (
                        <div className="flex items-center gap-2 text-sm text-slate-500">
                          <Loader2 className="w-4 h-4 animate-spin" />
                          正在分析…
                        </div>
                      ) : item.status === 'error' ? (
                        <div className="rounded-xl border border-amber-200 bg-amber-50/90 px-4 py-3 text-sm text-amber-900" role="alert">
                          {item.error || '分析失败'}
                        </div>
                      ) : r ? (
                        <>
                          <div className="flex flex-wrap items-center justify-between gap-3">
                            <div className="text-sm font-semibold text-slate-700">本次结果</div>
                            <button
                              type="button"
                              onClick={() => handleCopy(item)}
                              className="flex items-center gap-1 px-3 py-1.5 text-sm text-gray-500 hover:text-violet-600 hover:bg-violet-50 rounded-lg transition-all"
                            >
                              {copiedId === item.id ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                              {copiedId === item.id ? '已复制' : '复制结果'}
                            </button>
                          </div>

                          <div className="mt-3 grid md:grid-cols-2 gap-4">
                            <div className="bg-gradient-to-br from-rose-50 to-pink-50 rounded-xl p-4 border border-rose-100">
                              <div className="flex items-center gap-2 mb-2">
                                <span className="text-base">🏷️</span>
                                <span className="font-medium text-gray-700">主题</span>
                              </div>
                              <div className="flex items-end gap-2 mb-2 flex-wrap">
                                <span className={`text-2xl font-bold ${r.topicError ? 'text-rose-400' : 'text-rose-600'}`}>{r.topic}</span>
                                <span className="text-xs text-gray-400 mb-1">主题</span>
                              </div>
                              {r.topicError ? (
                                <p className="text-sm text-amber-800 bg-amber-50 border border-amber-100 rounded-lg px-3 py-2 mb-2">
                                  {r.topicError}
                                </p>
                              ) : null}
                              <div className="w-full bg-gray-200 rounded-full h-2">
                                <div
                                  className="bg-gradient-to-r from-rose-400 to-pink-500 h-2 rounded-full transition-all duration-500"
                                  style={{ width: `${r.topicError ? 0 : r.topicConfidence * 100}%` }}
                                />
                              </div>
                              <div className="flex justify-between mt-1">
                                <span className="text-xs text-gray-400">置信度</span>
                                <span className="text-xs font-medium text-rose-500">
                                  {r.topicError ? '—' : `${(r.topicConfidence * 100).toFixed(1)}%`}
                                </span>
                              </div>
                            </div>

                            <div
                              className={`bg-gradient-to-br rounded-xl p-4 border ${
                                r.sentimentError
                                  ? 'from-gray-50 to-slate-50 border-gray-100'
                                  : r.sentiment === 'positive'
                                    ? 'from-green-50 to-emerald-50 border-green-100'
                                    : r.sentiment === 'negative'
                                      ? 'from-red-50 to-rose-50 border-red-100'
                                      : 'from-gray-50 to-slate-50 border-gray-100'
                              }`}
                            >
                              <div className="flex items-center gap-2 mb-2">
                                <span className="text-base">{sentimentConfig[r.sentiment].icon}</span>
                                <span className="font-medium text-gray-700">情感</span>
                              </div>
                              <div className="flex items-end gap-2 mb-2">
                                <span
                                  className={`text-2xl font-bold ${
                                    r.sentimentError
                                      ? 'text-gray-500'
                                      : r.sentiment === 'positive'
                                        ? 'text-green-600'
                                        : r.sentiment === 'negative'
                                          ? 'text-red-600'
                                          : 'text-gray-600'
                                  }`}
                                >
                                  {r.sentimentError ? '—' : sentimentConfig[r.sentiment].label}
                                </span>
                                <span className="text-xs text-gray-400 mb-1">情感</span>
                              </div>
                              {r.sentimentError ? (
                                <p className="text-sm text-amber-800 bg-amber-50 border border-amber-100 rounded-lg px-3 py-2 mb-2">
                                  {r.sentimentError}
                                </p>
                              ) : null}
                              <div className="w-full bg-gray-200 rounded-full h-2">
                                <div
                                  className={`bg-gradient-to-r h-2 rounded-full transition-all duration-500 ${
                                    r.sentiment === 'positive'
                                      ? 'from-green-400 to-emerald-500'
                                      : r.sentiment === 'negative'
                                        ? 'from-red-400 to-rose-500'
                                        : 'from-gray-400 to-slate-500'
                                  }`}
                                  style={{ width: `${r.sentimentError ? 0 : r.sentimentScore * 100}%` }}
                                />
                              </div>
                              <div className="flex justify-between mt-1">
                                <span className="text-xs text-gray-400">置信度（预测类概率）</span>
                                <span
                                  className={`text-xs font-medium ${
                                    r.sentimentError
                                      ? 'text-gray-400'
                                      : r.sentiment === 'positive'
                                        ? 'text-green-500'
                                        : r.sentiment === 'negative'
                                          ? 'text-red-500'
                                          : 'text-gray-500'
                                  }`}
                                >
                                  {r.sentimentError ? '—' : `${(r.sentimentScore * 100).toFixed(1)}%`}
                                </span>
                              </div>
                            </div>
                          </div>

                          {r.keywords.length > 0 && (
                            <div className="mt-4 pt-4 border-t border-violet-100">
                              <span className="text-sm text-gray-500 mr-2">关键词:</span>
                              <div className="inline-flex flex-wrap gap-2 mt-2">
                                {r.keywords.map((keyword, kidx) => (
                                  <span key={kidx} className="px-3 py-1 bg-violet-100 text-violet-600 text-sm rounded-full">
                                    {keyword}
                                  </span>
                                ))}
                              </div>
                            </div>
                          )}

                          {import.meta.env.DEV && r.debug && r.debug.length > 0 && (
                            <p className="mt-3 text-xs text-gray-400 font-mono break-all">{r.debug.join(' · ')}</p>
                          )}
                        </>
                      ) : null}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        ) : null}

        {/* 空状态提示 */}
        {history.length === 0 && !isLoading && (
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
