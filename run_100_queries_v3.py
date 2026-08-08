"""100 Query 批量测试 v3 —— 20% 命中本地知识库(判定细则.md) + 80% 网搜兜底

与 v2 的区别：
- query 集适配「当前」知识库(判定细则.md)，20 道本地题改用判定细则内容，
  旧版那 20 道豆豆三部曲题库里已没有，会全部落到网搜导致 100%。
- 用 stdlib urllib + ThreadPoolExecutor 并发，无 httpx 依赖。
"""
import json
import sys
import time
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "http://127.0.0.1:8000"
TIMEOUT = 90          # 单题超时(秒)
MAX_CONCURRENCY = 5   # 并发数

# ─────────────────────────────────────────────
# 100 个测试 Query：A=本地命中(20) + W=网搜兜底(80)
# ─────────────────────────────────────────────
QUERIES = {
    # ====== A. 命中本地知识库「判定细则.md」（20题）======
    "A01": "机评的评分刻度是几分制？badcase 怎么定义？",
    "A02": "机评体系中哪些维度是一票否决维度？",
    "A03": "机评的总分计算公式是什么？建议的合格线是多少？",
    "A04": "相关性-纠错维度的适用条件是什么？",
    "A05": "时效性维度在什么情况下适用？",
    "A06": "本地化维度的适用条件是什么？",
    "A07": "情商维度的适用条件是什么？",
    "A08": "信息广度-信息缺失维度对哪种 query 不适用？",
    "A09": "结构化维度的适用条件是什么？",
    "A10": "机评的执行流程分为哪几步？",
    "A11": "口径校准有什么要求？单维度一致率低于多少必须收紧规则？",
    "A12": "裁判输入的评测样本包含哪些字段？",
    "A13": "价值观维度得 0 分的标准是什么？",
    "A14": "内容质量-风控维度对自伤类 query 有什么特殊要求？",
    "A15": "4 分惊喜的通用锚点是什么？",
    "A16": "权威性维度如何评估来源的权威性层级？",
    "A17": "信息深度维度如何判断答案有没有深度？",
    "A18": "相关性、需求闭环、可用性三个维度分别看什么？怎么区分？",
    "A19": "机评的 16 个维度分别是什么？列举一下。",
    "A20": "机评配套的代码文件是哪两个？分别负责什么？",

    # ====== W. 网搜兜底 —— 时事/实时（16题）======
    "W01": "今天北京的天气怎么样？",
    "W02": "比特币今天的最新价格是多少美元？",
    "W03": "今天美元兑人民币汇率是多少？",
    "W04": "2026年诺贝尔文学奖得主是谁？",
    "W05": "2026年春节是哪一天？",
    "W06": "特斯拉今天的股价是多少？",
    "W07": "中国2025年GDP增长率是多少？",
    "W08": "苹果最新款iPhone是什么时候发布的？",
    "W09": "SpaceX星舰最近一次发射是什么时候？",
    "W10": "现在日本的首相是谁？",
    "W11": "最近一次月食是什么时候？",
    "W12": "2026年诺贝尔物理学奖得主是谁？",
    "W13": "黄金价格今天走势如何？",
    "W14": "OpenAI最新发布的模型叫什么？",
    "W15": "2026年F1最近一站比赛结果如何？",
    "W16": "今年奥斯卡最佳影片是哪部？",

    # ====== W. 网搜兜底 —— 科技/AI（12题）======
    "W17": "什么是RAG检索增强生成技术？",
    "W18": "什么是大语言模型？它的工作原理是什么？",
    "W19": "Claude 4有哪些新功能？",
    "W20": "GPT-5什么时候发布？",
    "W21": "什么是多模态大模型？",
    "W22": "什么是Transformer架构？",
    "W23": "如何用LangChain构建一个RAG应用？",
    "W24": "量子计算机最新进展是什么？",
    "W25": "最新的AI芯片有哪些？",
    "W26": "如何训练一个神经网络？",
    "W27": "什么是向量数据库？常用的有哪些？",
    "W28": "什么是Agent智能体？",

    # ====== W. 网搜兜底 —— 生活/美食/健康（12题）======
    "W29": "如何制作红烧肉？详细步骤",
    "W30": "如何制作意大利面？",
    "W31": "如何制作手工面包？",
    "W32": "什么是地中海饮食？",
    "W33": "如何减掉腹部脂肪？",
    "W34": "失眠了怎么办？如何改善睡眠质量？",
    "W35": "什么是间歇性断食？",
    "W36": "如何学习游泳？",
    "W37": "什么是凯格尔运动？",
    "W38": "喝绿茶有什么健康益处？",
    "W39": "如何制作酸面包sourdough？",
    "W40": "高血压患者的饮食注意事项有哪些？",

    # ====== W. 网搜兜底 —— 英文查询（10题）======
    "W41": "Who is the president of the United States in 2026?",
    "W42": "What is the latest iPhone model?",
    "W43": "How to make sourdough bread at home?",
    "W44": "What are the latest developments in quantum computing?",
    "W45": "Who won the 2026 World Cup?",
    "W46": "What is the weather like in Tokyo today?",
    "W47": "How does a large language model work?",
    "W48": "What is the capital of Australia?",
    "W49": "Explain the concept of blockchain technology",
    "W50": "What are the health benefits of green tea?",

    # ====== W. 网搜兜底 —— 财经/商业（10题）======
    "W51": "如何开通股票账户？",
    "W52": "什么是ETF基金？",
    "W53": "巴菲特最近买了哪些股票？",
    "W54": "如何计算年化收益率？",
    "W55": "什么是REITs？",
    "W56": "如何制定个人理财计划？",
    "W57": "什么是市盈率？怎么计算？",
    "W58": "黄金和比特币哪个更适合避险？",
    "W59": "什么是比特币减半？",
    "W60": "如何看懂上市公司财报？",

    # ====== W. 网搜兜底 —— 教育/考试（8题）======
    "W61": "如何准备雅思考试？",
    "W62": "什么是GMAT考试？",
    "W63": "如何写一篇学术论文？",
    "W64": "什么是PMP认证？",
    "W65": "如何备考公务员考试？",
    "W66": "什么是MOOC课程？",
    "W67": "什么是GRE考试？",
    "W68": "如何提高英语口语？",

    # ====== W. 网搜兜底 —— 旅游/地理/文化（8题）======
    "W69": "去日本旅游需要什么签证？",
    "W70": "马尔代夫什么时候去最好？",
    "W71": "什么是申根签证？",
    "W72": "巴黎有哪些必去的景点？",
    "W73": "什么是世界遗产？",
    "W74": "冰岛旅游攻略",
    "W75": "如何申请护照？",
    "W76": "泰国曼谷有哪些美食推荐？",

    # ====== W. 网搜兜底 —— 历史/科学/其他（4题）======
    "W77": "唐朝是哪一年建立的？",
    "W78": "光速是多少？",
    "W79": "DNA双螺旋结构是谁发现的？",
    "W80": "如何制作一杯手冲咖啡？",
}


def send_query(qid: str, query: str) -> dict:
    """同步发送单题到 /api/chat，解析 SSE，返回结果。"""
    result = {
        "qid": qid, "query": query, "status": "unknown", "http_code": None,
        "used_web_search": None, "trace_id": None, "answer_chars": 0,
        "answer_preview": "", "sources_kb": 0, "sources_web": 0,
        "has_reasoning": False, "error": None, "duration_ms": 0,
    }
    payload = json.dumps({"message": query, "history": []}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/api/chat", data=payload,
        headers={"Content-Type": "application/json"},
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            result["http_code"] = resp.status
            ans, buf = [], b""
            while True:
                chunk = resp.read(2048)
                if not chunk:
                    break
                buf += chunk
                while b"\n\n" in buf:
                    frame, buf = buf.split(b"\n\n", 1)
                    ev, data = "", ""
                    for line in frame.decode("utf-8", "replace").split("\n"):
                        if line.startswith("event:"):
                            ev = line[6:].strip()
                        elif line.startswith("data:"):
                            data = line[5:].strip()
                    if ev == "sources":
                        cl = json.loads(data).get("chunks", [])
                        if json.loads(data).get("type", "kb") == "kb":
                            result["sources_kb"] += len(cl)
                        else:
                            result["sources_web"] += len(cl)
                    elif ev == "reasoning":
                        result["has_reasoning"] = True
                    elif ev == "token":
                        try:
                            ans.append(json.loads(data))
                        except Exception:
                            ans.append(data)
                    elif ev == "done":
                        d = json.loads(data)
                        result["trace_id"] = d.get("trace_id")
                        result["used_web_search"] = d.get("used_web_search", False)
                    elif ev == "error":
                        result["error"] = json.loads(data).get("message", str(data))
            answer = "".join(str(x) for x in ans)
            result["answer_chars"] = len(answer)
            result["answer_preview"] = answer[:200]
            result["status"] = "error" if result["error"] else "ok"
    except urllib.error.HTTPError as e:
        result["status"] = "http_error"
        result["http_code"] = e.code
        result["error"] = f"HTTP {e.code}: {e.read()[:200].decode('utf-8','replace')}"
    except TimeoutError:
        result["status"] = "timeout"
        result["error"] = f"Timeout after {TIMEOUT}s"
    except Exception as e:
        result["status"] = "exception"
        result["error"] = str(e)
    result["duration_ms"] = round((time.time() - start) * 1000, 1)
    return result


def main():
    print("=" * 72)
    print("《个人知识库助手》100 Query 批量测试 v3")
    print(f"分布: 20% 命中本地库(判定细则.md) + 80% 网搜兜底 | 并发={MAX_CONCURRENCY}")
    print("=" * 72, flush=True)

    # 健康检查
    try:
        with urllib.request.urlopen(f"{BASE_URL}/api/health", timeout=5) as r:
            h = json.loads(r.read())
            print(f"✅ 服务健康 | 模型: {h.get('model','?')}")
    except Exception as e:
        print(f"❌ 无法连接服务: {e}")
        return

    try:
        with urllib.request.urlopen(f"{BASE_URL}/api/documents", timeout=5) as r:
            d = json.loads(r.read())
            print(f"📚 知识库文档: {d.get('total',0)} 个 -> {d.get('documents',[])}")
    except Exception:
        pass

    total = len(QUERIES)
    print(f"\n共 {total} 题，开始执行...\n" + "-" * 72, flush=True)

    results = [None] * total
    qid_list = list(QUERIES.keys())
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as pool:
        future_map = {
            pool.submit(send_query, qid, QUERIES[qid]): idx
            for idx, qid in enumerate(qid_list)
        }
        for fut in as_completed(future_map):
            idx = future_map[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"qid": qid_list[idx], "query": QUERIES[qid_list[idx]],
                     "status": "exception", "error": str(e),
                     "used_web_search": None, "duration_ms": 0,
                     "answer_chars": 0, "sources_kb": 0, "sources_web": 0,
                     "trace_id": None, "answer_preview": "", "has_reasoning": False,
                     "http_code": None}
            results[idx] = r
            done += 1
            icon = "✅" if r["status"] == "ok" else "⏱" if r["status"] == "timeout" else "❌"
            web = "🌐" if r["used_web_search"] else ("📚" if r["used_web_search"] is False else "❓")
            q = (r["query"][:34] + "..") if len(r["query"]) > 36 else r["query"]
            print(f"[{done:3d}/{total}] {r['qid']} {icon}{web} {r['duration_ms']:.0f}ms "
                  f"KB:{r['sources_kb']} Web:{r['sources_web']} {r['answer_chars']}字  {q}", flush=True)
            if r["error"]:
                print(f"          ⚠ {r['error'][:120]}", flush=True)

    elapsed = time.time() - t0

    # ── 汇总 ──
    print("\n" + "=" * 72)
    print("📊 测试汇总报告")
    print("=" * 72)
    ok = sum(1 for r in results if r["status"] == "ok")
    err = sum(1 for r in results if r["status"] in ("error", "exception", "http_error"))
    tmo = sum(1 for r in results if r["status"] == "timeout")
    web_used = sum(1 for r in results if r["used_web_search"])
    kb_only = sum(1 for r in results if r["status"] == "ok" and r["used_web_search"] is False)
    avg_dur = sum(r["duration_ms"] for r in results) / total
    avg_chars = sum(r["answer_chars"] for r in results if r["status"] == "ok") / max(ok, 1)

    print(f"  总题数:          {total}")
    print(f"  成功 (ok):       {ok} ({ok/total*100:.0f}%)")
    print(f"  超时:            {tmo}")
    print(f"  错误:            {err}")
    print(f"  ★ 触发网搜:      {web_used} ({web_used/total*100:.0f}%)   <- 目标 80%")
    print(f"  ★ 仅本地 KB:     {kb_only} ({kb_only/total*100:.0f}%)   <- 目标 20%")
    print(f"  平均耗时:        {avg_dur:.0f}ms")
    print(f"  平均答案长度:    {avg_chars:.0f}字")
    print(f"  总耗时:          {elapsed:.0f}s")

    # 分类
    cats = {
        "A-本地命中(判定细则)": [r for r in results if r["qid"].startswith("A")],
        "W-时事实时": [r for r in results if r["qid"] in {f"W{i:02d}" for i in range(1,17)}],
        "W-科技AI": [r for r in results if r["qid"] in {f"W{i:02d}" for i in range(17,29)}],
        "W-生活健康": [r for r in results if r["qid"] in {f"W{i:02d}" for i in range(29,41)}],
        "W-英文": [r for r in results if r["qid"] in {f"W{i:02d}" for i in range(41,51)}],
        "W-财经商业": [r for r in results if r["qid"] in {f"W{i:02d}" for i in range(51,61)}],
        "W-教育考试": [r for r in results if r["qid"] in {f"W{i:02d}" for i in range(61,69)}],
        "W-旅游文化": [r for r in results if r["qid"] in {f"W{i:02d}" for i in range(69,77)}],
        "W-历史科学": [r for r in results if r["qid"] in {f"W{i:02d}" for i in range(77,81)}],
    }
    print(f"\n{'─'*72}\n分类统计:\n{'─'*72}")
    print(f"{'类别':<20} {'总数':>4} {'成功':>4} {'网搜':>4} {'本地':>4} {'错误':>4} {'均耗时':>8}")
    print(f"{'─'*72}")
    for name, rs in cats.items():
        if not rs:
            continue
        c_ok = sum(1 for r in rs if r["status"] == "ok")
        c_web = sum(1 for r in rs if r["used_web_search"])
        c_local = sum(1 for r in rs if r["used_web_search"] is False)
        c_err = sum(1 for r in rs if r["status"] not in ("ok",))
        c_avg = sum(r["duration_ms"] for r in rs) / len(rs)
        print(f"{name:<20} {len(rs):>4} {c_ok:>4} {c_web:>4} {c_local:>4} {c_err:>4} {c_avg:>7.0f}ms")

    # 详细表
    print(f"\n{'─'*72}\n详细结果:\n{'─'*72}")
    print(f"{'ID':<5}{'状态':<10}{'网搜':>5}{'KB':>4}{'Web':>5}{'耗时':>9}{'字数':>6}  Query")
    print(f"{'─'*72}")
    for r in results:
        q = (r["query"][:30] + "..") if len(r["query"]) > 32 else r["query"]
        w = "✓" if r["used_web_search"] else ("✗" if r["used_web_search"] is False else "?")
        print(f"{r['qid']:<5}{r['status']:<10}{w:>5}{r['sources_kb']:>4}{r['sources_web']:>5}"
              f"{r['duration_ms']:>8.0f}ms{r['answer_chars']:>6}  {q}")

    # 存 JSON
    out = "test_100_queries_result_v3.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {"total": total, "ok": ok, "error": err, "timeout": tmo,
                        "web_search_used": web_used, "kb_only": kb_only,
                        "web_search_pct": round(web_used / total * 100, 1),
                        "kb_only_pct": round(kb_only / total * 100, 1),
                        "avg_duration_ms": round(avg_dur, 1),
                        "avg_answer_chars": round(avg_chars, 1),
                        "elapsed_s": round(elapsed, 1)},
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n完整结果已保存: {out}")


if __name__ == "__main__":
    main()
