"""网搜模块 - 提供 豆包搜索 / Tavily / DuckDuckGo 三种供应商的统一异步接口。

通过 settings.web_search_provider 选择供应商，网搜失败不会抛出异常，
统一返回空列表以保证聊天链路不被打断。
"""
import asyncio
import re
from urllib.parse import unquote, parse_qs, urlparse

from app.config import settings


async def search_web(query: str, max_results: int) -> list[dict]:
    """网搜统一接口。

    Args:
        query: 搜索关键词。
        max_results: 最大返回条数。

    Returns:
        去重后的结果列表，每条结构为：
        {"title": str, "url": str, "snippet": str, "content": str}
        出错或超时时返回空列表。
    """
    provider = settings.web_search_provider.lower().strip()

    try:
        if provider == "doubao":
            raw = await asyncio.wait_for(
                _search_doubao(query, max_results),
                timeout=settings.web_search_timeout,
            )
        elif provider == "tavily":
            raw = await asyncio.wait_for(
                _search_tavily(query, max_results),
                timeout=settings.web_search_timeout,
            )
        else:
            # 默认走 duckduckgo
            raw = await asyncio.wait_for(
                _search_duckduckgo(query, max_results),
                timeout=settings.web_search_timeout,
            )
    except asyncio.TimeoutError:
        print(f"[web_search] 搜索超时（{settings.web_search_timeout}s），query={query!r}")
        return []
    except Exception as e:
        print(f"[web_search] 搜索失败: {e}")
        return []

    if not raw:
        return []

    # ── 后处理：去重 + 截断 ──
    seen_urls: set[str] = set()
    results: list[dict] = []
    for item in raw:
        url = item.get("url", "")
        if url in seen_urls:
            continue
        seen_urls.add(url)

        snippet = (item.get("snippet") or "")[:500]
        content = (item.get("content") or "")[:2000]
        results.append({
            "title": item.get("title", ""),
            "url": url,
            "snippet": snippet,
            "content": content,
        })

    return results


# ── 豆包搜索供应商（火山引擎 Global 版，原生异步 httpx） ──
async def _search_doubao(query: str, max_results: int) -> list[dict]:
    """调用豆包搜索 Global 版 API，返回原始结果列表。

    接口文档：https://open.feedcoopapi.com/search_api/global_search
    鉴权方式：Authorization: Bearer <api_key>
    请求体：{"Query": str, "SearchType": "web", "Count": int}
    """
    import httpx

    api_key = settings.doubao_search_api_key
    if not api_key:
        print("[web_search] 豆包搜索 API key 为空，跳过网搜。")
        return []

    url = settings.doubao_search_url
    # 豆包是国内服务，不走代理（web_search_proxy 仅用于 duckduckgo 等境外服务）

    # Global 版 Count 上限 20，客户端自行截断
    count = min(max_results, 20)

    async with httpx.AsyncClient(
        timeout=settings.web_search_timeout,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    ) as client:
        resp = await client.post(url, json={
            "Query": query,
            "SearchType": "web",
            "Count": count,
        })

    if resp.status_code != 200:
        print(f"[web_search] 豆包搜索返回 {resp.status_code}: {resp.text[:200]}")
        return []

    data = resp.json()
    result = data.get("Result") or {}
    documents = result.get("Documents") or []

    raw: list[dict] = []
    for doc in documents:
        title = doc.get("Title", "")
        doc_url = doc.get("Url", "")

        # Snippet 是数组，含 text 和 image 两种类型，拼接所有 text 段落
        snippet_parts: list[str] = []
        for s in doc.get("Snippet", []):
            if s.get("Type") == "text" and s.get("Text"):
                snippet_parts.append(s["Text"])
        full_content = "\n".join(snippet_parts)

        doc_info = doc.get("DocumentInfo") or {}
        host_info = doc.get("HostInfo") or {}
        publish_time = doc_info.get("PublishTime", "")
        hostname = host_info.get("Hostname", "")

        # 在内容头部补充来源元信息，帮助 LLM 判断时效性和权威性
        meta_prefix = ""
        if hostname:
            meta_prefix += f"来源站点: {hostname}\n"
        if publish_time:
            meta_prefix += f"发布时间: {publish_time}\n"

        raw.append({
            "title": title,
            "url": doc_url,
            "snippet": full_content[:500],
            "content": (meta_prefix + full_content) if full_content else meta_prefix,
        })

    return raw


# ── Tavily 供应商（原生异步） ──
async def _search_tavily(query: str, max_results: int) -> list[dict]:
    """调用 Tavily API 进行搜索，返回原始结果列表。"""
    from tavily import AsyncTavilyClient

    api_key = settings.tavily_api_key
    if not api_key:
        print("[web_search] Tavily API key 为空，跳过网搜。")
        return []

    client = AsyncTavilyClient(api_key=api_key)
    resp = await client.search(query, max_results=max_results)

    # Tavily 返回 {"results": [{"title", "url", "content"}, ...]}
    items = resp.get("results", []) if isinstance(resp, dict) else []
    raw: list[dict] = []
    for item in items:
        content = item.get("content", "")
        raw.append({
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "snippet": content[:200],
            "content": content,
        })
    return raw


# ── DuckDuckGo 供应商（httpx + 代理，直接抓 HTML） ──
async def _search_duckduckgo(query: str, max_results: int) -> list[dict]:
    """通过 httpx 请求 DuckDuckGo HTML 接口，解析搜索结果。

    使用 settings.web_search_proxy 配置代理（如 http://127.0.0.1:7897），
    绕过 duckduckgo-search 库的代理兼容问题。
    """
    import httpx

    proxy = settings.web_search_proxy or None

    async with httpx.AsyncClient(
        proxy=proxy,
        timeout=settings.web_search_timeout,
        follow_redirects=True,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        },
    ) as client:
        resp = await client.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query, "s": "0"},
        )

    if resp.status_code != 200:
        print(f"[web_search] DuckDuckGo 返回 {resp.status_code}")
        return []

    return _parse_ddg_html(resp.text, max_results)


def _parse_ddg_html(html: str, max_results: int) -> list[dict]:
    """解析 DuckDuckGo HTML 搜索结果页。

    DuckDuckGo HTML 页面结构：
    - 每条结果在 <a class="result__a" href="...">标题</a>
    - 链接是重定向格式：//duckduckgo.com/l/?uddg=<编码后的真实URL>&...
    - 摘要在 <a class="result__snippet">文本</a>
    """
    results: list[dict] = []

    # 提取标题+链接
    link_pattern = re.compile(
        r'<a[^>]+class="result__a"[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    for match in link_pattern.finditer(html):
        url_raw = match.group(1)
        title_html = match.group(2)

        # 解码 DuckDuckGo 重定向 URL
        url = url_raw
        if "uddg=" in url_raw:
            parsed = urlparse(url_raw)
            params = parse_qs(parsed.query)
            url = unquote(params.get("uddg", [""])[0])

        # 去除 HTML 标签
        title = re.sub(r"<[^>]+>", "", title_html).strip()

        if url and title:
            results.append({
                "title": title,
                "url": url,
                "snippet": "",
                "content": "",
            })

    # 提取摘要
    snippet_pattern = re.compile(
        r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    for i, match in enumerate(snippet_pattern.finditer(html)):
        if i < len(results):
            snippet = re.sub(r"<[^>]+>", "", match.group(1)).strip()
            results[i]["snippet"] = snippet
            results[i]["content"] = snippet

    return results[:max_results]
