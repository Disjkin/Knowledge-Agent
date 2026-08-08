/* 前端交互 + SSE 流式渲染 */
(() => {
  const $messages = document.getElementById("messages");
  const $form = document.getElementById("chat-form");
  const $input = document.getElementById("input");
  const $send = document.getElementById("btn-send");
  const $docList = document.getElementById("doc-list");
  const $ingest = document.getElementById("btn-ingest");
  const $badge = document.getElementById("model-badge");

  // 设置面板元素
  const $modal = document.getElementById("settings-modal");
  const $btnSettings = document.getElementById("btn-settings");
  const $btnCloseSettings = document.getElementById("btn-close-settings");
  const $setProvider = document.getElementById("set-provider");
  const $setApikey = document.getElementById("set-apikey");
  const $setBaseurl = document.getElementById("set-baseurl");
  const $setModel = document.getElementById("set-model");
  const $btnSave = document.getElementById("btn-save-settings");
  const $btnTest = document.getElementById("btn-test");
  const $settingsMsg = document.getElementById("settings-msg");

  // 移动端侧栏抽屉
  const $sidebar = document.getElementById("sidebar");
  const $backdrop = document.getElementById("sidebar-backdrop");
  const $btnMenu = document.getElementById("btn-menu");

  let busy = false;
  let autoScroll = true; // 用户是否在底部附近（控制自动滚动）

  // ── 输入区工具开关 ──
  const $btnDeepThink = document.getElementById("btn-deep-think");
  const $btnWebSearch = document.getElementById("btn-web-search");
  const $btnAttach = document.getElementById("btn-attach");
  let deepThink = false;  // 深度思考：切推理模型（需配置 OPENAI_REASONING_MODEL）
  let webSearchOn = true; // 智能搜索：本地不足时自动联网（默认开）

  function bindToggle(btn, onChange) {
    if (!btn) return;
    btn.addEventListener("click", () => {
      const on = !btn.classList.contains("active");
      btn.classList.toggle("active", on);
      btn.setAttribute("aria-pressed", on ? "true" : "false");
      onChange(on);
    });
  }
  bindToggle($btnDeepThink, (on) => (deepThink = on));
  bindToggle($btnWebSearch, (on) => (webSearchOn = on));

  // 回形针：打开 data/ 目录添加文档
  if ($btnAttach) {
    $btnAttach.addEventListener("click", async () => {
      try {
        const r = await fetch("/api/open-data-folder", { method: "POST" });
        const j = await r.json();
        if (j.status !== "ok") {
          prompt("请手动打开此路径放入文档，然后点 🔄 重建：", j.path);
        }
      } catch (e) {
        prompt("请手动打开 data/ 文件夹放入文档，然后点击 🔄 重建。", "data/");
      }
    });
  }

  // 滚动到底部浮动按钮
  const $scrollBtn = document.createElement("button");
  $scrollBtn.className = "scroll-to-bottom hidden";
  $scrollBtn.innerHTML = "↓";
  $scrollBtn.title = "滚动到最新消息";
  $messages.parentElement.style.position = "relative";
  $messages.parentElement.appendChild($scrollBtn);
  $scrollBtn.addEventListener("click", () => forceScrollBottom());

  // 滚动监听：用户主动上滑时暂停自动滚动，滑回底部附近时恢复
  $messages.addEventListener("scroll", () => {
    const threshold = 120; // 距底部多少像素以内算"在底部"
    autoScroll =
      $messages.scrollHeight - $messages.scrollTop - $messages.clientHeight <
      threshold;
    $scrollBtn.classList.toggle("hidden", autoScroll);
  });

  // ── 工具 ──
  function escapeHtml(s) {
    return s.replace(/[&<>"']/g, (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
    );
  }

  // 轻量 Markdown：代码块、行内代码、粗体、标题、列表、链接、换行
  // 占位用 @@B{n}@@ / @@I{n}@@ 哨兵（纯 ASCII、不被 trim/escapeHtml 影响）
  function renderMd(text) {
    if (!text) return "";
    let h = escapeHtml(text);

    // 1. 围栏代码块 ```lang \n code \n ``` -> 占位，保护内部内容
    const blocks = [];
    h = h.replace(/```([^\n`]*)\n?([\s\S]*?)```/g, (_, lang, code) => {
      const idx = blocks.length;
      blocks.push("<pre><code>" + code.replace(/\n$/, "") + "</code></pre>");
      return "@@B" + idx + "@@";
    });

    // 2. 行内代码 -> 占位
    const inlines = [];
    h = h.replace(/`([^`\n]+)`/g, (_, code) => {
      const idx = inlines.length;
      inlines.push("<code>" + code + "</code>");
      return "@@I" + idx + "@@";
    });

    // 3. 行级块解析：标题、列表、段落
    const lines = h.split("\n");
    const out = [];
    let listType = null;
    let para = [];

    const closeList = () => {
      if (listType) { out.push("</" + listType + ">"); listType = null; }
    };
    const flushPara = () => {
      if (para.length) { out.push("<p>" + para.join("<br>") + "</p>"); para = []; }
    };

    for (const line of lines) {
      const t = line.trim();

      // 代码块占位行 -> 原样输出
      if (/^@@B\d+@@$/.test(t)) { flushPara(); closeList(); out.push(t); continue; }

      let m;
      if ((m = t.match(/^(#{1,3})\s+(.*)$/))) {
        flushPara(); closeList();
        out.push("<h" + m[1].length + ">" + m[2] + "</h" + m[1].length + ">"); continue;
      }
      if ((m = t.match(/^[-*]\s+(.*)$/))) {
        flushPara();
        if (listType !== "ul") { closeList(); out.push("<ul>"); listType = "ul"; }
        out.push("<li>" + m[1] + "</li>"); continue;
      }
      if ((m = t.match(/^\d+\.\s+(.*)$/))) {
        flushPara();
        if (listType !== "ol") { closeList(); out.push("<ol>"); listType = "ol"; }
        out.push("<li>" + m[1] + "</li>"); continue;
      }
      if (t === "") { flushPara(); closeList(); continue; }

      para.push(t);
    }
    flushPara();
    closeList();

    let html = out.join("");

    // 4. 行内：粗体、链接（占位保护了代码内容）
    html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

    // 5. 还原行内代码、代码块
    html = html.replace(/@@I(\d+)@@/g, (_, i) => inlines[+i]);
    html = html.replace(/@@B(\d+)@@/g, (_, i) => blocks[+i]);

    return html;
  }

  function scrollBottom() {
    if (!autoScroll) return; // 用户上滑后不强制拉回
    $messages.scrollTop = $messages.scrollHeight;
  }

  function forceScrollBottom() {
    autoScroll = true;
    $messages.scrollTop = $messages.scrollHeight;
  }

  function setBusy(v) {
    busy = v;
    $send.disabled = v;
    $input.disabled = v;
    // 发送按钮为图标，靠 disabled 样式表示生成中
  }

  // ── 消息气泡 ──
  function addBubble(role, html) {
    const wrap = document.createElement("div");
    wrap.className = "msg " + role;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = html;
    wrap.appendChild(bubble);
    $messages.appendChild(wrap);
    scrollBottom();
    return bubble;
  }

  function addUser(text) {
    addBubble("user", renderMd(text));
    forceScrollBottom(); // 用户发消息时强制滚到底
    closeSidebar(); // 移动端发完消息收起侧栏
  }

  // 流式助手消息：含「推理过程」（折叠）+ 答案气泡
  function createStreamingAssistant() {
    const wrap = document.createElement("div");
    wrap.className = "msg assistant";

    // 思考过程区（默认折叠，有内容时显示）
    const reasoning = document.createElement("div");
    reasoning.className = "reasoning";
    const rHead = document.createElement("div");
    rHead.className = "reasoning-head";
    rHead.textContent = "💭 思考过程";
    const rChevron = document.createElement("span");
    rChevron.className = "chevron";
    rChevron.textContent = "▶";
    rHead.appendChild(rChevron);
    const rBody = document.createElement("div");
    rBody.className = "reasoning-body";
    rBody.style.display = "none";
    reasoning.appendChild(rHead);
    reasoning.appendChild(rBody);
    reasoning.style.display = "none"; // 无推理则整体隐藏
    rHead.addEventListener("click", () => {
      const open = rBody.style.display !== "none";
      rBody.style.display = open ? "none" : "block";
      rChevron.textContent = open ? "▶" : "▼";
    });
    wrap.appendChild(reasoning);

    // 答案气泡：answer-text（累积全文渲染）+ 光标
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    const answerText = document.createElement("div");
    answerText.className = "answer-text";
    const cursor = document.createElement("span");
    cursor.className = "cursor";
    bubble.appendChild(answerText);
    bubble.appendChild(cursor);
    wrap.appendChild(bubble);

    $messages.appendChild(wrap);
    forceScrollBottom(); // 新建助手消息时强制滚到底
    return { wrap, reasoning, rBody, rHead, bubble, answerText, cursor, statusEl: null, answer: "", _raf: null };
  }

  // 渲染调度：用 rAF 合并同一帧内的多次 token，避免频繁 innerHTML 重排
  function scheduleRender(state) {
    if (state._raf) return;
    state._raf = requestAnimationFrame(() => {
      state._raf = null;
      state.answerText.innerHTML = renderMd(state.answer);
      scrollBottom();
    });
  }

  // 追加答案 token（累积全文，统一渲染）
  function appendToken(state, delta) {
    if (delta == null) return;
    state.answer += delta;
    scheduleRender(state);
  }

  // 追加推理 token
  function appendReason(state, delta) {
    if (!delta) return;
    state.reasoning.style.display = ""; // 有推理 -> 显示
    const span = document.createElement("span");
    span.textContent = delta; // 推理不做 markdown，避免乱码
    state.rBody.appendChild(span);
    scrollBottom();
  }

  // ── 引用卡片（折叠，挂在答案消息内） ──
  function renderSources(state, chunks, type) {
    if (!chunks || !chunks.length) return;
    type = type || "kb"; // 默认本地知识库
    const card = document.createElement("div");
    card.className = "sources-card";

    const title = document.createElement("div");
    title.className = "sources-title";
    const titleText = type === "web"
      ? "🌐 网络来源 (" + chunks.length + ")"
      : "📎 参考来源 (" + chunks.length + ")";
    title.textContent = titleText;
    const chevron = document.createElement("span");
    chevron.className = "chevron";
    chevron.textContent = "▶"; // 默认折叠
    title.appendChild(chevron);
    card.appendChild(title);

    const items = document.createElement("div");
    items.className = "sources-items";
    items.style.display = "none"; // 默认折叠
    chunks.forEach((c, i) => {
      const item = document.createElement("div");
      item.className = type === "web" ? "source-item source-web" : "source-item";
      if (type === "web") {
        // 网络来源：可点击链接 + 域名 + snippet
        const href = escapeHtml(c.url || "#");
        const name = escapeHtml(c.file_name || c.source || c.title || "未知来源");
        const snippet = escapeHtml(c.snippet || "");
        item.innerHTML =
          '<div class="source-head"><a href="' + href + '" target="_blank" rel="noopener">#' + (i + 1) + " · " + name + "</a></div>" +
          '<div class="source-snippet">' + snippet + "</div>";
      } else {
        // 本地知识库来源
        item.innerHTML =
          '<div class="source-head">#' + (c.index ?? (i + 1)) + " · " + escapeHtml(c.file_name || c.source) + "</div>" +
          '<div class="source-snippet">' + escapeHtml(c.snippet || "") + "</div>";
      }
      items.appendChild(item);
    });
    card.appendChild(items);

    // 折叠：点击标题切换
    title.addEventListener("click", () => {
      const open = items.style.display !== "none";
      items.style.display = open ? "none" : "block";
      chevron.textContent = open ? "▶" : "▼";
    });

    state.wrap.appendChild(card);
    scrollBottom();
  }

  // ── 状态行 ──
  // 在答案气泡上方插入/更新灰色小字状态
  function showStatus(state, message) {
    if (state.statusEl) {
      state.statusEl.textContent = message;
      return;
    }
    const el = document.createElement("div");
    el.className = "status-line";
    el.textContent = message;
    // 插入到气泡前面（气泡是 wrap 的最后一个子元素）
    state.wrap.insertBefore(el, state.bubble);
    state.statusEl = el;
    scrollBottom();
  }

  // 移除状态行
  function removeStatus(state) {
    if (state.statusEl) {
      state.statusEl.remove();
      state.statusEl = null;
    }
  }

  // 在答案消息底部追加 trace_id 脚注
  function showTraceId(state, traceId) {
    const el = document.createElement("div");
    el.className = "trace-id";
    el.textContent = "trace: " + traceId.slice(0, 8);
    state.wrap.appendChild(el);
    scrollBottom();
  }

  // ── SSE 发送 ──
  async function sendQuestion(text) {
    if (busy || !text.trim()) return;
    setBusy(true);
    addUser(text);

    const state = createStreamingAssistant();

    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          // 智能搜索：开=null（后端自动判定），关=false（仅本地知识库）
          web_search: webSearchOn ? null : false,
          deep_think: deepThink,
        }),
      });

      if (!resp.ok) {
        appendToken(state, "[请求失败: HTTP " + resp.status + "]");
        setBusy(false);
        return;
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buf = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buf += decoder.decode(value, { stream: true });

        // 按 SSE 双换行切帧
        let idx;
        while ((idx = buf.indexOf("\n\n")) !== -1) {
          const frame = buf.slice(0, idx);
          buf = buf.slice(idx + 2);
          handleFrame(frame, state);
        }
      }
    } catch (e) {
      appendToken(state, "[连接失败: " + e.message + "]");
    } finally {
      // 确保最后一帧渲染完毕后移除光标
      if (state._raf) { cancelAnimationFrame(state._raf); state._raf = null; }
      state.answerText.innerHTML = renderMd(state.answer);
      state.cursor.remove();
      setBusy(false);
    }
  }

  function handleFrame(frame, state) {
    let event = "message";
    const lines = frame.split("\n");
    for (const ln of lines) {
      if (ln.startsWith("event:")) event = ln.slice(6).trim();
      else if (ln.startsWith("data:")) {
        const raw = ln.slice(5).trim();
        let data;
        try {
          data = JSON.parse(raw);
        } catch {
          data = raw;
        }
        if (event === "token") {
          appendToken(state, data.delta ?? data);
        } else if (event === "reasoning") {
          appendReason(state, data.delta ?? data);
        } else if (event === "sources") {
          renderSources(state, data.chunks, data.type);
          removeStatus(state); // sources 到达后移除状态行
        } else if (event === "status") {
          showStatus(state, data.message ?? data);
        } else if (event === "error") {
          appendToken(state, "\n[错误: " + (data.message ?? data) + "]");
        } else if (event === "done") {
          if (data.trace_id) {
            showTraceId(state, data.trace_id);
          }
          removeStatus(state); // done 到达后移除状态行
        }
      }
    }
  }

  // ── 表单 ──
  $form.addEventListener("submit", (e) => {
    e.preventDefault();
    sendQuestion($input.value);
    $input.value = "";
    $input.style.height = "auto";
  });

  $input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      $form.requestSubmit();
    }
  });

  // 自动调整高度
  $input.addEventListener("input", () => {
    $input.style.height = "auto";
    $input.style.height = Math.min($input.scrollHeight, 200) + "px";
  });

  // ── 文档列表 / 重建 ──
  async function loadDocs() {
    try {
      const r = await fetch("/api/documents");
      const j = await r.json();
      $docList.innerHTML = "";
      if (!j.documents || j.documents.length === 0) {
        $docList.innerHTML = '<li class="doc-empty">暂无文档<br/>把文件放入 data/ 后点"重建"</li>';
        return;
      }
      j.documents.forEach((d) => {
        const li = document.createElement("li");
        li.className = "doc-item";
        li.textContent = d;
        $docList.appendChild(li);
      });
    } catch {
      $docList.innerHTML = '<li class="doc-empty">无法连接服务</li>';
    }
  }

  // "添加文档" -- 打开 data/ 文件夹，方便用户拖入文件
  document.getElementById("btn-add-doc").addEventListener("click", async () => {
    try {
      const r = await fetch("/api/open-data-folder", { method: "POST" });
      const j = await r.json();
      if (j.status !== "ok") {
        prompt("请手动打开此路径放入文档，然后点 🔄 重建：", j.path);
      }
    } catch (e) {
      prompt("请手动打开 data/ 文件夹放入文档，然后点击 🔄 重建。", "data/");
    }
  });

  $ingest.addEventListener("click", async () => {
    $ingest.disabled = true;
    $ingest.textContent = "索引中…";
    try {
      const r = await fetch("/api/ingest", { method: "POST" });
      const j = await r.json();
      alert("索引完成: 新增 " + j.files_indexed + " 个文件, " + j.chunks_generated + " 个片段");
      loadDocs();
    } catch (e) {
      alert("索引失败: " + e.message);
    } finally {
      $ingest.disabled = false;
      $ingest.textContent = "🔄 重建";
    }
  });

  // "清空重建" -- 先清空全部向量再从头索引（删除文件后清理孤儿片段用）
  const $clearIngest = document.getElementById("btn-clear-ingest");
  if ($clearIngest) {
    $clearIngest.addEventListener("click", async () => {
      if (!confirm("将清空知识库全部向量后从头重新索引 data/ 下的文件。\n（用于彻底清理已删除文件的残留片段）\n\n确定继续吗？")) return;
      $clearIngest.disabled = true;
      $ingest.disabled = true;
      const prev = $clearIngest.textContent;
      $clearIngest.textContent = "清空重建中…";
      try {
        const r = await fetch("/api/ingest?clear=true", { method: "POST" });
        const j = await r.json();
        alert("清空重建完成: 入库 " + j.files_indexed + " 个文件, " + j.chunks_generated + " 个片段");
        loadDocs();
      } catch (e) {
        alert("清空重建失败: " + e.message);
      } finally {
        $clearIngest.disabled = false;
        $ingest.disabled = false;
        $clearIngest.textContent = prev;
      }
    });
  }

  // 启动：拉文档 + 模型信息
  loadDocs();
  fetch("/api/health")
    .then((r) => r.json())
    .then((j) => {
      const m = j.model || "unknown";
      $badge.textContent = "模型: " + m;
      $badge.title = m; // 完整信息放 title，文本靠 CSS 截断
    })
    .catch(() => {});

  // ── 移动端侧栏抽屉 ──
  function openSidebar() { $sidebar.classList.add("open"); $backdrop.classList.add("show"); }
  function closeSidebar() { $sidebar.classList.remove("open"); $backdrop.classList.remove("show"); }
  if ($btnMenu) {
    $btnMenu.addEventListener("click", () => {
      $sidebar.classList.contains("open") ? closeSidebar() : openSidebar();
    });
  }
  if ($backdrop) $backdrop.addEventListener("click", closeSidebar);

  // ── 设置面板 ──
  function openModal() {
    $modal.classList.remove("hidden");
    loadSettings();
  }
  function closeModal() {
    $modal.classList.add("hidden");
    $settingsMsg.textContent = "";
  }

  async function loadSettings() {
    try {
      const r = await fetch("/api/settings");
      const j = await r.json();
      $setProvider.value = j.provider || "openai";
      $setApikey.value = j.api_key_masked || "";
      $setBaseurl.value = j.base_url || "";
      $setModel.value = j.model || "";
      onProviderChange();
    } catch {
      /* 用默认值 */
    }
  }

  function onProviderChange() {
    const isAnthropic = $setProvider.value === "anthropic";
    $setBaseurl.disabled = isAnthropic;
    document.getElementById("set-baseurl-label").style.opacity = isAnthropic ? 0.5 : 1;
  }

  function showMsg(text, ok) {
    $settingsMsg.textContent = text;
    $settingsMsg.style.color = ok ? "var(--accent)" : "#f87171";
  }

  $btnSettings.addEventListener("click", openModal);
  $btnCloseSettings.addEventListener("click", closeModal);
  $modal.querySelector(".modal-mask").addEventListener("click", closeModal);
  $setProvider.addEventListener("change", onProviderChange);

  $btnTest.addEventListener("click", async () => {
    $btnTest.disabled = true;
    $btnTest.textContent = "测试中…";
    showMsg("正在测试连接…", true);
    try {
      const r = await fetch("/api/settings/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: $setProvider.value,
          api_key: $setApikey.value,
          base_url: $setBaseurl.value,
          model: $setModel.value,
          _is_masked: $setApikey.value.includes("•"), // 掩码 key 说明没改
        }),
      });
      const j = await r.json();
      if (j.ok) {
        showMsg('✓ 连接成功！模型回复："' + (j.reply || "").slice(0, 80) + '"', true);
      } else {
        showMsg("✗ 失败：" + (j.error || "未知错误"), false);
      }
    } catch (e) {
      showMsg("✗ 请求失败：" + e.message, false);
    } finally {
      $btnTest.disabled = false;
      $btnTest.textContent = "测试连接";
    }
  });

  $btnSave.addEventListener("click", async () => {
    $btnSave.disabled = true;
    $btnSave.textContent = "保存中…";
    showMsg("保存中…", true);
    try {
      const r = await fetch("/api/settings", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: $setProvider.value,
          // 掩码 key 且没改，则不传 api_key（保留原值）
          api_key: $setApikey.value.includes("•") ? undefined : $setApikey.value,
          base_url: $setBaseurl.value,
          model: $setModel.value,
        }),
      });
      const j = await r.json();
      if (j.ok) {
        showMsg("✓ 保存成功，即刻生效", true);
        const m = $setModel.value || "unknown";
        $badge.textContent = "模型: " + m;
        $badge.title = m;
      } else {
        showMsg("✗ " + (j.error || "保存失败"), false);
      }
    } catch (e) {
      showMsg("✗ 保存失败：" + e.message, false);
    } finally {
      $btnSave.disabled = false;
      $btnSave.textContent = "保存";
    }
  });
})();
