(() => {
  'use strict';

  function init() {
    const one = (selector, root = document) => root.querySelector(selector);
    const all = (selector, root = document) => [...root.querySelectorAll(selector)];
    const viewLabels = {
      overview: '运行概览', chat: '对话', models: '模型库', thermal: '温度管理',
      logs: '运行日志', settings: '设置', about: '关于 Splash',
    };
    const nav = all('[data-view]');
    const panels = all('[data-panel]');
    const menu = one('[data-action="menu"]');
    const menuClose = one('[data-action="menu-close"]');
    const workspace = one('.workspace');
    const narrowScreen = window.matchMedia('(max-width: 760px)');
    const scrim = one('#nav-scrim');
    const toast = one('#toast');
    let toastTimer;

    function notify(message) {
      if (!toast) return;
      window.clearTimeout(toastTimer);
      toast.hidden = false;
      toast.textContent = message;
      toastTimer = window.setTimeout(() => {
        toast.textContent = '';
        toast.hidden = true;
      }, 2800);
    }

    function setMenu(open, restoreFocus = false) {
      document.body.classList.toggle('nav-open', open);
      menu?.setAttribute('aria-expanded', String(open));
      if (workspace) workspace.inert = open;
      if (open) menuClose?.focus();
      if (restoreFocus) menu?.focus();
    }

    const viewStorageKey = 'splash.demo.view';

    function storedView() {
      try {
        const saved = window.localStorage.getItem(viewStorageKey);
        if (saved && Object.hasOwn(viewLabels, saved)) return saved;
      } catch {
        // Storage may be blocked.
      }
      return 'models';
    }

    function activateView(view, focusHeading = false) {
      if (!Object.hasOwn(viewLabels, view)) return;
      panels.forEach(panel => { panel.hidden = panel.dataset.panel !== view; });
      nav.forEach(button => {
        if (button.dataset.view === view) button.setAttribute('aria-current', 'page');
        else button.removeAttribute('aria-current');
      });
      const heading = one('#current-view');
      if (heading) heading.textContent = viewLabels[view];
      document.title = `${viewLabels[view]} · Splash`;
      try {
        window.localStorage.setItem(viewStorageKey, view);
      } catch {
        // The remembered view is optional.
      }
      setMenu(false);
      if (focusHeading) {
        const title = one(`[data-panel="${view}"] h1`);
        if (title) {
          title.tabIndex = -1;
          title.focus({ preventScroll: true });
        }
        window.scrollTo(0, 0);
      }
      scheduleChart();
    }

    function bindClick(control, callback) {
      control?.addEventListener('click', callback);
    }

    nav.forEach(button => bindClick(button, () => activateView(button.dataset.view, true)));
    bindClick(menu, () => setMenu(true));
    bindClick(menuClose, () => setMenu(false, true));
    bindClick(scrim, () => setMenu(false, true));
    narrowScreen.addEventListener('change', () => setMenu(false));
    document.addEventListener('keydown', event => {
      if (event.key === 'Escape' && document.body.classList.contains('nav-open')) {
        event.preventDefault();
        setMenu(false, true);
      }
    });

    const chart = one('#throughput-chart');
    const context = chart?.getContext('2d');
    const ranges = all('[data-range]');
    const samples = {
      1: [51, 57, 49, 64, 61, 68, 66, 73, 67, 62, 69, 71, 62.4],
      5: [43, 49, 46, 58, 54, 62, 48, 55, 65, 59, 72, 63, 68, 65, 73, 62.4],
      15: [35, 40, 37, 48, 42, 58, 61, 52, 45, 50, 63, 56, 65, 59, 73, 68, 62.4],
    };
    const serverStatuses = all('[data-server-status]');
    const serverToggles = all('[data-action="server-toggle"]');
    let online = !serverStatuses.some(status => status.dataset.state === 'offline');
    let range = 5;
    let chartFrame = 0;
    let hoverX = null;

    function setPressed(controls, isSelected) {
      controls.forEach(control => control.setAttribute('aria-pressed', String(isSelected(control))));
    }

    function scheduleChart() {
      if (!context || chart.closest('[hidden]') || chartFrame) return;
      chartFrame = window.requestAnimationFrame(() => {
        chartFrame = 0;
        if (!chart.closest('[hidden]')) drawChart();
      });
    }

    function timeLabel(index) {
      const seconds = 10 * 3600 + 39 * 60 - range * 60 + index * range * 12;
      const minutes = Math.floor(seconds / 60);
      const clock = `${Math.floor(minutes / 60)}:${String(minutes % 60).padStart(2, '0')}`;
      return range === 1 ? `${clock}:${String(seconds % 60).padStart(2, '0')}` : clock;
    }

    function drawChart() {
      const width = chart.clientWidth;
      const height = chart.clientHeight;
      if (width < 100 || height < 75) return;
      const scale = window.devicePixelRatio || 1;
      const pixelWidth = Math.round(width * scale);
      const pixelHeight = Math.round(height * scale);
      if (chart.width !== pixelWidth) chart.width = pixelWidth;
      if (chart.height !== pixelHeight) chart.height = pixelHeight;
      context.setTransform(pixelWidth / width, 0, 0, pixelHeight / height, 0, 0);
      context.clearRect(0, 0, width, height);
      const left = 34;
      const right = width - 13;
      const top = 14;
      const bottom = height - 27;
      const accent = '#7ee0c1';
      const axis = '#86a49b';
      const values = samples[range];
      const points = values.map((value, index) => ({
        x: left + index * (right - left) / (values.length - 1),
        y: bottom - (online ? value : 0) * (bottom - top) / 80,
      }));

      context.font = '11px system-ui, sans-serif';
      context.fillStyle = axis;
      context.textBaseline = 'middle';
      context.textAlign = 'right';
      context.lineWidth = 1;
      for (let value = 0; value <= 80; value += 20) {
        const y = bottom - value * (bottom - top) / 80;
        context.strokeStyle = value === 0 ? 'rgba(134, 164, 155, .48)' : 'rgba(134, 164, 155, .20)';
        context.beginPath();
        context.moveTo(left, y);
        context.lineTo(right, y);
        context.stroke();
        context.fillText(String(value), left - 8, y);
      }
      context.strokeStyle = 'rgba(134, 164, 155, .48)';
      context.beginPath();
      context.moveTo(left, top);
      context.lineTo(left, bottom);
      context.stroke();
      const stride = width < 210 ? 5 : width < 420 ? 2 : 1;
      for (let index = 0; index <= 5; index += 1) {
        if (index % stride !== 0 && index !== 5) continue;
        if (index === 4 && stride === 2 && width < 350) continue;
        context.textAlign = index === 0 ? 'left' : index === 5 ? 'right' : 'center';
        context.fillText(timeLabel(index), left + index * (right - left) / 5, bottom + 15);
      }

      function traceSeries() {
        context.beginPath();
        context.moveTo(points[0].x, points[0].y);
        points.slice(1).forEach(point => context.lineTo(point.x, point.y));
      }

      const gradient = context.createLinearGradient(0, top, 0, bottom);
      gradient.addColorStop(0, 'rgba(126, 224, 193, .30)');
      gradient.addColorStop(1, 'rgba(126, 224, 193, .02)');
      traceSeries();
      context.lineTo(right, bottom);
      context.lineTo(left, bottom);
      context.closePath();
      context.fillStyle = gradient;
      context.fill();
      traceSeries();
      context.strokeStyle = accent;
      context.lineWidth = 2;
      context.lineJoin = 'round';
      context.stroke();

      if (hoverX === null || hoverX < left || hoverX > right) return;
      const index = Math.round((hoverX - left) * (values.length - 1) / (right - left));
      const point = points[index];
      context.strokeStyle = 'rgba(134, 164, 155, .48)';
      context.lineWidth = 1;
      context.beginPath();
      context.moveTo(point.x, top);
      context.lineTo(point.x, bottom);
      context.stroke();
      context.fillStyle = accent;
      context.beginPath();
      context.arc(point.x, point.y, 4, 0, Math.PI * 2);
      context.fill();
      const label = `${online ? values[index] : 0} tok/s`;
      const labelWidth = context.measureText(label).width + 14;
      const labelX = Math.max(left, Math.min(right - labelWidth, point.x - labelWidth / 2));
      const labelY = Math.max(1, point.y - 29);
      context.fillStyle = '#13211e';
      context.fillRect(labelX, labelY, labelWidth, 20);
      context.fillStyle = '#f3faf7';
      context.textAlign = 'center';
      context.fillText(label, labelX + labelWidth / 2, labelY + 10);
    }

    function setRange(nextRange) {
      if (!Object.hasOwn(samples, nextRange)) return;
      range = Number(nextRange);
      hoverX = null;
      setPressed(ranges, button => Number(button.dataset.range) === range);
      const description = one('#range-description');
      if (description) description.textContent = `最近 ${range} 分钟 · tok/s`;
      updateChartLabel();
      scheduleChart();
    }

    ranges.forEach(button => bindClick(button, () => setRange(button.dataset.range)));
    if (context) {
      window.addEventListener('resize', scheduleChart);
      if (typeof ResizeObserver === 'function') new ResizeObserver(scheduleChart).observe(chart);
      chart.addEventListener('mousemove', event => {
        hoverX = event.clientX - chart.getBoundingClientRect().left;
        scheduleChart();
      });
      chart.addEventListener('mouseleave', () => {
        hoverX = null;
        scheduleChart();
      });
    }

    function updateChartLabel() {
      chart?.setAttribute('aria-label', `最近 ${range} 分钟的演示解码速度，${online ? '单位 tok/s' : '服务已停止，速度为 0 tok/s'}`);
    }

    function renderServer() {
      serverStatuses.forEach(status => {
        status.dataset.state = online ? 'online' : 'offline';
        const label = one('[data-status-label]', status);
        if (label) label.textContent = online ? '服务在线' : '服务已停止';
      });
      serverToggles.forEach(button => {
        const label = one('[data-server-action-label]', button);
        if (label) label.textContent = online ? '停止服务' : '启动服务';
      });
      all('[data-if-online]').forEach(node => {
        node.textContent = online ? node.dataset.ifOnline : '—';
      });
      updateChartLabel();
      scheduleChart();
    }

    serverToggles.forEach(button => bindClick(button, () => {
      online = !online;
      renderServer();
      notify(online ? '演示服务已启动' : '演示服务已停止');
    }));

    const chatForm = one('#demo-chat-form');
    const chatInput = one('#demo-chat-input');
    const chatMessages = one('#demo-chat-messages');
    const chatIntro = one('[data-chat-intro]');

    function appendMessage(role, title, lines) {
      const message = document.createElement('article');
      const author = document.createElement('strong');
      message.className = `chat-message chat-message--${role}`;
      message.dataset.chatRole = role;
      author.textContent = title;
      message.append(author);
      lines.forEach(line => {
        const paragraph = document.createElement('p');
        paragraph.textContent = line;
        message.append(paragraph);
      });
      chatMessages.append(message);
    }

    const sessions = {
      0: [
        ['user', '27B 和 9B 在 M3 Max 上的速度差多少？'],
        ['assistant', '按这组演示数据，27B 约 62 tok/s，9B 约 98 tok/s。实际数字会随量化方式和上下文长度变化。'],
      ],
      1: [
        ['user', 'DFlash 2 会拖慢小模型吗？'],
        ['assistant', '一般不会。草稿模型只在命中概率高时才参与推测，否则自动回退到目标模型单独生成。'],
      ],
      2: [
        ['user', 'Q4 量化的 27B 需要多少内存？'],
        ['assistant', '权重约 15 GB，加上 KV 缓存和运行时开销，建议预留 20 GB 以上统一内存。'],
      ],
      3: [
        ['user', '长时间跑推理，风扇很吵怎么办？'],
        ['assistant', '可以把温控模式设为「均衡」或「静音」，引擎会按风扇和温度自适应降低预填速度。'],
      ],
    };
    const historyButtons = all('[data-session]');

    function showSession(button) {
      const messages = sessions[button.dataset.session];
      if (!messages || !chatMessages) return;
      historyButtons.forEach(item => {
        if (item === button) item.setAttribute('aria-current', 'true');
        else item.removeAttribute('aria-current');
      });
      all('[data-chat-role]', chatMessages).forEach(message => message.remove());
      if (chatIntro) chatIntro.hidden = true;
      messages.forEach(([role, text]) => appendMessage(role, role === 'user' ? '你' : 'Splash', [text]));
      chatMessages.scrollTop = chatMessages.scrollHeight;
      chatInput?.focus();
    }
    historyButtons.forEach(button => bindClick(button, () => showSession(button)));

    chatForm?.addEventListener('submit', event => {
      event.preventDefault();
      const prompt = chatInput?.value.trim();
      if (!prompt || !chatMessages) return;
      if (chatIntro) chatIntro.hidden = true;
      historyButtons.forEach(item => item.removeAttribute('aria-current'));
      appendMessage('user', '你', [prompt]);
      appendMessage('assistant', 'Splash', [
        '已收到你的消息。这里是 Splash 的本地推理演示。',
        '你可以在「模型库」切换示例模型，再到「运行概览」查看解码速度。',
      ]);
      chatInput.value = '';
      chatMessages.scrollTop = chatMessages.scrollHeight;
      chatInput.focus();
    });
    all('[data-action="new-chat"]').forEach(button => bindClick(button, () => {
      if (chatMessages) all('[data-chat-role]', chatMessages).forEach(message => message.remove());
      historyButtons.forEach(item => item.removeAttribute('aria-current'));
      if (chatIntro) chatIntro.hidden = false;
      if (!chatInput) return;
      chatInput.value = '';
      chatInput.focus();
      if (chatMessages) chatMessages.scrollTop = chatMessages.scrollHeight;
    }));

    const modelButtons = all('[data-model-index]').filter(button => /^\d+$/.test(button.dataset.modelIndex));
    const modelRows = all('[data-model-row]');
    function setModel(button, announce = false) {
      if (!button) return;
      const index = button.dataset.modelIndex;
      const selectedRow = button.closest('[data-model-row]') ??
        modelRows.find(row => row.dataset.modelRow === index) ?? modelRows[modelButtons.indexOf(button)];
      setPressed(modelButtons, item => item.dataset.modelIndex === index);
      modelRows.forEach(row => {
        const selected = row === selectedRow;
        row.classList.toggle('is-selected', selected);
        row.dataset.selected = String(selected);
        const badge = one('[data-model-badge]', row);
        if (!badge) return;
        badge.hidden = !selected;
        badge.textContent = selected ? '当前模型' : '';
      });
      const name = button.dataset.modelName;
      if (name) all('[data-active-model]').forEach(node => { node.textContent = name; });
      if (announce) notify(name ? `已切换至 ${name}` : '已切换演示模型');
    }

    modelButtons.forEach(button => bindClick(button, () => {
      setModel(button, true);
      activateView('chat', true);
      one('#demo-chat-input')?.focus();
    }));
    setModel(modelButtons.find(button => button.getAttribute('aria-pressed') === 'true') ?? modelButtons[0]);

    const thermalLabels = { auto: '智能', balanced: '均衡', quiet: '静音' };
    const thermalButtons = all('[data-thermal-mode]');
    function setThermal(mode, announce = false) {
      if (!Object.hasOwn(thermalLabels, mode)) return;
      setPressed(thermalButtons, button => button.dataset.thermalMode === mode);
      const label = one('#thermal-mode-label');
      if (label) label.textContent = thermalLabels[mode];
      if (announce) notify(`演示散热模式已设为${thermalLabels[mode]}`);
    }
    thermalButtons.forEach(button => bindClick(button, () => setThermal(button.dataset.thermalMode, true)));
    setThermal(thermalButtons.find(button => button.getAttribute('aria-pressed') === 'true')?.dataset.thermalMode ?? 'auto');

    const logList = one('#demo-log-list');
    const logFilters = all('[data-log-filter]');
    function filterLogs(level) {
      if (!['all', 'info', 'warn'].includes(level)) return;
      setPressed(logFilters, button => button.dataset.logFilter === level);
      if (logList) all('[data-log-level]', logList).forEach(entry => {
        entry.hidden = level !== 'all' && entry.dataset.logLevel !== level;
      });
    }
    logFilters.forEach(button => bindClick(button, () => filterLogs(button.dataset.logFilter)));
    filterLogs(logFilters.find(button => button.getAttribute('aria-pressed') === 'true')?.dataset.logFilter ?? 'all');
    all('[data-action="clear-logs"]').forEach(button => bindClick(button, () => {
      logList?.replaceChildren();
      notify('演示日志已清空，刷新页面可恢复');
    }));

    const settings = one('#demo-settings-form');
    const portInput = one('#demo-port');
    const endpointNodes = all('[data-endpoint]');
    const storageKey = 'splash.demo.settings';
    let port = 8000;

    function validPort(value) {
      return /^\d+$/.test(String(value)) && Number(value) >= 1 && Number(value) <= 65535;
    }

    function apiEndpoint() {
      return `http://127.0.0.1:${port}/v1`;
    }

    function renderEndpoint() {
      if (portInput) portInput.value = String(port);
      endpointNodes.forEach(node => {
        node.textContent = node.dataset.endpoint === 'api' ? apiEndpoint() : `127.0.0.1:${port}`;
      });
    }

    try {
      const saved = JSON.parse(window.localStorage.getItem(storageKey));
      if (validPort(saved?.port)) port = Number(saved.port);
    } catch {
      // Storage may be blocked or contain stale data.
    }
    renderEndpoint();

    portInput?.addEventListener('input', () => portInput.setCustomValidity(''));
    settings?.addEventListener('submit', event => {
      event.preventDefault();
      const enteredPort = portInput.value.trim();
      if (!validPort(enteredPort)) {
        portInput.setCustomValidity('请输入 1 到 65535 之间的端口号。');
        portInput.reportValidity();
        return;
      }
      port = Number(enteredPort);
      renderEndpoint();
      let summary = '设置已保存';
      try {
        window.localStorage.setItem(storageKey, JSON.stringify({ port }));
      } catch {
        summary = '设置仅在当前页面生效';
      }
      notify(`${summary} · ${apiEndpoint()}`);
    });

    function legacyCopy(text) {
      const previousFocus = document.activeElement;
      const field = document.createElement('textarea');
      field.value = text;
      field.readOnly = true;
      field.style.cssText = 'position:fixed;left:-9999px;top:0';
      document.body.append(field);
      try {
        field.select();
        return Boolean(document.execCommand?.('copy'));
      } finally {
        field.remove();
        previousFocus?.focus();
      }
    }

    all('[data-action="copy-endpoint"]').forEach(button => bindClick(button, async () => {
      const text = apiEndpoint();
      let copied = false;
      try {
        await navigator.clipboard.writeText(text);
        copied = true;
      } catch {
        try { copied = legacyCopy(text); } catch { /* Clipboard is unavailable. */ }
      }
      notify(copied ? '本地端点已复制' : `复制失败，请手动复制 ${text}`);
    }));

    setRange(ranges.find(button => button.getAttribute('aria-pressed') === 'true')?.dataset.range ?? '5');
    renderServer();
    activateView(storedView());
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();
