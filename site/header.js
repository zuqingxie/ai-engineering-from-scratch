/**
 * Shared header behaviors: language selector and live GitHub star counter.
 * Loaded by every page that includes the shared site header.
 */
(function () {
  var REPO = 'rohitg00/ai-engineering-from-scratch';
  var CACHE_KEY = 'gh:stars:' + REPO;
  var CACHE_TTL_MS = 10 * 60 * 1000; // 10 minutes
  var LANG_KEY = 'site:lang';
  var DEFAULT_LANG = 'en';

  var I18N = {
    en: {
      'nav.contents': 'Contents',
      'nav.catalog': 'Catalog',
      'nav.roadmap': 'Roadmap',
      'nav.glossary': 'Glossary',
      'nav.home': 'Home',
      'nav.report': 'Report',
      'nav.reportSuggest': 'Report / Suggest',
      'skip.content': 'Skip to content',
      'search.label': 'Search (⌘K)',
      'theme.toggle': 'Toggle theme',
      'footer.open': 'AI Engineering from Scratch · open source · free forever.',
      'footer.short': '© 2026 · open source · free forever',
      'home.metaRight': 'open source · MIT',
      'home.tagline': '503 lessons. 20 phases. Every algorithm built from raw math before a single framework gets imported.',
      'home.attribution': 'Maintained by Rohit Ghumare and contributors. Run on your own machine.',
      'home.star': 'Star on GitHub',
      'home.follow': 'Follow @rohitg00',
      'home.how': 'How this works',
      'home.preface1': "Most AI material teaches in scattered pieces. A paper here, a fine-tuning post there, a flashy agent demo somewhere else. The pieces rarely line up. You ship a chatbot but can't explain its loss curve. You hook a function to an agent but can't say what attention does inside the model that's calling it.",
      'home.preface2': "This curriculum is the spine. 20 phases, 503 lessons, four languages: Python, TypeScript, Rust, Julia. Linear algebra at one end, autonomous swarms at the other. Every algorithm gets built from raw math first. Backprop. Tokenizer. Attention. Agent loop. By the time PyTorch shows up, you already know what it's doing under the hood.",
      'home.preface3': 'Each lesson runs the same loop: read the problem, derive the math, write the code, run the test, keep the artifact. No five-minute videos, no copy-paste deploys, no hand-holding. Free, open source, and built to run on your own laptop.',
      'home.progress': 'Current Progress',
      'home.finishedLessons': 'Finished Lessons',
      'home.phases': 'Phases',
      'home.languages': 'Languages',
      'home.glossaryTerms': 'Glossary Terms',
      'home.tocTitle': 'Curriculum · 20 phases · 503 lessons',
      'home.tocSubtitle': 'Tap a phase to expand its lessons. Each one ships when its math, code, and test are all written.',
      'home.complete': 'Complete',
      'home.inProgress': 'In progress',
      'home.planned': 'Planned',
      'home.progressSaved': 'Progress saved in browser only',
      'home.resetProgress': 'Reset progress',
      'home.colophon': 'Colophon',
      'home.colophonBody': 'The entire curriculum is on GitHub. Clone it, fork it, learn at your own pace. No paywall, no signup. Every lesson has runnable code in Python, TypeScript, Rust, or Julia, depending on what fits the concept best.',
      'catalog.title': 'Lesson Catalog',
      'catalog.subtitle': 'Every lesson across all 20 phases. Search, filter, sort.',
      'catalog.search': 'Search lessons...',
      'catalog.allPhases': 'All Phases',
      'catalog.allStatus': 'All Status',
      'catalog.phase': 'Phase',
      'catalog.lesson': 'Lesson',
      'catalog.type': 'Type',
      'catalog.language': 'Language',
      'catalog.status': 'Status',
      'catalog.count': '{shown} of {total} lessons',
      'catalog.empty': 'No lessons match your filters.',
      'glossary.title': 'AI Glossary',
      'glossary.subtitle': 'What people <em>say</em> vs what things actually <em>mean</em>',
      'glossary.search': 'Search terms...',
      'glossary.count': '{shown} of {total} terms',
      'glossary.empty': 'No terms match your search.',
      'glossary.say': 'What people say',
      'glossary.mean': 'What it actually means',
      'roadmap.title': 'Roadmap',
      'roadmap.subtitle': 'Click any phase to see its prerequisites and what it unlocks downstream.',
      'roadmap.clear': '✕ Clear selection',
      'roadmap.scroll': '↔ Scroll to explore the full graph',
      'roadmap.prerequisites': 'Prerequisites',
      'roadmap.unlocks': 'Unlocks',
      'roadmap.none': 'None. This is a starting point.',
      'roadmap.final': 'Final destination. End of the curriculum.',
      'roadmap.lessonsComplete': '<strong>{done}</strong> of <strong>{total}</strong> lessons complete',
      'roadmap.prereqCount': '<strong>{count}</strong> prerequisite phases',
      'roadmap.unlockCount': '<strong>{count}</strong> phases unlocked',
      'roadmap.read': 'Read',
      'roadmap.github': 'View on GitHub',
      'lesson.loading': 'Loading lesson...',
      'lesson.backHome': 'Back to Home',
      'lesson.previous': '← Previous',
      'lesson.next': 'Next →',
      'lesson.shipsTitle': 'What This Lesson Ships',
      'lesson.shipsSubtitle': 'Prompts, skills, and artifacts you can use right now',
      'lesson.outputsLoading': 'Loading outputs...',
      'lesson.descLoading': 'Loading description...',
      'lesson.codeTitle': 'Run the Code',
      'lesson.codeSubtitle': 'Executable files from this lesson',
      'lesson.codeLoading': 'Loading code files...',
      'lesson.viewGithub': 'View on GitHub',
      'lesson.copyCommand': 'Copy command',
      'lesson.copied': 'Copied!',
      'lesson.copy': 'Copy',
      'lesson.run': 'Run',
      'lesson.running': 'Running',
      'lesson.output': 'Output',
      'lesson.noOutput': '(completed with no output)',
      'lesson.timedOut': 'Timed out.',
      'lesson.exitCode': 'exit code: {code}',
      'lesson.runnerFailed': 'Runner request failed.'
    },
    zh: {
      'nav.contents': '目录',
      'nav.catalog': '课程表',
      'nav.roadmap': '路线图',
      'nav.glossary': '术语表',
      'nav.home': '首页',
      'nav.report': '反馈',
      'nav.reportSuggest': '反馈 / 建议',
      'skip.content': '跳到正文',
      'search.label': '搜索 (⌘K)',
      'theme.toggle': '切换主题',
      'footer.open': 'AI Engineering from Scratch · 开源 · 永久免费。',
      'footer.short': '© 2026 · 开源 · 永久免费',
      'home.metaRight': '开源 · MIT',
      'home.tagline': '503 节课，20 个阶段。先从原始数学手写每个算法，再引入框架。',
      'home.attribution': '由 Rohit Ghumare 和贡献者维护。可在你自己的机器上运行。',
      'home.star': '在 GitHub 点星',
      'home.follow': '关注 @rohitg00',
      'home.how': '学习方式',
      'home.preface1': '大多数 AI 材料都是碎片化的：一篇论文、一个微调帖子、一个炫目的 Agent 演示。它们很少连成体系。你可能能上线一个聊天机器人，却解释不了 loss 曲线；能给 Agent 接一个函数，却说不清模型内部的 attention 在做什么。',
      'home.preface2': '这套课程就是主线。20 个阶段，503 节课，覆盖 Python、TypeScript、Rust、Julia。起点是线性代数，终点是自治智能体群。每个算法都先从数学手写：反向传播、Tokenizer、Attention、Agent 循环。等 PyTorch 出现时，你已经知道它在底层做什么。',
      'home.preface3': '每节课都遵循同一个循环：读懂问题，推导数学，写代码，跑测试，留下可复用产物。没有五分钟速成视频，没有复制粘贴式部署，也没有过度手把手。免费、开源，并且为本地电脑运行而设计。',
      'home.progress': '当前进度',
      'home.finishedLessons': '已完成课程',
      'home.phases': '阶段',
      'home.languages': '语言',
      'home.glossaryTerms': '术语',
      'home.tocTitle': '课程体系 · 20 个阶段 · 503 节课',
      'home.tocSubtitle': '点击阶段展开课程。每节课都在数学、代码和测试完成后交付。',
      'home.complete': '已完成',
      'home.inProgress': '进行中',
      'home.planned': '计划中',
      'home.progressSaved': '进度只保存在当前浏览器',
      'home.resetProgress': '重置进度',
      'home.colophon': '说明',
      'home.colophonBody': '完整课程在 GitHub 上。你可以克隆、fork，并按自己的节奏学习。没有付费墙，也不需要注册。每节课都包含可运行代码，根据概念选择 Python、TypeScript、Rust 或 Julia。',
      'catalog.title': '课程表',
      'catalog.subtitle': '浏览 20 个阶段的所有课程。支持搜索、筛选和排序。',
      'catalog.search': '搜索课程...',
      'catalog.allPhases': '全部阶段',
      'catalog.allStatus': '全部状态',
      'catalog.phase': '阶段',
      'catalog.lesson': '课程',
      'catalog.type': '类型',
      'catalog.language': '语言',
      'catalog.status': '状态',
      'catalog.count': '显示 {shown} / 共 {total} 节课',
      'catalog.empty': '没有课程匹配当前筛选。',
      'glossary.title': 'AI 术语表',
      'glossary.subtitle': '人们<em>常说什么</em>，以及它<em>实际是什么意思</em>',
      'glossary.search': '搜索术语...',
      'glossary.count': '显示 {shown} / 共 {total} 个术语',
      'glossary.empty': '没有术语匹配当前搜索。',
      'glossary.say': '人们常说',
      'glossary.mean': '实际含义',
      'roadmap.title': '路线图',
      'roadmap.subtitle': '点击任意阶段，查看它依赖哪些内容，以及后续会解锁什么。',
      'roadmap.clear': '✕ 清除选择',
      'roadmap.scroll': '↔ 横向滚动查看完整图谱',
      'roadmap.prerequisites': '前置阶段',
      'roadmap.unlocks': '解锁内容',
      'roadmap.none': '无。这是起点。',
      'roadmap.final': '这是课程终点。',
      'roadmap.lessonsComplete': '<strong>{done}</strong> / <strong>{total}</strong> 节课已完成',
      'roadmap.prereqCount': '<strong>{count}</strong> 个前置阶段',
      'roadmap.unlockCount': '解锁 <strong>{count}</strong> 个后续阶段',
      'roadmap.read': '阅读',
      'roadmap.github': '在 GitHub 查看',
      'lesson.loading': '正在加载课程...',
      'lesson.backHome': '返回首页',
      'lesson.previous': '← 上一课',
      'lesson.next': '下一课 →',
      'lesson.shipsTitle': '本课交付内容',
      'lesson.shipsSubtitle': '你现在就能使用的 prompts、skills 和 artifacts',
      'lesson.outputsLoading': '正在加载产物...',
      'lesson.descLoading': '正在加载说明...',
      'lesson.codeTitle': '运行代码',
      'lesson.codeSubtitle': '本课包含的可执行文件',
      'lesson.codeLoading': '正在加载代码文件...',
      'lesson.viewGithub': '在 GitHub 查看',
      'lesson.copyCommand': '复制命令',
      'lesson.copied': '已复制',
      'lesson.copy': '复制',
      'lesson.run': '运行',
      'lesson.running': '运行中',
      'lesson.output': '输出',
      'lesson.noOutput': '（执行完成，无输出）',
      'lesson.timedOut': '执行超时。',
      'lesson.exitCode': '退出码：{code}',
      'lesson.runnerFailed': 'Runner 请求失败。'
    }
  };

  function currentLang() {
    var lang = localStorage.getItem(LANG_KEY) || DEFAULT_LANG;
    return I18N[lang] ? lang : DEFAULT_LANG;
  }

  function t(key, vars) {
    var lang = currentLang();
    var value = (I18N[lang] && I18N[lang][key]) || I18N.en[key] || key;
    if (vars) {
      Object.keys(vars).forEach(function (name) {
        value = value.replace(new RegExp('\\{' + name + '\\}', 'g'), vars[name]);
      });
    }
    return value;
  }

  function setLang(lang) {
    if (!I18N[lang]) lang = DEFAULT_LANG;
    localStorage.setItem(LANG_KEY, lang);
    document.documentElement.setAttribute('lang', lang === 'zh' ? 'zh-CN' : 'en');
    applyI18n();
    window.dispatchEvent(new CustomEvent('aifs:languagechange', { detail: { lang: lang } }));
  }

  function applyI18n() {
    var lang = currentLang();
    document.documentElement.setAttribute('lang', lang === 'zh' ? 'zh-CN' : 'en');

    document.querySelectorAll('[data-i18n]').forEach(function (el) {
      var value = t(el.getAttribute('data-i18n'));
      if (el.getAttribute('data-i18n-html') === 'true') el.innerHTML = value;
      else el.textContent = value;
    });
    document.querySelectorAll('[data-i18n-placeholder]').forEach(function (el) {
      el.setAttribute('placeholder', t(el.getAttribute('data-i18n-placeholder')));
    });
    document.querySelectorAll('[data-i18n-label]').forEach(function (el) {
      var value = t(el.getAttribute('data-i18n-label'));
      el.setAttribute('aria-label', value);
      el.setAttribute('title', value);
    });

    var select = document.getElementById('languageSelect');
    if (select) {
      select.value = lang;
      select.setAttribute('aria-label', lang === 'zh' ? '语言' : 'Language');
      select.setAttribute('title', lang === 'zh' ? '语言' : 'Language');
    }
  }

  function initLanguageSelect() {
    var select = document.getElementById('languageSelect');
    if (!select) return;
    select.value = currentLang();
    select.addEventListener('change', function () {
      setLang(select.value);
    });
    applyI18n();
  }

  function format(n) {
    if (n >= 10000) return (n / 1000).toFixed(1).replace(/\.0$/, '') + 'k';
    if (n >= 1000) return (n / 1000).toFixed(1).replace(/\.0$/, '') + 'k';
    return String(n);
  }

  function paint(n) {
    var els = document.querySelectorAll(
      '.header-github .star-count, #starCount, [data-gh-stars="' + REPO + '"]'
    );
    for (var i = 0; i < els.length; i++) {
      els[i].textContent = format(n);
      els[i].removeAttribute('data-loading');
    }
  }

  function readCache() {
    try {
      var raw = localStorage.getItem(CACHE_KEY);
      if (!raw) return null;
      var parsed = JSON.parse(raw);
      if (Date.now() - parsed.t > CACHE_TTL_MS) return null;
      return parsed.n;
    } catch (e) {
      return null;
    }
  }

  function writeCache(n) {
    try {
      localStorage.setItem(CACHE_KEY, JSON.stringify({ n: n, t: Date.now() }));
    } catch (e) {
      // localStorage may be disabled
    }
  }

  function load() {
    var cached = readCache();
    if (cached != null) {
      paint(cached);
      return;
    }
    fetch('https://api.github.com/repos/' + REPO, {
      headers: { Accept: 'application/vnd.github+json' },
    })
      .then(function (r) {
        if (!r.ok) throw new Error('gh ' + r.status);
        return r.json();
      })
      .then(function (data) {
        var n = data.stargazers_count;
        if (typeof n !== 'number') return;
        writeCache(n);
        paint(n);
      })
      .catch(function () {
        // Leave the placeholder; the link still works.
      });
  }

  window.AIFSI18N = {
    currentLang: currentLang,
    setLang: setLang,
    t: t,
    apply: applyI18n
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
      initLanguageSelect();
      load();
    });
  } else {
    initLanguageSelect();
    load();
  }
})();
