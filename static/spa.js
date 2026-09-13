/**
 * Sentinel SPA Router
 * Intercepts sidebar nav clicks, loads pages via AJAX, swaps content.
 * Preserves: header, sidebar, analytics engines, video streams, map state.
 */
(function() {
    'use strict';

    const CONTENT_SELECTOR = '#page-content';
    let currentPage = window.location.pathname;
    let pageCache = {};
    let activeTimers = [];

    function clearTimers() {
        activeTimers.forEach(id => { clearInterval(id); clearTimeout(id); });
        activeTimers = [];
    }

    window.sentinelInterval = function(fn, ms) {
        const id = setInterval(fn, ms);
        activeTimers.push(id);
        return id;
    };

    function updateActiveNav(path) {
        document.querySelectorAll('.nav-link').forEach(link => {
            const href = link.getAttribute('href') || '';
            link.classList.toggle('active', href === path);
        });
    }

    function extractPageParts(html) {
        const parser = new DOMParser();
        const doc = parser.parseFromString(html, 'text/html');
        const content = doc.querySelector(CONTENT_SELECTOR);
        const scripts = Array.from(doc.querySelectorAll('script'));
        return {
            html: content ? content.innerHTML : '',
            scripts: scripts.map(s => ({ text: s.textContent, src: s.src }))
        };
    }

    function executeScripts(scripts) {
        scripts.forEach(s => {
            if (s.src) {
                if (document.querySelector(`script[src="${s.src}"]`)) return;
                const el = document.createElement('script');
                el.src = s.src;
                document.head.appendChild(el);
            } else if (s.text && s.text.trim()) {
                const el = document.createElement('script');
                el.textContent = s.text;
                document.head.appendChild(el);
                el.remove();
            }
        });
    }

    function loadPage(path, pushState) {
        if (path === currentPage && pushState) return;

        clearTimers();

        const contentEl = document.querySelector(CONTENT_SELECTOR);
        if (!contentEl) return;

        if (pageCache[path]) {
            const cached = pageCache[path];
            contentEl.innerHTML = cached.html;
            currentPage = path;
            updateActiveNav(path);
            if (pushState) history.pushState({ path }, '', path);
            executeScripts(cached.scripts);
            window.scrollTo(0, 0);
            return;
        }

        contentEl.innerHTML = `
            <div class="flex items-center justify-center h-full">
                <div class="text-center">
                    <div class="inline-block h-8 w-8 animate-spin rounded-full border-4 border-indigo-500 border-r-transparent mb-3"></div>
                    <p class="text-slate-400 text-sm">Loading...</p>
                </div>
            </div>`;

        fetch(path)
            .then(r => r.text())
            .then(html => {
                const parts = extractPageParts(html);
                pageCache[path] = parts;
                contentEl.innerHTML = parts.html;
                currentPage = path;
                updateActiveNav(path);
                if (pushState) history.pushState({ path }, '', path);
                executeScripts(parts.scripts);
                window.scrollTo(0, 0);
            })
            .catch(err => {
                console.error('[SPA] Load error:', err);
                contentEl.innerHTML = `
                    <div class="flex items-center justify-center h-full">
                        <div class="text-center text-red-400">
                            <p class="text-lg font-semibold">Failed to load page</p>
                            <p class="text-sm mt-2 text-slate-500">${err.message}</p>
                        </div>
                    </div>`;
            });
    }

    function initRouter() {
        document.addEventListener('click', function(e) {
            const navLink = e.target.closest('.nav-link');
            if (!navLink) return;

            const href = navLink.getAttribute('href');
            if (!href || href.startsWith('http') || href.startsWith('#') || href.startsWith('javascript:')) return;

            e.preventDefault();
            loadPage(href, true);
        });

        window.addEventListener('popstate', function(e) {
            const path = (e.state && e.state.path) || window.location.pathname;
            loadPage(path, false);
        });
    }

    window.spaNavigate = function(path) { loadPage(path, true); };
    window.spaPreload = function(path) {
        if (!pageCache[path]) {
            fetch(path).then(r => r.text()).then(html => {
                pageCache[path] = extractPageParts(html);
            });
        }
    };

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initRouter);
    } else {
        initRouter();
    }
})();
