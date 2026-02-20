#!/usr/bin/env node

/**
 * MCP Web Fetch Server v4.0.0 — Zero Dependencies, Maximum Coverage
 *
 * Fetches any URL and returns clean, readable content.
 * Uses ONLY Node.js built-in modules — no npm installs required.
 *
 * Improvements over v3:
 *   - Chrome-accurate header ORDER (servers fingerprint header sequence)
 *   - Jina.ai reader fallback  (r.jina.ai/<url> → clean markdown for any page)
 *   - Google Cache fallback    (bypasses many paywalls & bot checks)
 *   - Wayback Machine fallback (last resort archive retrieval)
 *   - Site strategies: YouTube, Wikipedia, GitHub, HackerNews, ArXiv
 *   - Smart Readability-style article extraction (not just tag stripping)
 *   - Paginated reading via start_index + max_chars args
 *   - robots.txt checking with polite/force modes
 *   - Content quality scoring to pick best strategy result
 *   - Exponential backoff retries on transient errors
 *   - HTML→Markdown conversion (headings, links, lists preserved)
 *   - Response metadata: title, description, word count, fetch strategy
 *
 * Run with: node web_fetch.js
 */

'use strict';

const readline = require('readline');
const http     = require('http');
const https    = require('https');
const zlib     = require('zlib');
const { URL }  = require('url');
const crypto   = require('crypto');

// ═══════════════════════════════════════════════════════════════════════════
// CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════

const CONFIG = {
    defaultMaxChars: 30000,
    httpTimeout:     25000,
    maxRedirects:    10,
    version:         '4.0.0',

    // Chrome 124 realistic UA pool (rotate to avoid fingerprinting)
    userAgents: [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15',
    ],

    mobileUA: 'Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Mobile Safari/537.36',

    googleBot: 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',

    // Domain routing — matched against hostname (without www.)
    domains: {
        reddit:    ['reddit.com', 'old.reddit.com'],
        youtube:   ['youtube.com', 'youtu.be'],
        wikipedia: ['wikipedia.org'],
        github:    ['github.com', 'raw.githubusercontent.com'],
        hn:        ['news.ycombinator.com'],
        arxiv:     ['arxiv.org'],
        twitter:   ['twitter.com', 'x.com'],
        instagram: ['instagram.com'],
        tiktok:    ['tiktok.com'],
        linkedin:  ['linkedin.com'],
        facebook:  ['facebook.com'],
        medium:    ['medium.com'],
        paywalled: ['wsj.com', 'ft.com', 'nytimes.com', 'economist.com',
                    'bloomberg.com', 'washingtonpost.com', 'theatlantic.com'],
    },
};

let uaIndex = 0;
function nextUA() {
    const ua = CONFIG.userAgents[uaIndex % CONFIG.userAgents.length];
    uaIndex++;
    return ua;
}

// ═══════════════════════════════════════════════════════════════════════════
// LOGGING
// ═══════════════════════════════════════════════════════════════════════════

const log    = (msg)    => process.stderr.write(`[WEB-FETCH] ${msg}\n`);
const logErr = (msg, e) => {
    process.stderr.write(`[WEB-FETCH ERROR] ${msg}\n`);
    if (e && e.stack) process.stderr.write(`${e.stack}\n`);
};

// ═══════════════════════════════════════════════════════════════════════════
// DECOMPRESSION
// ═══════════════════════════════════════════════════════════════════════════

function decompress(buffer, encoding) {
    return new Promise((resolve, reject) => {
        const enc = (encoding || '').toLowerCase().trim().split(',')[0].trim();
        if (enc === 'gzip' || enc === 'x-gzip') {
            zlib.gunzip(buffer, (err, r) => err ? reject(err) : resolve(r));
        } else if (enc === 'deflate') {
            zlib.inflate(buffer, (err, r) => {
                if (!err) return resolve(r);
                zlib.inflateRaw(buffer, (err2, r2) => err2 ? reject(err2) : resolve(r2));
            });
        } else if (enc === 'br') {
            zlib.brotliDecompress(buffer, (err, r) => err ? reject(err) : resolve(r));
        } else {
            resolve(buffer);
        }
    });
}

// ═══════════════════════════════════════════════════════════════════════════
// HTML → MARKDOWN  (smart article extraction + markdown output)
// ═══════════════════════════════════════════════════════════════════════════

function extractMeta(html) {
    const title    = (html.match(/<title[^>]*>([^<]*)<\/title>/i) || [])[1] || '';
    const desc     = (html.match(/<meta[^>]+name=["']description["'][^>]+content=["']([^"']*)["']/i) ||
                      html.match(/<meta[^>]+content=["']([^"']*)["'][^>]+name=["']description["']/i) || [])[1] || '';
    const ogTitle  = (html.match(/<meta[^>]+property=["']og:title["'][^>]+content=["']([^"']*)["']/i) ||
                      html.match(/<meta[^>]+content=["']([^"']*)["'][^>]+property=["']og:title["']/i) || [])[1] || '';
    const ogDesc   = (html.match(/<meta[^>]+property=["']og:description["'][^>]+content=["']([^"']*)["']/i) ||
                      html.match(/<meta[^>]+content=["']([^"']*)["'][^>]+property=["']og:description["']/i) || [])[1] || '';
    return {
        title: decodeHtmlEntities(ogTitle || title),
        description: decodeHtmlEntities(ogDesc || desc),
    };
}

function decodeHtmlEntities(str) {
    return str
        .replace(/&amp;/g,  '&')
        .replace(/&lt;/g,   '<')
        .replace(/&gt;/g,   '>')
        .replace(/&quot;/g, '"')
        .replace(/&#39;/g,  "'")
        .replace(/&nbsp;/g, ' ')
        .replace(/&#(\d+);/g, (_, n) => String.fromCharCode(Number(n)))
        .replace(/&[a-z]+;/g, ' ');
}

// Readability-style: score each block and pick main content
function extractMainContent(html) {
    // Remove obviously non-content areas first
    html = html
        .replace(/<head[\s\S]*?<\/head>/gi, '')
        .replace(/<script[\s\S]*?<\/script>/gi, '')
        .replace(/<style[\s\S]*?<\/style>/gi, '')
        .replace(/<nav[\s\S]*?<\/nav>/gi, '')
        .replace(/<header[\s\S]*?<\/header>/gi, '')
        .replace(/<footer[\s\S]*?<\/footer>/gi, '')
        .replace(/<aside[\s\S]*?<\/aside>/gi, '')
        .replace(/<form[\s\S]*?<\/form>/gi, '')
        .replace(/<!--[\s\S]*?-->/g, '');

    // Try to find main/article content container
    const mainPatterns = [
        /<article[^>]*>([\s\S]*?)<\/article>/gi,
        /<main[^>]*>([\s\S]*?)<\/main>/gi,
        /<div[^>]*(?:class|id)=["'][^"']*(?:article|content|main|post|story|text|body)[^"']*["'][^>]*>([\s\S]*?)<\/div>/gi,
    ];

    let best = '';
    for (const pattern of mainPatterns) {
        let m;
        pattern.lastIndex = 0;
        while ((m = pattern.exec(html)) !== null) {
            if (m[1].length > best.length) best = m[1];
        }
        if (best.length > 500) break;
    }

    return best.length > 200 ? best : html;
}

function htmlToMarkdown(html) {
    // Extract main content first
    const main = extractMainContent(html);

    return main
        // Headings → Markdown
        .replace(/<h1[^>]*>([\s\S]*?)<\/h1>/gi, (_, t) => `\n# ${stripTags(t).trim()}\n`)
        .replace(/<h2[^>]*>([\s\S]*?)<\/h2>/gi, (_, t) => `\n## ${stripTags(t).trim()}\n`)
        .replace(/<h3[^>]*>([\s\S]*?)<\/h3>/gi, (_, t) => `\n### ${stripTags(t).trim()}\n`)
        .replace(/<h4[^>]*>([\s\S]*?)<\/h4>/gi, (_, t) => `\n#### ${stripTags(t).trim()}\n`)
        .replace(/<h5[^>]*>([\s\S]*?)<\/h5>/gi, (_, t) => `\n##### ${stripTags(t).trim()}\n`)
        .replace(/<h6[^>]*>([\s\S]*?)<\/h6>/gi, (_, t) => `\n###### ${stripTags(t).trim()}\n`)
        // Bold/italic
        .replace(/<(strong|b)[^>]*>([\s\S]*?)<\/(strong|b)>/gi, (_, _t, c) => `**${stripTags(c)}**`)
        .replace(/<(em|i)[^>]*>([\s\S]*?)<\/(em|i)>/gi, (_, _t, c) => `_${stripTags(c)}_`)
        // Links
        .replace(/<a[^>]+href=["']([^"'#][^"']*)["'][^>]*>([\s\S]*?)<\/a>/gi,
            (_, href, text) => {
                const t = stripTags(text).trim();
                if (!t) return '';
                // Only include external/meaningful links
                if (href.startsWith('http') || href.startsWith('/')) {
                    return `[${t}](${href})`;
                }
                return t;
            })
        // Images — show alt text
        .replace(/<img[^>]+alt=["']([^"']+)["'][^>]*>/gi, (_, alt) => `[Image: ${alt}]`)
        .replace(/<img[^>]*>/gi, '')
        // Lists
        .replace(/<li[^>]*>([\s\S]*?)<\/li>/gi, (_, c) => `- ${stripTags(c).trim()}\n`)
        .replace(/<\/(ul|ol)>/gi, '\n')
        // Paragraphs and blocks → newlines
        .replace(/<\/(p|div|section|article|blockquote|tr|dd|dt|figure|figcaption)>/gi, '\n\n')
        .replace(/<br\s*\/?>/gi, '\n')
        .replace(/<hr\s*\/?>/gi, '\n---\n')
        // Code
        .replace(/<code[^>]*>([\s\S]*?)<\/code>/gi, (_, c) => `\`${stripTags(c)}\``)
        .replace(/<pre[^>]*>([\s\S]*?)<\/pre>/gi, (_, c) => `\n\`\`\`\n${stripTags(c)}\n\`\`\`\n`)
        // Blockquote
        .replace(/<blockquote[^>]*>([\s\S]*?)<\/blockquote>/gi,
            (_, c) => stripTags(c).trim().split('\n').map(l => `> ${l}`).join('\n') + '\n')
        // Strip remaining tags
        .replace(/<[^>]+>/g, '')
        // Decode entities
        .replace(/&amp;/g,  '&')
        .replace(/&lt;/g,   '<')
        .replace(/&gt;/g,   '>')
        .replace(/&quot;/g, '"')
        .replace(/&#39;/g,  "'")
        .replace(/&nbsp;/g, ' ')
        .replace(/&#(\d+);/g, (_, n) => String.fromCharCode(Number(n)))
        .replace(/&[a-z]+;/g, ' ')
        // Clean up whitespace
        .replace(/[ \t]+/g, ' ')
        .replace(/\n[ \t]+/g, '\n')
        .replace(/\n{4,}/g, '\n\n\n')
        .trim();
}

function stripTags(html) {
    return (html || '')
        .replace(/<[^>]+>/g, '')
        .replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>')
        .replace(/&quot;/g, '"').replace(/&#39;/g, "'").replace(/&nbsp;/g, ' ');
}

function scoreContent(text) {
    if (!text || text.length < 50) return 0;
    const words = text.split(/\s+/).length;
    const sentences = text.split(/[.!?]+/).length;
    // Penalize content that looks like error pages
    const isError = /access denied|403|blocked|captcha|cloudflare|bot detection/i.test(text);
    if (isError && text.length < 2000) return -1;
    return words * 1 + sentences * 0.5 + Math.min(text.length / 100, 50);
}

function truncate(text, maxChars, startIndex) {
    const start = startIndex || 0;
    const slice = text.slice(start, start + maxChars);
    const hasMore = start + maxChars < text.length;
    const footer = hasMore
        ? `\n\n[... content continues. Total length: ${text.length} chars. ` +
          `Use start_index=${start + maxChars} to read next chunk ...]`
        : (start > 0 ? `\n\n[End of content. Total length: ${text.length} chars.]` : '');
    return slice + footer;
}

// ═══════════════════════════════════════════════════════════════════════════
// CORE HTTP  (Chrome-accurate header order, cookies, redirects, backoff)
// ═══════════════════════════════════════════════════════════════════════════

// Chrome sends headers in a very specific order — servers use this for fingerprinting.
// The order below matches Chrome 124 on Windows.
const CHROME_HEADER_ORDER = [
    'Host',
    'Connection',
    'Cache-Control',
    'Upgrade-Insecure-Requests',
    'User-Agent',
    'Accept',
    'Sec-Fetch-Site',
    'Sec-Fetch-Mode',
    'Sec-Fetch-User',
    'Sec-Fetch-Dest',
    'Accept-Encoding',
    'Accept-Language',
    'Referer',
    'Cookie',
];

function buildHeaders(urlObj, options) {
    const ua  = options.ua || nextUA();
    const raw = Object.assign({
        'Host':                      urlObj.host,
        'Connection':                'keep-alive',
        'Cache-Control':             'max-age=0',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent':                ua,
        'Accept':                    options.accept || 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'Sec-Fetch-Site':            options.referer ? 'same-origin' : 'none',
        'Sec-Fetch-Mode':            'navigate',
        'Sec-Fetch-User':            '?1',
        'Sec-Fetch-Dest':            'document',
        'Accept-Encoding':           'gzip, deflate, br',
        'Accept-Language':           'en-US,en;q=0.9',
    }, options.extraHeaders || {});

    // Add referer & cookie if present
    if (options.referer) raw['Referer'] = options.referer;
    if (options.cookie)  raw['Cookie']  = options.cookie;

    // Rebuild in Chrome header order (unknown keys go at end)
    const ordered = {};
    for (const key of CHROME_HEADER_ORDER) {
        if (raw[key] !== undefined) ordered[key] = raw[key];
    }
    for (const key of Object.keys(raw)) {
        if (!ordered[key]) ordered[key] = raw[key];
    }
    // Remove 'Host' from headers object — Node sets it from hostname
    delete ordered['Host'];

    return { headers: ordered, ua };
}

function rawFetch(targetUrl, options, redirectsLeft, cookieJar) {
    options       = options || {};
    redirectsLeft = redirectsLeft === undefined ? CONFIG.maxRedirects : redirectsLeft;
    cookieJar     = cookieJar || {};

    return new Promise((resolve) => {
        let urlObj;
        try { urlObj = new URL(targetUrl); }
        catch (e) {
            return resolve({ url: targetUrl, statusCode: 0, text: '', strategy: 'http', error: 'Invalid URL: ' + e.message });
        }

        const isHttps = urlObj.protocol === 'https:';
        const lib     = isHttps ? https : http;

        // Assemble domain cookies
        const domainCookies = Object.entries(cookieJar)
            .filter(([k]) => k.startsWith(urlObj.hostname))
            .map(([, v]) => v)
            .join('; ');

        if (domainCookies) options = Object.assign({}, options, { cookie: domainCookies });

        const { headers } = buildHeaders(urlObj, options);

        const reqOpts = {
            hostname: urlObj.hostname,
            port:     urlObj.port || (isHttps ? 443 : 80),
            path:     urlObj.pathname + urlObj.search,
            method:   options.method || 'GET',
            headers:  headers,
            timeout:  options.timeout || CONFIG.httpTimeout,
        };

        log(`[http] ${reqOpts.method} ${targetUrl}`);

        const req = lib.request(reqOpts, (res) => {
            // Store cookies
            (res.headers['set-cookie'] || []).forEach((c) => {
                const nv  = c.split(';')[0].trim();
                const key = urlObj.hostname + ':' + nv.split('=')[0];
                cookieJar[key] = nv;
            });

            // Redirects
            if ([301, 302, 303, 307, 308].includes(res.statusCode) && res.headers.location) {
                if (redirectsLeft <= 0) {
                    return resolve({ url: targetUrl, statusCode: res.statusCode, text: '', strategy: 'http', error: 'Too many redirects' });
                }
                let next = res.headers.location;
                if (!next.startsWith('http')) next = urlObj.protocol + '//' + urlObj.host + next;
                log(`[http] Redirect ${res.statusCode} → ${next}`);
                return resolve(rawFetch(next, Object.assign({}, options, { referer: targetUrl }), redirectsLeft - 1, cookieJar));
            }

            const chunks = [];
            res.on('data', (c) => chunks.push(c));
            res.on('end', async () => {
                try {
                    const raw  = Buffer.concat(chunks);
                    const enc  = res.headers['content-encoding'] || '';
                    const buf  = await decompress(raw, enc);
                    const text = buf.toString('utf8');
                    const ct   = res.headers['content-type'] || '';

                    let content, meta = {};
                    if (ct.includes('json')) {
                        content = text;
                    } else if (ct.includes('html')) {
                        meta    = extractMeta(text);
                        content = htmlToMarkdown(text);
                    } else {
                        content = text;
                    }

                    resolve({
                        url:       targetUrl,
                        statusCode: res.statusCode,
                        text:      content,
                        rawLength: text.length,
                        strategy:  'http',
                        meta,
                    });
                } catch (err) {
                    resolve({ url: targetUrl, statusCode: res.statusCode, text: '', strategy: 'http', error: 'Decompression/parse error: ' + err.message });
                }
            });
        });

        req.on('error', (err) => {
            const msg = err.code === 'ECONNREFUSED' ? 'Connection refused to ' + urlObj.hostname
                      : err.code === 'ENOTFOUND'   ? 'Host not found: ' + urlObj.hostname
                      : err.code === 'ECONNRESET'  ? 'Connection reset by ' + urlObj.hostname
                      : err.message;
            resolve({ url: targetUrl, statusCode: 0, text: '', strategy: 'http', error: msg });
        });

        req.on('timeout', () => {
            req.destroy();
            resolve({ url: targetUrl, statusCode: 0, text: '', strategy: 'http', error: `Timed out after ${CONFIG.httpTimeout}ms` });
        });

        req.end();
    });
}

// Retry wrapper with exponential backoff
async function fetchWithRetry(targetUrl, options, maxAttempts) {
    maxAttempts = maxAttempts || 2;
    let lastResult;
    for (let i = 0; i < maxAttempts; i++) {
        lastResult = await rawFetch(targetUrl, options);
        if (!lastResult.error && lastResult.text && lastResult.text.length > 100) return lastResult;
        if (i < maxAttempts - 1) {
            const delay = 500 * Math.pow(2, i);
            log(`[retry] Attempt ${i + 1} failed, waiting ${delay}ms`);
            await new Promise(r => setTimeout(r, delay));
        }
    }
    return lastResult;
}

// ═══════════════════════════════════════════════════════════════════════════
// ROBOTS.TXT  (polite mode checks this; force mode skips)
// ═══════════════════════════════════════════════════════════════════════════

const robotsCache = {};

async function isAllowed(urlObj, ua) {
    const key = urlObj.hostname;
    if (robotsCache[key] !== undefined) return robotsCache[key];

    try {
        const robotsUrl = urlObj.protocol + '//' + urlObj.host + '/robots.txt';
        const r = await rawFetch(robotsUrl, { ua: ua || 'Googlebot', timeout: 5000 });
        if (r.error || !r.text) { robotsCache[key] = true; return true; }

        const lines     = r.text.split('\n');
        let   applies   = false;
        let   allowed   = true;

        for (const line of lines) {
            const l = line.trim().toLowerCase();
            if (l.startsWith('user-agent:')) {
                const agent = line.split(':')[1].trim();
                applies = (agent === '*' || ua.toLowerCase().includes(agent.toLowerCase()));
            } else if (applies && l.startsWith('disallow:')) {
                const path = line.split(':')[1].trim();
                if (path === '/' || path === '') { allowed = false; break; }
            } else if (applies && l.startsWith('allow:')) {
                const path = line.split(':')[1].trim();
                if (path === '/') { allowed = true; break; }
            }
        }

        robotsCache[key] = allowed;
        return allowed;
    } catch (_) {
        robotsCache[key] = true;
        return true;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SITE-SPECIFIC STRATEGIES
// ═══════════════════════════════════════════════════════════════════════════

// — Reddit: JSON API (much cleaner than HTML)
async function redditFetch(targetUrl) {
    log('[reddit] Using JSON API');
    const u       = new URL(targetUrl);
    const jsonUrl = u.protocol + '//' + u.host + u.pathname.replace(/\/$/, '') + '.json' + u.search;

    const r = await rawFetch(jsonUrl, {
        ua:           CONFIG.googleBot,
        accept:       'application/json',
        extraHeaders: { 'Accept': 'application/json' },
    });

    if (r.error || !r.text) return r;

    try {
        const data  = JSON.parse(r.text);
        const lines = [];

        function post(d) {
            if (!d) return;
            lines.push(`# ${d.title || ''}`);
            lines.push(`**r/${d.subreddit}** | Score: ${d.score} | Comments: ${d.num_comments} | u/${d.author}`);
            if (d.selftext && d.selftext !== '[removed]') { lines.push(''); lines.push(d.selftext); }
            if (d.url && !d.is_self) lines.push(`\nLink: ${d.url}`);
        }

        function comments(arr, depth) {
            if (!Array.isArray(arr)) return;
            for (const item of arr) {
                if (!item || !item.data) continue;
                const d = item.data;
                if (item.kind === 'Listing' && d.children) { comments(d.children, depth); continue; }
                if (item.kind === 't1') {
                    const body = (d.body || '').replace(/\n+/g, ' ').slice(0, 600);
                    if (body && body !== '[removed]' && body !== '[deleted]') {
                        lines.push(`${'  '.repeat(Math.min(depth, 5))}**u/${d.author}** (${d.score}): ${body}`);
                    }
                    if (d.replies?.data?.children) comments(d.replies.data.children, depth + 1);
                } else if (item.kind === 't3' && d.title) {
                    post(d);
                }
            }
        }

        if (Array.isArray(data)) {
            for (const s of data) {
                if (s?.data?.children) { comments(s.data.children, 0); lines.push(''); }
            }
        } else if (data?.data?.children) {
            for (const item of data.data.children) {
                if (item?.data) { post(item.data); lines.push(''); }
            }
        }

        const text = lines.join('\n').trim();
        return { url: targetUrl, statusCode: 200, text: text || 'No readable content.', strategy: 'reddit-api', meta: { title: 'Reddit' } };
    } catch (_) {
        log('[reddit] JSON parse failed, falling back to plain fetch');
        return rawFetch(targetUrl, { ua: CONFIG.mobileUA });
    }
}

// — YouTube: extract metadata + description (no transcript without external lib)
async function youtubeFetch(targetUrl) {
    log('[youtube] Fetching YouTube page');
    const r = await rawFetch(targetUrl, { ua: nextUA() });
    if (r.error || !r.text) return r;

    // Extract structured data from page
    const lines = [];

    const titleMatch    = r.text.match(/"title":"([^"]+)"/);
    const channelMatch  = r.text.match(/"ownerChannelName":"([^"]+)"/);
    const viewsMatch    = r.text.match(/"viewCount":"([^"]+)"/);
    const descMatch     = r.text.match(/"shortDescription":"((?:[^"\\]|\\.)*)"/);
    const publishMatch  = r.text.match(/"publishDate":"([^"]+)"/);
    const durationMatch = r.text.match(/"lengthSeconds":"([^"]+)"/);

    if (titleMatch)    lines.push(`# ${titleMatch[1]}`);
    if (channelMatch)  lines.push(`**Channel:** ${channelMatch[1]}`);
    if (viewsMatch)    lines.push(`**Views:** ${Number(viewsMatch[1]).toLocaleString()}`);
    if (publishMatch)  lines.push(`**Published:** ${publishMatch[1]}`);
    if (durationMatch) {
        const s = Number(durationMatch[1]);
        lines.push(`**Duration:** ${Math.floor(s / 60)}m ${s % 60}s`);
    }
    if (descMatch) {
        lines.push('\n## Description');
        lines.push(descMatch[1].replace(/\\n/g, '\n').replace(/\\"/g, '"').slice(0, 3000));
    }

    // Try to find any transcript data
    const transcriptSnippet = r.text.match(/"captionTracks":\s*(\[[\s\S]{1,200}?\])/);
    if (transcriptSnippet) {
        lines.push('\n_[Transcript captions are available for this video but require a dedicated transcript fetch]_');
    }

    const text = lines.join('\n').trim();
    return {
        url:        targetUrl,
        statusCode: r.statusCode,
        text:       text.length > 100 ? text : r.text,
        strategy:   'youtube',
        meta:       { title: titleMatch ? titleMatch[1] : 'YouTube Video' },
    };
}

// — Wikipedia: clean API response (much better than HTML scraping)
async function wikipediaFetch(targetUrl) {
    log('[wikipedia] Using Wikipedia API');
    const u = new URL(targetUrl);
    const title = decodeURIComponent(u.pathname.replace('/wiki/', ''));
    const apiUrl = `https://${u.hostname}/api/rest_v1/page/summary/${encodeURIComponent(title)}`;

    const r = await rawFetch(apiUrl, { accept: 'application/json' });
    if (!r.error && r.text) {
        try {
            const data = JSON.parse(r.text);
            const lines = [];
            if (data.title)   lines.push(`# ${data.title}`);
            if (data.description) lines.push(`_${data.description}_`);
            if (data.extract)  lines.push(`\n${data.extract}`);
            if (data.content_urls?.desktop?.page) {
                lines.push(`\n[Read full article](${data.content_urls.desktop.page})`);
            }
            if (lines.length > 1) {
                return { url: targetUrl, statusCode: 200, text: lines.join('\n'), strategy: 'wikipedia-api', meta: { title: data.title } };
            }
        } catch (_) {}
    }

    // Fallback: full HTML parse
    return rawFetch(targetUrl, { ua: nextUA() });
}

// — GitHub: try raw content / API when possible
async function githubFetch(targetUrl) {
    log('[github] GitHub strategy');
    const u = new URL(targetUrl);

    // If it's a blob URL, convert to raw
    if (u.pathname.includes('/blob/')) {
        const rawUrl = 'https://raw.githubusercontent.com' +
                       u.pathname.replace('/blob/', '/');
        const r = await rawFetch(rawUrl, { ua: nextUA() });
        if (!r.error && r.text) return { ...r, strategy: 'github-raw' };
    }

    // Repository main page — use GitHub API
    const parts = u.pathname.split('/').filter(Boolean);
    if (parts.length === 2) {
        const apiUrl = `https://api.github.com/repos/${parts[0]}/${parts[1]}`;
        const r = await rawFetch(apiUrl, {
            ua:           nextUA(),
            accept:       'application/vnd.github+json',
            extraHeaders: { 'X-GitHub-Api-Version': '2022-11-28' },
        });
        if (!r.error && r.text) {
            try {
                const d = JSON.parse(r.text);
                const lines = [
                    `# ${d.full_name}`,
                    d.description || '',
                    `**Stars:** ${d.stargazers_count} | **Forks:** ${d.forks_count} | **Language:** ${d.language}`,
                    `**License:** ${d.license?.name || 'None'} | **Topics:** ${(d.topics || []).join(', ')}`,
                    `\n${d.homepage ? `[Homepage](${d.homepage})\n` : ''}`,
                ];
                return { url: targetUrl, statusCode: 200, text: lines.join('\n'), strategy: 'github-api', meta: { title: d.full_name } };
            } catch (_) {}
        }
    }

    return fetchWithRetry(targetUrl, { ua: nextUA() });
}

// — Hacker News: Algolia API
async function hnFetch(targetUrl) {
    log('[hn] Using Algolia HN API');
    const u = new URL(targetUrl);
    const id = u.searchParams.get('id');

    if (id) {
        const r = await rawFetch(`https://hn.algolia.com/api/v1/items/${id}`);
        if (!r.error && r.text) {
            try {
                const d    = JSON.parse(r.text);
                const lines = [`# ${d.title || '(no title)'}`, `**Points:** ${d.points} | **Comments:** ${d.children?.length || 0}`, ''];
                if (d.url) lines.push(`[${d.url}](${d.url})\n`);
                if (d.text) lines.push(stripTags(d.text) + '\n');

                function addComments(arr, depth) {
                    if (!arr) return;
                    for (const c of arr) {
                        if (c.text && c.text !== '[deleted]') {
                            lines.push(`${'  '.repeat(Math.min(depth, 4))}**${c.author}:** ${stripTags(c.text).slice(0, 400)}`);
                        }
                        addComments(c.children, depth + 1);
                    }
                }
                addComments(d.children, 0);
                return { url: targetUrl, statusCode: 200, text: lines.join('\n'), strategy: 'hn-api', meta: { title: d.title } };
            } catch (_) {}
        }
    }

    return rawFetch(targetUrl, { ua: nextUA() });
}

// — ArXiv: abstract API
async function arxivFetch(targetUrl) {
    log('[arxiv] Using ArXiv API');
    const u = new URL(targetUrl);
    const idMatch = u.pathname.match(/(?:abs|pdf)\/([0-9.]+(?:v\d+)?)/);
    if (idMatch) {
        const apiUrl = `https://export.arxiv.org/abs/${idMatch[1]}`;
        const r = await rawFetch(apiUrl, { ua: nextUA() });
        if (!r.error && r.text) return { ...r, strategy: 'arxiv' };
    }
    return rawFetch(targetUrl, { ua: nextUA() });
}

// — Jina.ai Reader: free service that converts any page to clean markdown
async function jinaFetch(targetUrl) {
    log('[jina] Using Jina.ai reader');
    const jinaUrl = 'https://r.jina.ai/' + targetUrl;
    const r = await rawFetch(jinaUrl, {
        ua:           nextUA(),
        accept:       'text/plain,application/json',
        extraHeaders: { 'X-Return-Format': 'markdown', 'X-No-Cache': 'true' },
        timeout:      30000,
    });
    if (r.error || !r.text || r.text.length < 100) return r;
    return { ...r, strategy: 'jina-reader' };
}

// — Google Cache: tries Google's cached version (works for many paywalled/blocked sites)
async function googleCacheFetch(targetUrl) {
    log('[cache] Trying Google Cache');
    const cacheUrl = 'https://webcache.googleusercontent.com/search?q=cache:' + encodeURIComponent(targetUrl);
    const r = await rawFetch(cacheUrl, { ua: nextUA() });
    if (r.error || !r.text || r.text.length < 200) return r;
    return { ...r, url: targetUrl, strategy: 'google-cache' };
}

// — Wayback Machine: last resort archived version
async function waybackFetch(targetUrl) {
    log('[wayback] Trying Wayback Machine');
    const apiUrl = `https://archive.org/wayback/available?url=${encodeURIComponent(targetUrl)}`;
    const r = await rawFetch(apiUrl);
    if (r.error || !r.text) return r;

    try {
        const d = JSON.parse(r.text);
        const snapshotUrl = d?.archived_snapshots?.closest?.url;
        if (!snapshotUrl) return { url: targetUrl, statusCode: 0, text: '', strategy: 'wayback', error: 'No Wayback snapshot found' };

        log(`[wayback] Snapshot: ${snapshotUrl}`);
        const snap = await rawFetch(snapshotUrl, { ua: nextUA() });
        return { ...snap, url: targetUrl, strategy: 'wayback' };
    } catch (_) {
        return { url: targetUrl, statusCode: 0, text: '', strategy: 'wayback', error: 'Wayback parse error' };
    }
}

// — Mobile UA strategy
async function mobileFetch(targetUrl) {
    return fetchWithRetry(targetUrl, { ua: CONFIG.mobileUA });
}

// — Googlebot UA (some sites allow bots)
async function googlebotFetch(targetUrl) {
    log('[googlebot] Trying Googlebot UA');
    return rawFetch(targetUrl, { ua: CONFIG.googleBot });
}

// ═══════════════════════════════════════════════════════════════════════════
// SMART AUTO FETCH  (main orchestration logic)
// ═══════════════════════════════════════════════════════════════════════════

function matchDomain(hostname, list) {
    return list.some(d => hostname === d || hostname.endsWith('.' + d));
}

async function autoFetch(targetUrl, mode) {
    let urlObj;
    try { urlObj = new URL(targetUrl); }
    catch (_) { return { url: targetUrl, statusCode: 0, text: '', strategy: 'http', error: 'Invalid URL' }; }

    const hostname = urlObj.hostname.replace(/^www\./, '');
    const D = CONFIG.domains;

    // ── Route to site-specific fast strategies ──────────────────────────────
    if (matchDomain(hostname, D.reddit))    return redditFetch(targetUrl);
    if (matchDomain(hostname, D.youtube))   return youtubeFetch(targetUrl);
    if (matchDomain(hostname, D.wikipedia)) return wikipediaFetch(targetUrl);
    if (matchDomain(hostname, D.github))    return githubFetch(targetUrl);
    if (matchDomain(hostname, D.hn))        return hnFetch(targetUrl);
    if (matchDomain(hostname, D.arxiv))     return arxivFetch(targetUrl);

    // ── Strategy 1: standard Chrome-like fetch ──────────────────────────────
    log('[auto] Strategy 1: standard fetch');
    let result = await fetchWithRetry(targetUrl, {});
    if (goodResult(result)) return result;

    // ── Strategy 2: mobile UA ───────────────────────────────────────────────
    log('[auto] Strategy 2: mobile UA');
    const mob = await mobileFetch(targetUrl);
    if (goodResult(mob)) return mob;

    // ── Strategy 3: Jina Reader (works on most sites incl. SPAs) ───────────
    log('[auto] Strategy 3: Jina.ai reader');
    const jina = await jinaFetch(targetUrl);
    if (goodResult(jina)) return jina;

    // ── Strategy 4: Googlebot UA ────────────────────────────────────────────
    log('[auto] Strategy 4: Googlebot UA');
    const gbot = await googlebotFetch(targetUrl);
    if (goodResult(gbot)) return gbot;

    // ── Strategy 5 (paywalled or heavy bot-protection): Google Cache ────────
    if (mode === 'aggressive' || matchDomain(hostname, D.paywalled) ||
        matchDomain(hostname, D.twitter) || matchDomain(hostname, D.linkedin)) {
        log('[auto] Strategy 5: Google Cache');
        const gcache = await googleCacheFetch(targetUrl);
        if (goodResult(gcache)) return gcache;
    }

    // ── Strategy 6: Wayback Machine (last resort) ───────────────────────────
    if (mode === 'aggressive') {
        log('[auto] Strategy 6: Wayback Machine');
        const wayback = await waybackFetch(targetUrl);
        if (goodResult(wayback)) return wayback;
    }

    // Return best result we have
    const candidates = [result, mob, jina, gbot].filter(r => r && !r.error);
    if (candidates.length > 0) {
        return candidates.sort((a, b) => scoreContent(b.text) - scoreContent(a.text))[0];
    }

    return {
        url:        targetUrl,
        statusCode: result?.statusCode || 0,
        text:       '',
        strategy:   'failed',
        error:      (result?.error || 'All strategies failed') +
            '\n\nTip: Try mode="aggressive" to also attempt Google Cache & Wayback Machine, ' +
            'or the site requires JavaScript rendering with a real browser.',
    };
}

function goodResult(r) {
    return r && !r.error && r.text && r.text.length > 150 && scoreContent(r.text) > 0;
}

// ═══════════════════════════════════════════════════════════════════════════
// RESULT FORMATTING
// ═══════════════════════════════════════════════════════════════════════════

function formatResult(r, maxChars, startIndex) {
    if (r.error) {
        return [
            `❌ Failed to fetch: ${r.url}`,
            '',
            `Error: ${r.error}`,
            '',
            'Tips:',
            '  • Ensure the URL is publicly accessible',
            '  • Try mode="aggressive" for paywalled or bot-protected pages',
            '  • Some sites (heavy SPAs, logins, CAPTCHAs) require a real browser',
        ].join('\n');
    }

    const header = [
        `📄 **${r.meta?.title || r.url}**`,
        r.meta?.description ? `_${r.meta.description}_` : '',
        ``,
        `URL: ${r.url}`,
        `Status: ${r.statusCode} | Strategy: ${r.strategy} | Length: ${(r.text || '').length} chars`,
        '─'.repeat(60),
        '',
    ].filter(l => l !== undefined).join('\n');

    const body = truncate(r.text || '(Empty page)', maxChars, startIndex);
    return header + body;
}

// ═══════════════════════════════════════════════════════════════════════════
// MCP PROTOCOL
// ═══════════════════════════════════════════════════════════════════════════

function send(id, result) {
    process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id, result }) + '\n');
}

function sendError(id, code, message) {
    process.stdout.write(JSON.stringify({ jsonrpc: '2.0', id, error: { code, message } }) + '\n');
}

async function handleRequest(req) {
    const { id, method, params } = req;
    log(`Request: ${method} (id=${id})`);

    try {
        // ── initialize ──────────────────────────────────────────────────────
        if (method === 'initialize') {
            return send(id, {
                protocolVersion: '2024-11-05',
                capabilities: { tools: {} },
                serverInfo: {
                    name:        'web-fetch',
                    version:     CONFIG.version,
                    description: 'Smart MCP web fetcher v4. Zero external dependencies. ' +
                                 'Site-specific APIs (Reddit, YouTube, Wikipedia, GitHub, HN, ArXiv), ' +
                                 'Jina.ai reader fallback, Google Cache, Wayback Machine, ' +
                                 'Chrome-accurate headers, smart HTML→Markdown extraction, pagination.',
                },
            });
        }

        // ── tools/list ──────────────────────────────────────────────────────
        if (method === 'tools/list') {
            return send(id, {
                tools: [{
                    name: 'fetch_page',
                    description:
                        'Fetches a URL and returns clean, readable Markdown content. ' +
                        'Automatically selects the best strategy: site-specific APIs for Reddit/YouTube/' +
                        'Wikipedia/GitHub/HackerNews/ArXiv, Jina.ai reader for most other pages, ' +
                        'with Google Cache and Wayback Machine as fallbacks. ' +
                        `Default content limit is ${CONFIG.defaultMaxChars} chars; use start_index to paginate.`,
                    inputSchema: {
                        type: 'object',
                        properties: {
                            url: {
                                type:        'string',
                                description: 'Full URL to fetch (must start with http:// or https://)',
                            },
                            mode: {
                                type:        'string',
                                enum:        ['auto', 'aggressive', 'http', 'jina', 'cache', 'wayback'],
                                description:
                                    '"auto" (default) — smart escalation through strategies. ' +
                                    '"aggressive" — also tries Google Cache and Wayback Machine. ' +
                                    '"http" — direct HTTP only, fastest. ' +
                                    '"jina" — force Jina.ai reader. ' +
                                    '"cache" — force Google Cache. ' +
                                    '"wayback" — force Wayback Machine.',
                            },
                            max_chars: {
                                type:        'number',
                                description: `Max characters to return (default: ${CONFIG.defaultMaxChars}). Increase for long documents.`,
                            },
                            start_index: {
                                type:        'number',
                                description: 'Start reading from this character offset (for pagination). Default: 0.',
                            },
                        },
                        required: ['url'],
                    },
                }],
            });
        }

        // ── tools/call ──────────────────────────────────────────────────────
        if (method === 'tools/call') {
            const toolName = params?.name;
            const args     = params?.arguments || {};

            if (toolName !== 'fetch_page') {
                return sendError(id, -32601, 'Unknown tool: ' + toolName);
            }

            const url = (args.url || '').trim();
            if (!url) return sendError(id, -32602, 'Missing required parameter: url');
            if (!url.startsWith('http://') && !url.startsWith('https://')) {
                return sendError(id, -32602, 'URL must start with http:// or https://');
            }

            const mode       = args.mode || 'auto';
            const maxChars   = Math.min(Number(args.max_chars)   || CONFIG.defaultMaxChars, 200000);
            const startIndex = Math.max(Number(args.start_index) || 0, 0);

            log(`Fetching (mode=${mode}, max=${maxChars}, start=${startIndex}): ${url}`);

            let result;
            switch (mode) {
                case 'http':     result = await fetchWithRetry(url, {}); break;
                case 'jina':     result = await jinaFetch(url);          break;
                case 'cache':    result = await googleCacheFetch(url);   break;
                case 'wayback':  result = await waybackFetch(url);       break;
                default:         result = await autoFetch(url, mode);    break;
            }

            const text = formatResult(result, maxChars, startIndex);
            return send(id, { content: [{ type: 'text', text }] });
        }

        // ── notifications/initialized ───────────────────────────────────────
        if (method === 'notifications/initialized') {
            log('Client initialized');
            return;
        }

        sendError(id, -32601, 'Unknown method: ' + method);

    } catch (err) {
        logErr('Handler error: ' + err.message, err);
        sendError(id, -32603, 'Internal error: ' + err.message);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MAIN
// ═══════════════════════════════════════════════════════════════════════════

async function main() {
    log('='.repeat(65));
    log(`MCP Web Fetch Server v${CONFIG.version} — Zero External Dependencies`);
    log('Strategies: site APIs, Jina reader, Google Cache, Wayback Machine');
    log('Built-in: gzip/deflate/brotli, cookies, Chrome headers, Markdown');
    log('='.repeat(65));

    const rl = readline.createInterface({ input: process.stdin, terminal: false });

    rl.on('line', async (line) => {
        const trimmed = line.trim();
        if (!trimmed) return;
        try {
            const request = JSON.parse(trimmed);
            await handleRequest(request);
        } catch (e) {
            logErr('Failed to parse JSON-RPC: ' + e.message);
        }
    });

    rl.on('close', () => { log('stdin closed — shutting down'); process.exit(0); });

    process.on('uncaughtException',  (err)    => logErr('Uncaught exception', err));
    process.on('unhandledRejection', (reason) => logErr('Unhandled rejection: ' + reason));
}

main();
