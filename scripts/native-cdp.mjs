// Minimal Chrome DevTools Protocol client for the Tauri WebView2 window.
// Launch the app first with WEBVIEW2_ADDITIONAL_BROWSER_ARGUMENTS=--remote-debugging-port=9222
// (see scripts/run-native-debug.bat). Node 21+ (global WebSocket/fetch) required.
//
// usage:
//   node scripts/native-cdp.mjs info
//   node scripts/native-cdp.mjs screenshot [out.png]
//   node scripts/native-cdp.mjs console [seconds]
//   node scripts/native-cdp.mjs eval "<js expression>"
//   node scripts/native-cdp.mjs click "<css selector>"
//   node scripts/native-cdp.mjs text  "<css selector>"
import { writeFileSync } from 'node:fs';

const PORT = process.env.CDP_PORT || '9222';
const HOST = `http://127.0.0.1:${PORT}`;

async function listTargets() {
  const res = await fetch(`${HOST}/json`);
  return res.json();
}

async function pickPage() {
  const targets = await listTargets();
  const pages = targets.filter((t) => t.type === 'page' && t.webSocketDebuggerUrl);
  const page = pages.find((t) => /^https?:/.test(t.url)) || pages[0];
  if (!page) {
    throw new Error(`No page target on :${PORT}. Is the app running with remote debugging?\n${JSON.stringify(targets, null, 2)}`);
  }
  return page;
}

function connect(wsUrl) {
  return new Promise((resolve, reject) => {
    const ws = new WebSocket(wsUrl);
    let id = 0;
    const pending = new Map();
    const listeners = [];
    const api = {
      send(method, params = {}) {
        return new Promise((res, rej) => {
          const mid = ++id;
          pending.set(mid, { res, rej });
          ws.send(JSON.stringify({ id: mid, method, params }));
        });
      },
      on(fn) { listeners.push(fn); },
      close() { ws.close(); },
    };
    ws.addEventListener('open', () => resolve(api));
    ws.addEventListener('error', (e) => reject(new Error('ws error: ' + (e.message || e.type))));
    ws.addEventListener('message', (ev) => {
      const msg = JSON.parse(ev.data);
      if (msg.id != null && pending.has(msg.id)) {
        const { res, rej } = pending.get(msg.id);
        pending.delete(msg.id);
        msg.error ? rej(new Error(JSON.stringify(msg.error))) : res(msg.result);
      } else if (msg.method) {
        for (const l of listeners) l(msg);
      }
    });
  });
}

const [cmd, arg] = process.argv.slice(2);

async function main() {
  if (cmd === 'info') {
    console.log(JSON.stringify(await listTargets(), null, 2));
    return;
  }
  const page = await pickPage();
  const c = await connect(page.webSocketDebuggerUrl);

  if (cmd === 'screenshot') {
    await c.send('Page.enable');
    const { data } = await c.send('Page.captureScreenshot', { format: 'png' });
    const out = arg || 'webview.png';
    writeFileSync(out, Buffer.from(data, 'base64'));
    console.log('saved ' + out);
  } else if (cmd === 'console') {
    await c.send('Runtime.enable');
    await c.send('Log.enable');
    const lines = [];
    c.on((m) => {
      if (m.method === 'Runtime.consoleAPICalled') {
        lines.push(`[${m.params.type}] ` + m.params.args.map((a) => a.value ?? a.description ?? '').join(' '));
      } else if (m.method === 'Log.entryAdded') {
        lines.push(`[${m.params.entry.level}] ${m.params.entry.text}`);
      } else if (m.method === 'Runtime.exceptionThrown') {
        const d = m.params.exceptionDetails;
        lines.push(`[exception] ${d.exception?.description || d.text}`);
      }
    });
    const secs = Number(arg) || 3;
    await new Promise((r) => setTimeout(r, secs * 1000));
    console.log(lines.join('\n') || `(no console output in ${secs}s)`);
  } else if (cmd === 'reloadwatch') {
    await c.send('Runtime.enable');
    await c.send('Log.enable');
    await c.send('Page.enable');
    const lines = [];
    c.on((m) => {
      if (m.method === 'Runtime.consoleAPICalled') {
        lines.push(`[${m.params.type}] ` + m.params.args.map((a) => a.value ?? a.description ?? '').join(' '));
      } else if (m.method === 'Log.entryAdded') {
        const e = m.params.entry;
        lines.push(`[${e.level}] ${e.text}` + (e.url ? ` (${e.url})` : ''));
      } else if (m.method === 'Runtime.exceptionThrown') {
        const d = m.params.exceptionDetails;
        lines.push(`[EXCEPTION] ${d.exception?.description || d.text}` + (d.url ? ` @${d.url}:${d.lineNumber}` : ''));
      }
    });
    await c.send('Page.reload', { ignoreCache: true });
    const secs = Number(arg) || 8;
    await new Promise((r) => setTimeout(r, secs * 1000));
    console.log(lines.join('\n') || `(no events in ${secs}s after reload)`);
  } else if (cmd === 'eval') {
    const { result, exceptionDetails } = await c.send('Runtime.evaluate', {
      expression: arg, returnByValue: true, awaitPromise: true,
    });
    console.log(exceptionDetails
      ? 'EXCEPTION: ' + (exceptionDetails.exception?.description || exceptionDetails.text)
      : JSON.stringify(result.value ?? result.description ?? null));
  } else if (cmd === 'text') {
    const expr = `document.querySelector(${JSON.stringify(arg)})?.innerText ?? null`;
    const { result } = await c.send('Runtime.evaluate', { expression: expr, returnByValue: true });
    console.log(result.value ?? '(selector not found)');
  } else if (cmd === 'click') {
    const expr = `(() => { const el = document.querySelector(${JSON.stringify(arg)}); if(!el) return null; el.scrollIntoView({block:'center'}); const r = el.getBoundingClientRect(); return {x: r.left + r.width/2, y: r.top + r.height/2}; })()`;
    const { result } = await c.send('Runtime.evaluate', { expression: expr, returnByValue: true });
    if (!result.value) { console.log('selector not found: ' + arg); }
    else {
      const { x, y } = result.value;
      await c.send('Input.dispatchMouseEvent', { type: 'mouseMoved', x, y });
      await c.send('Input.dispatchMouseEvent', { type: 'mousePressed', x, y, button: 'left', clickCount: 1 });
      await c.send('Input.dispatchMouseEvent', { type: 'mouseReleased', x, y, button: 'left', clickCount: 1 });
      console.log(`clicked ${arg} at ${Math.round(x)},${Math.round(y)}`);
    }
  } else {
    console.log('usage: native-cdp.mjs <info|screenshot [out.png]|console [secs]|eval "<js>"|text "<sel>"|click "<sel>">');
  }
  c.close();
}

main().then(() => process.exit(0)).catch((e) => { console.error(e.message); process.exit(1); });
