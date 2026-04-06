"""Web-based agent monitor with SSE streaming.

Provides an HTTP server that serves a browser dashboard for
interacting with the coding agent.  Uses Server-Sent Events (SSE)
to stream agent output in real time.

Usage::

    local-cli --web-monitor                    # default port 7070
    local-cli --web-monitor --web-port 8080    # custom port
"""

import http.server
import json
import os
import queue
import threading
from typing import Any

from local_cli.config import Config
from local_cli.model_presets import SUPPORTS_THINKING, get_model_family, get_model_preset
from local_cli.providers import LLMProvider, ProviderConnectionError, ProviderRequestError, ProviderStreamError
from local_cli.tools import get_default_tools
from local_cli.tools.base import Tool


def _build_system_prompt(tools: list[Tool]) -> str:
    tool_lines = [f"- {t.name}: {t.description}" for t in tools]
    cwd = os.getcwd()
    return (
        "You are a coding agent. Complete tasks by using tools.\n\n"
        f"WORKING DIRECTORY: {cwd}\n\n"
        "AVAILABLE TOOLS:\n" + "\n".join(tool_lines) + "\n\n"
        "RULES:\n"
        "1. ALWAYS use tools. Never output code as markdown.\n"
        "2. Use write to create files, bash to run them.\n"
        "3. If code has errors, fix by rewriting and re-running.\n"
        "4. Read files before editing.\n"
        "5. Verify changes by reading back or running tests.\n"
    )


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------

_HTML = r"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<title>Bonsai Agent Monitor — local-cli</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:'Menlo','Monaco','Courier New',monospace;font-size:13px}
.hdr{background:#161b22;border-bottom:1px solid #30363d;padding:12px 20px;display:flex;align-items:center;gap:12px}
.hdr h1{font-size:16px;color:#58a6ff}
.hdr .badge{background:#238636;color:#fff;padding:2px 8px;border-radius:12px;font-size:10px}
.hdr .info{color:#8b949e;font-size:11px;margin-left:auto}
.wrap{display:flex;height:calc(100vh - 48px)}
.chat{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px}
.side{width:280px;background:#161b22;border-left:1px solid #30363d;overflow-y:auto;padding:14px}
.side h3{color:#8b949e;font-size:11px;text-transform:uppercase;margin:12px 0 6px}
.msg{animation:fi .25s}
@keyframes fi{from{opacity:0;transform:translateY(6px)}to{opacity:1}}
.m-user{color:#58a6ff;font-weight:bold;margin-bottom:2px;font-size:11px}
.m-ai{color:#a5d6ff;font-weight:bold;margin-bottom:2px;font-size:11px}
.m-tool{color:#f0883e;font-weight:bold;margin-bottom:2px;font-size:11px}
.m-res{color:#56d364;font-weight:bold;margin-bottom:2px;font-size:11px}
.bubble{white-space:pre-wrap;line-height:1.5;padding:8px 12px;border-radius:6px;max-width:100%}
.b-user{background:#1c2333;border-left:3px solid #58a6ff}
.b-ai{background:#13171e}
.b-tool{background:#1c2333;border-left:3px solid #f0883e}
.b-res{background:#0d1f0d;border-left:3px solid #238636}
.b-err{background:#2d0f0f;border-left:3px solid #f85149}
.code{background:#1a1e24;border:1px solid #30363d;border-radius:6px;padding:10px;margin:6px 0;overflow-x:auto;color:#abb2bf;font-size:12px}
.inp{background:#161b22;border-top:1px solid #30363d;padding:12px 20px;display:flex;gap:10px}
.inp input{flex:1;background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:8px 14px;border-radius:6px;font-family:inherit;font-size:13px;outline:none}
.inp input:focus{border-color:#58a6ff}
.inp button{background:#238636;color:#fff;border:none;padding:8px 20px;border-radius:6px;cursor:pointer;font-family:inherit;font-weight:bold;font-size:13px}
.inp button:hover{background:#2ea043}
.inp button:disabled{background:#30363d;cursor:not-allowed}
.spin{display:inline-block;width:10px;height:10px;border:2px solid #30363d;border-top-color:#58a6ff;border-radius:50%;animation:sp .7s linear infinite;margin-left:6px}
@keyframes sp{to{transform:rotate(360deg)}}
.cursor{display:inline-block;width:7px;height:14px;background:#58a6ff;animation:bk 1s step-end infinite;vertical-align:text-bottom}
@keyframes bk{50%{opacity:0}}
.sr{display:flex;justify-content:space-between;padding:3px 0;font-size:11px}
.sl{color:#8b949e}.sv{color:#58a6ff}
.fl{font-size:11px}.fi{padding:3px 0;color:#8b949e;border-bottom:1px solid #21262d}.fi .fn{color:#58a6ff}
</style>
</head>
<body>
<div class="hdr">
  <h1 id="title">Agent Monitor</h1>
  <span class="badge" id="model-badge">-</span>
  <button onclick="clearSession()" style="background:#30363d;color:#c9d1d9;border:none;padding:4px 12px;border-radius:4px;cursor:pointer;font-size:11px">Clear</button>
  <span class="info" id="status">Connecting...</span>
</div>
<div class="wrap">
  <div class="chat" id="chat"></div>
  <div class="side">
    <h3>Files</h3>
    <div class="fl" id="files">No files yet</div>
    <h3>Stats</h3>
    <div class="sr"><span class="sl">Turns</span><span class="sv" id="s-turns">0</span></div>
    <div class="sr"><span class="sl">Tool Calls</span><span class="sv" id="s-tools">0</span></div>
    <div class="sr"><span class="sl">Tokens</span><span class="sv" id="s-tokens">0</span></div>
    <div class="sr"><span class="sl">Speed</span><span class="sv" id="s-speed">-</span></div>
    <h3>Config</h3>
    <div class="sr"><span class="sl">Provider</span><span class="sv" id="s-provider">-</span></div>
    <div class="sr"><span class="sl">Model</span><span class="sv" id="s-model">-</span></div>
    <div class="sr"><span class="sl">Backend</span><span class="sv" id="s-backend">-</span></div>
  </div>
</div>
<div class="inp">
  <input type="text" id="input" placeholder="タスクを入力..." autofocus>
  <button id="btn" onclick="send()">Run</button>
</div>
<script>
const C=document.getElementById('chat');
let turns=0,tcs=0,toks=0,composing=false;
const files=new Set();
const I=document.getElementById('input');
I.addEventListener('compositionstart',()=>{composing=true});
I.addEventListener('compositionend',()=>{composing=false});
I.addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey&&!composing){e.preventDefault();send()}});

function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}

function addMsg(role,txt,cls){
  const d=document.createElement('div');d.className='msg';
  const labels={user:'YOU',assistant:'BONSAI',tool:'TOOL',result:'RESULT',error:'ERROR'};
  const rcls={user:'m-user',assistant:'m-ai',tool:'m-tool',result:'m-res',error:'m-tool'};
  const bcls={user:'b-user',assistant:'b-ai',tool:'b-tool',result:'b-res',error:'b-err'};
  d.innerHTML='<div class="'+(rcls[role]||'m-ai')+'">'+labels[role]+'</div><div class="bubble '+(cls||bcls[role]||'b-ai')+'" id="m'+Date.now()+'">'+esc(txt)+'</div>';
  C.appendChild(d);C.scrollTop=C.scrollHeight;
  return d.querySelector('.bubble');
}

function stat(){
  document.getElementById('s-turns').textContent=turns;
  document.getElementById('s-tools').textContent=tcs;
  document.getElementById('s-tokens').textContent=toks;
}

function updFiles(){
  const el=document.getElementById('files');
  if(files.size===0){el.textContent='No files yet';return}
  el.innerHTML=[...files].map(f=>'<div class="fi"><span class="fn">'+esc(f)+'</span></div>').join('');
}

async function send(){
  const task=I.value.trim();if(!task)return;
  I.value='';
  const btn=document.getElementById('btn');
  btn.disabled=true;btn.innerHTML='Running<span class="spin"></span>';
  document.getElementById('status').textContent='Processing...';
  addMsg('user',task);
  try{
    const r=await fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({task})});
    const reader=r.body.getReader();const dec=new TextDecoder();
    let buf='',el=null;
    while(true){
      const{done,value}=await reader.read();if(done)break;
      buf+=dec.decode(value,{stream:true});
      const lines=buf.split('\n');buf=lines.pop();
      for(const ln of lines){
        if(!ln.startsWith('data: '))continue;
        let d;try{d=JSON.parse(ln.slice(6))}catch{continue}
        if(d.type==='stream_start'){el=addMsg('assistant','');turns++;stat()}
        else if(d.type==='token'){if(el)el.textContent+=d.content;C.scrollTop=C.scrollHeight}
        else if(d.type==='tool_call'){
          tcs++;stat();
          if(d.name==='write'&&d.args&&d.args.file_path){
            files.add(d.args.file_path);updFiles();
            addMsg('tool','write → '+d.args.file_path);
            const cd=document.createElement('div');cd.className='code';cd.textContent=d.args.content||'';
            C.lastElementChild.appendChild(cd);
          }else{
            const a=typeof d.args==='object'?JSON.stringify(d.args):'';
            addMsg('tool',d.name+'('+(a.length>150?a.slice(0,150)+'...':a)+')');
          }
        }
        else if(d.type==='tool_result'){addMsg('result',d.output||'',d.error?'b-err':'b-res')}
        else if(d.type==='stats'){
          if(d.gen_speed)document.getElementById('s-speed').textContent=d.gen_speed+' t/s';
          if(d.tokens)toks+=d.tokens;stat();
        }
        else if(d.type==='done'){
          let info='Done';
          if(d.avg_speed)info+=' | '+d.avg_speed+' t/s';
          if(d.total_tokens)info+=' | '+d.total_tokens+' tokens';
          if(d.elapsed)info+=' | '+d.elapsed+'s';
          document.getElementById('status').textContent=info;
        }
        else if(d.type==='config'){
          document.getElementById('s-provider').textContent=d.provider||'-';
          document.getElementById('s-model').textContent=d.model||'-';
          document.getElementById('s-backend').textContent=d.backend||'-';
          document.getElementById('model-badge').textContent=d.model||'-';
          document.getElementById('title').textContent=(d.provider==='llama-server'?'Bonsai':'local-cli')+' Agent Monitor';
        }
        C.scrollTop=C.scrollHeight;
      }
    }
  }catch(e){addMsg('error','Connection error: '+e.message)}
  btn.disabled=false;btn.textContent='Run';
}
function clearSession(){
  fetch('/api/clear',{method:'POST'}).then(()=>{
    C.innerHTML='';turns=0;tcs=0;toks=0;files.clear();stat();updFiles();
    document.getElementById('s-speed').textContent='-';
    document.getElementById('status').textContent='Session cleared';
  });
}
// Fetch initial config
fetch('/api/config').then(r=>r.json()).then(d=>{
  document.getElementById('s-provider').textContent=d.provider||'-';
  document.getElementById('s-model').textContent=d.model||'-';
  document.getElementById('s-backend').textContent=d.backend||'-';
  document.getElementById('model-badge').textContent=d.model||'-';
  document.getElementById('title').textContent=(d.provider==='llama-server'?'Bonsai':'local-cli')+' Agent Monitor';
  document.getElementById('status').textContent='Ready';
}).catch(()=>{document.getElementById('status').textContent='Ready'});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Agent runner (background thread)
# ---------------------------------------------------------------------------

def _run_agent(
    provider: LLMProvider,
    config: Config,
    tools: list[Tool],
    tool_defs: list[dict[str, Any]],
    tool_map: dict[str, Tool],
    task: str,
    eq: queue.Queue,
    messages: list[dict[str, Any]],
) -> None:
    """Run the agent loop, pushing SSE events to *eq*.

    *messages* is the shared conversation history (mutated in place).
    """
    messages.append({"role": "user", "content": task})

    # Build inference options.
    inference_options: dict[str, Any] = {"num_ctx": config.num_ctx}
    if config.temperature is not None:
        inference_options["temperature"] = config.temperature

    chat_kwargs: dict[str, Any] = {
        "model": config.model,
        "messages": messages,
        "tools": tool_defs,
    }
    if provider.name == "ollama":
        chat_kwargs["options"] = inference_options
        family = get_model_family(config.model)
        if config.think_mode and family in SUPPORTS_THINKING:
            chat_kwargs["think"] = True
        if config.keep_alive is not None:
            chat_kwargs["keep_alive"] = config.keep_alive

    import time as _time

    total_gen_tokens = 0
    total_gen_time = 0.0

    for _ in range(15):
        eq.put(json.dumps({"type": "stream_start"}))

        text_buf = ""
        tool_calls: list[dict[str, Any]] = []
        gen_start = _time.monotonic()
        chunk_count = 0

        try:
            for chunk in provider.chat_stream(**chat_kwargs):
                msg = chunk.get("message", {})
                delta = msg.get("content", "")
                if delta:
                    text_buf += delta
                    chunk_count += 1
                    eq.put(json.dumps({"type": "token", "content": delta}))
                if msg.get("tool_calls"):
                    tool_calls = msg["tool_calls"]
                if chunk.get("done"):
                    break
        except (ProviderConnectionError, ProviderRequestError, ProviderStreamError) as exc:
            eq.put(json.dumps({"type": "tool_result", "output": f"Error: {exc}", "error": True}))
            break
        except Exception as exc:
            eq.put(json.dumps({"type": "tool_result", "output": f"Error: {exc}", "error": True}))
            break

        gen_elapsed = _time.monotonic() - gen_start
        total_gen_tokens += chunk_count
        total_gen_time += gen_elapsed
        if gen_elapsed > 0 and chunk_count > 0:
            speed = chunk_count / gen_elapsed
            eq.put(json.dumps({"type": "stats", "gen_speed": round(speed, 1), "tokens": chunk_count}))

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": text_buf}
        if tool_calls:
            assistant_msg["tool_calls"] = tool_calls
        messages.append(assistant_msg)

        if not tool_calls:
            break

        for tc in tool_calls:
            func = tc.get("function", {})
            tool_name = func.get("name", "")
            tool_args = func.get("arguments", {})
            if isinstance(tool_args, str):
                try:
                    tool_args = json.loads(tool_args)
                except (json.JSONDecodeError, ValueError):
                    tool_args = {}
            tool_call_id = tc.get("id")

            eq.put(json.dumps({"type": "tool_call", "name": tool_name, "args": tool_args}))

            tool = tool_map.get(tool_name)
            if tool is None:
                output = f"Error: unknown tool '{tool_name}'"
                is_error = True
            else:
                try:
                    output = tool.execute(**tool_args)
                    is_error = False
                except Exception as exc:
                    output = f"Error: {exc}"
                    is_error = True

            if len(output) > 5000:
                output = output[:5000] + "\n...(truncated)"

            eq.put(json.dumps({"type": "tool_result", "output": output, "error": is_error}))

            tool_msg: dict[str, Any] = {"role": "tool", "content": output}
            if tool_call_id:
                tool_msg["tool_call_id"] = tool_call_id
            messages.append(tool_msg)

    avg_speed = total_gen_tokens / total_gen_time if total_gen_time > 0 else 0
    eq.put(json.dumps({
        "type": "done",
        "total_tokens": total_gen_tokens,
        "avg_speed": round(avg_speed, 1),
        "elapsed": round(total_gen_time, 2),
    }))
    eq.put(None)  # sentinel


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class _Handler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for the web monitor."""

    # Set by run_web_monitor before starting the server.
    provider: LLMProvider
    config: Config
    tools: list[Tool]
    tool_defs: list[dict[str, Any]]
    tool_map: dict[str, Tool]
    event_queue: queue.Queue
    session_messages: list[dict[str, Any]]  # persistent conversation

    def log_message(self, format: str, *args: Any) -> None:
        pass  # silence request logs

    def do_GET(self) -> None:
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(_HTML.encode("utf-8"))
        elif self.path == "/api/config":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            info = {
                "provider": self.config.provider,
                "model": self.config.model,
                "backend": getattr(self.config, "llama_server_url", self.config.ollama_host),
            }
            self.wfile.write(json.dumps(info).encode("utf-8"))
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/api/run":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            task = body.get("task", "")

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            # Send config info.
            config_event = json.dumps({
                "type": "config",
                "provider": self.config.provider,
                "model": self.config.model,
                "backend": getattr(self.config, "llama_server_url", self.config.ollama_host),
            })
            self.wfile.write(f"data: {config_event}\n\n".encode())
            self.wfile.flush()

            # Clear queue.
            while not self.event_queue.empty():
                try:
                    self.event_queue.get_nowait()
                except queue.Empty:
                    break

            t = threading.Thread(
                target=_run_agent,
                args=(
                    self.provider, self.config,
                    self.tools, self.tool_defs, self.tool_map,
                    task, self.event_queue,
                    self.session_messages,
                ),
                daemon=True,
            )
            t.start()

            while True:
                item = self.event_queue.get()
                if item is None:
                    break
                self.wfile.write(f"data: {item}\n\n".encode())
                self.wfile.flush()
        elif self.path == "/api/clear":
            # Reset conversation but keep system prompt.
            self.session_messages.clear()
            self.session_messages.append({
                "role": "system",
                "content": _build_system_prompt(self.tools),
            })
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"ok":true}')
        else:
            self.send_error(404)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_web_monitor(config: Config | None = None, port: int = 7070) -> None:
    """Start the web monitor HTTP server.

    Args:
        config: Application config. If ``None``, a default is created.
        port: HTTP port to listen on.
    """
    if config is None:
        config = Config()

    # Build provider based on config.
    from local_cli.providers import get_provider

    if config.provider == "llama-server":
        provider = get_provider("llama-server", base_url=config.llama_server_url)
    elif config.provider == "claude":
        provider = get_provider("claude")
    else:
        provider = get_provider("ollama", base_url=config.ollama_host)

    tools = get_default_tools()
    tool_defs = provider.format_tools(tools)
    tool_map = {t.name: t for t in tools}

    # Inject shared state into handler class.
    _Handler.provider = provider
    _Handler.config = config
    _Handler.tools = tools
    _Handler.tool_defs = tool_defs
    _Handler.tool_map = tool_map
    _Handler.event_queue = queue.Queue()
    _Handler.session_messages = [
        {"role": "system", "content": _build_system_prompt(tools)},
    ]

    server = http.server.HTTPServer(("0.0.0.0", port), _Handler)

    print(f"\n  local-cli Web Monitor")
    print(f"  http://localhost:{port}")
    print(f"  Provider: {config.provider} | Model: {config.model}")
    if config.provider == "llama-server":
        print(f"  Backend:  {config.llama_server_url}")
    else:
        print(f"  Backend:  {config.ollama_host}")
    print()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutdown.")
