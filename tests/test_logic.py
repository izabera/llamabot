#!/usr/bin/env python3
"""In-process logic test for llamabot (no sockets): stubs IRC + llama layers."""
import os
import queue
import sys
import threading
import time
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llamabot as lb

sent = []
llama_requests = []


class FakeIrc:
    def __init__(self, host, port, nickname, password=None):
        self.nickname = nickname
        self.on_privmsg = None

    def connect(self):
        pass

    def join(self, channel):
        sent.append(("join", channel))

    def send(self, target, text):
        sent.append(("privmsg", target, text))

    def quit(self, reason="bye"):
        pass

    def wait_closed(self):
        threading.Event().wait()


class FakeLlama:
    def __init__(self, base_url, model="default", temperature=0.7, max_tokens=300, timeout=300):
        self.base_url = base_url

    def is_ready(self):
        return True

    def chat(self, messages):
        llama_requests.append(messages)
        user = [m["content"] for m in messages if m["role"] == "user"][-1]
        return f"echo: {user}"


lb.IrcClient = FakeIrc
lb.LlamaClient = FakeLlama

args = types.SimpleNamespace(
    server="s", port=1, channel="#chan", nick="llamabot", irc_password=None,
    llama_url="http://x", model="default", temperature=0.0, max_tokens=16,
    llama_timeout=5, system_prompt="sp", history=4, max_message_length=100,
    queue_size=4, verbose=False,
)
bot = lb.Bot(args)
threading.Thread(target=bot.start, daemon=True).start()
time.sleep(0.2)

bot._on_privmsg("alice", "#chan", "what is up")
bot._on_privmsg("carol", "llamabot", "psst")
deadline = time.time() + 5
while time.time() < deadline and len(llama_requests) < 2:
    time.sleep(0.05)
bot._on_privmsg("dave", "#chan", "reset")
bot._on_privmsg("eve", "#chan", "hello")
bot._on_privmsg("frank", "#chan", "help")
bot._on_privmsg("llamabot", "#chan", "ignore me, i am the bot")
time.sleep(0.5)

before = len(sent)
bot.jobs = queue.Queue(maxsize=1)
bot.jobs.put_nowait(("dummy", "dummy", "dummy", "dummy", False))
bot._on_privmsg("grace", "#chan", "overflow test")
time.sleep(0.3)

failures = []
privmsgs = [s for s in sent if s[0] == "privmsg"]
expected = [
    ("#chan", "alice: echo: what is up"),
    ("carol", "echo: psst"),
    ("#chan", "dave: conversation reset."),
    ("#chan", "eve: echo: hello"),
    ("#chan", "ask me anything; say 'reset' to start a fresh conversation."),
    ("#chan", "one sec, still thinking about the last question"),
]
for target, text in expected:
    if (target, text) not in [(t, m) for _, t, m in privmsgs]:
        failures.append(f"missing reply: {target!r} <- {text!r}")
if any("ignore me" in m for _, t, m in privmsgs):
    failures.append("bot answered a message from its own nick")
if not llama_requests:
    failures.append("llama was never called")
else:
    last = llama_requests[-1]
    if last[0] != {"role": "system", "content": "sp"}:
        failures.append("system prompt missing from llama request")
    if [m for m in last if m["role"] == "user"] != [{"role": "user", "content": "hello"}]:
        failures.append(f"reset did not clear channel history: {last!r}")
    joined = llama_requests[0]
    if any(m["content"] == "psst" for m in joined):
        failures.append("PM and channel histories are mixed")

print("--- privmsgs sent ---")
for kind, t, m in privmsgs:
    print(f"{t}: {m}")
if failures:
    print("FAIL")
    for f in failures:
        print(" -", f)
    sys.exit(1)
print("PASS")
