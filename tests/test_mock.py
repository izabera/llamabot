#!/usr/bin/env python3
"""Smoke test: run llamabot against a fake IRC server + fake llama HTTP server."""
import json
import os
import socket
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import llamabot

llama_requests = []


class LlamaHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            body = b"OK"
            code = 200
        else:
            body = b"not found"
            code = 404
        self.send_response(code)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(length))
        llama_requests.append(data)
        user_texts = [m["content"] for m in data["messages"] if m["role"] == "user"]
        last = user_texts[-1]
        time.sleep(0.05)
        content = "lorem " * 40 if "long" in last.lower() else f"echo: {last}"
        body = json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


received = []
join_seen = threading.Event()


def fake_irc(port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    conn, _ = srv.accept()
    buffer = b""
    while True:
        data = conn.recv(4096)
        if not data:
            break
        buffer += data
        while b"\r\n" in buffer:
            line, buffer = buffer.split(b"\r\n", 1)
            line = line.decode()
            received.append(line)
            if line.startswith("USER "):
                conn.sendall(b":irc.example 001 llamabot :Welcome to the test IRC\r\n")
            elif line.startswith("JOIN "):
                join_seen.set()
                time.sleep(0.1)
                conn.sendall(b":alice!alice@x PRIVMSG #chan :long answer please\r\n")
                conn.sendall(b":bob!bob@x PRIVMSG #chan :hi there\r\n")
                conn.sendall(b":carol!carol@x PRIVMSG llamabot :psst, talk to me\r\n")
                conn.sendall(b":dave!dave@x PRIVMSG #chan :reset\r\n")
                time.sleep(1.0)
                conn.sendall(b":eve!eve@x PRIVMSG #chan :hello\r\n")
    srv.close()


def main():
    httpd = HTTPServer(("127.0.0.1", 19444), LlamaHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    threading.Thread(target=fake_irc, args=(19321,), daemon=True).start()

    args = types.SimpleNamespace(
        server="127.0.0.1", port=19321, channel="#chan", nick="llamabot", irc_password=None,
        llama_url="http://127.0.0.1:19444", model="default", temperature=0.0, max_tokens=64,
        llama_timeout=30, system_prompt="test system", history=4, max_message_length=100,
        queue_size=4, verbose=False,
    )
    bot = llamabot.Bot(args)
    threading.Thread(target=bot.start, daemon=True).start()

    assert join_seen.wait(5), "server never saw JOIN"
    expected = [
        "PRIVMSG carol :echo: psst, talk to me",
        "PRIVMSG #chan :bob: echo: hi there",
        "PRIVMSG #chan :dave: conversation reset.",
        "PRIVMSG #chan :eve: echo: hello",
    ]
    deadline = time.time() + 15
    while time.time() < deadline and not all(e in received for e in expected):
        time.sleep(0.1)

    failures = []
    for line in expected:
        if line not in received:
            failures.append(f"missing: {line!r}")
    # only the first chunk carries the "alice: " prefix, so count all chunks
    alice_lines = [l for l in received if l.startswith("PRIVMSG #chan :") and "lorem" in l]
    if len(alice_lines) < 3:
        failures.append(f"expected long reply split into >=3 IRC messages, got {len(alice_lines)}")
    if llama_requests:
        msgs = llama_requests[-1]["messages"]
        if msgs[0] != {"role": "system", "content": "test system"}:
            failures.append("system prompt missing from llama request")
        if any(m["role"] == "user" and m["content"] == "hi there" for m in msgs):
            failures.append("reset did not clear the channel conversation")
        if len(llama_requests) != 4:
            failures.append(f"expected 4 llama requests, got {len(llama_requests)}")

    print("--- bot traffic ---")
    for line in received:
        print(line)
    if failures:
        print("FAIL")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
