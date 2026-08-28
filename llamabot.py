#!/usr/bin/env python3
"""llamabot — a simple IRC bot that answers privmsgs using a llama.cpp server.

It connects to an IRC server and a llama.cpp instance (llama-server, which
exposes an OpenAI-compatible HTTP API) and answers any question it receives
via privmsg, remembering a little of the recent conversation per channel or
per private peer until it is told to reset.

Only the Python standard library is used.
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Callable, List, Optional, Tuple

log = logging.getLogger("llamabot")

DEFAULT_SYSTEM_PROMPT = (
    "You are LlamaBot, a friendly assistant chatting over IRC. "
    "Keep answers concise, in plain text (no markdown), and get straight to the point."
)


def parse_irc_line(line: str) -> Tuple[Optional[str], str, List[str]]:
    """Split an IRC line into (prefix, command, params).

    The last parameter may be a trailing parameter introduced by ':',
    which may itself contain spaces.
    """
    prefix = None
    if line.startswith(":"):
        prefix, _, line = line[1:].partition(" ")
    command, _, rest = line.partition(" ")
    params: List[str] = []
    if rest:
        if rest.startswith(":"):
            params = [rest[1:]]
        elif " :" in rest:
            head, trailing = rest.split(" :", 1)
            params = (head.split(" ") if head else []) + [trailing]
        else:
            params = rest.split(" ")
    return prefix, command, params


def split_message(text: str, limit: int) -> List[str]:
    """Split text into chunks of at most `limit` chars, preferring word breaks."""
    text = text.strip()
    if not text:
        return ["..."]
    if limit <= 0:
        return [text]
    chunks: List[str] = []
    while len(text) > limit:
        cut = text.rfind(" ", 0, limit)
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    chunks.append(text)
    return chunks


class LlamaClient:
    """Client for a llama.cpp server (llama-server) over HTTP."""

    def __init__(
        self,
        base_url: str,
        model: str = "default",
        temperature: float = 0.7,
        max_tokens: int = 300,
        timeout: int = 300,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def is_ready(self) -> bool:
        try:
            with urllib.request.urlopen(self.base_url + "/health", timeout=3) as response:
                return 200 <= response.status < 300
        except Exception:
            return False

    def chat(self, messages: List[dict]) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False,
            }
        ).encode()
        request = urllib.request.Request(
            self.base_url + "/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode(errors="replace")[:200]
            raise RuntimeError(f"llama server returned HTTP {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"cannot reach llama server at {self.base_url}: {exc.reason}") from exc
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"unexpected response from llama server: {data!r}") from exc
        return (content or "").strip()


class Memory:
    """Per-conversation chat history (one key per channel or private peer)."""

    def __init__(self, system_prompt: str, max_messages: int = 12):
        self.system_prompt = system_prompt
        self.max_messages = max(2, max_messages)
        self._convos: dict = {}
        self._epochs: dict = {}
        self._lock = threading.Lock()

    def epoch_of(self, key: str) -> int:
        with self._lock:
            return self._epochs.get(key, 0)

    def build_messages(self, key: str, text: str, job_epoch: int) -> Tuple[List[dict], bool]:
        """Build the model request for a queued question.

        Returns (messages, remember_allowed). If the conversation was reset
        after the question was queued, the question is answered on its own
        without touching the fresh history.
        """
        with self._lock:
            current = self._epochs.get(key, 0)
            if current != job_epoch:
                return (
                    [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": text},
                    ],
                    False,
                )
            convo = self._convos.setdefault(key, [])
            convo.append({"role": "user", "content": text})
            if len(convo) > self.max_messages:
                del convo[: len(convo) - self.max_messages]
            messages = (
                [{"role": "system", "content": self.system_prompt}]
                + convo[-self.max_messages :]
            )
            return messages, True

    def remember(self, key: str, answer: str, epoch: int) -> None:
        with self._lock:
            if self._epochs.get(key, 0) != epoch:
                return  # conversation was reset while the reply was being generated
            convo = self._convos.setdefault(key, [])
            convo.append({"role": "assistant", "content": answer})
            if len(convo) > self.max_messages:
                del convo[: len(convo) - self.max_messages]

    def reset(self, key: str) -> None:
        with self._lock:
            self._convos.pop(key, None)
            self._epochs[key] = self._epochs.get(key, 0) + 1


class IrcClient:
    """A minimal IRC client: register, optionally join a channel, chat, keep alive."""

    def __init__(self, host: str, port: int, nickname: str, password: Optional[str] = None):
        self.host = host
        self.port = port
        self.nickname = nickname
        self.password = password
        self.on_privmsg: Optional[Callable[[str, str, str], None]] = None
        self._sock: Optional[socket.socket] = None
        self._send_lock = threading.Lock()
        self._registered = threading.Event()
        self._closed = threading.Event()
        self._closed.set()

    def connect(self) -> None:
        self._closed.clear()
        self._registered.clear()
        sock = socket.create_connection((self.host, self.port), timeout=60)
        sock.settimeout(None)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        except OSError:
            pass
        self._sock = sock
        if self.password:
            self._raw(f"PASS {self.password}")
        self._raw(f"NICK {self.nickname}")
        self._raw("USER llamabot 0 * :llama.cpp IRC bot")
        threading.Thread(target=self._read_loop, daemon=True).start()

    def join(self, channel: str) -> None:
        if not self._registered.wait(timeout=60):
            raise TimeoutError("timed out waiting for the server to accept us")
        self._raw(f"JOIN {channel}")

    def send(self, target: str, text: str) -> None:
        self._raw(f"PRIVMSG {target} :{text}")

    def quit(self, reason: str = "bye") -> None:
        self._raw(f"QUIT :{reason}")
        self._close_socket()

    def wait_closed(self) -> None:
        self._closed.wait()

    def _close_socket(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def _raw(self, line: str) -> None:
        sock = self._sock
        if sock is None:
            return
        try:
            with self._send_lock:
                sock.sendall((line + "\r\n").encode("utf-8", "replace"))
        except OSError as exc:
            log.warning("failed to send %r: %s", line, exc)

    def _read_loop(self) -> None:
        sock = self._sock
        buffer = ""
        try:
            while True:
                data = sock.recv(4096)
                if not data:
                    break
                buffer += data.decode("utf-8", "replace")
                while "\r\n" in buffer:
                    line, buffer = buffer.split("\r\n", 1)
                    if line:
                        self._handle_line(line)
        except OSError as exc:
            log.warning("IRC connection lost: %s", exc)
        finally:
            self._close_socket()
            self._closed.set()

    def _handle_line(self, line: str) -> None:
        prefix, command, rest = parse_irc_line(line)
        nick = prefix.split("!", 1)[0] if prefix else ""
        if command == "PING":
            self._raw(f"PONG :{rest[0] if rest else 'llamabot'}")
        elif command == "001":
            self._registered.set()
            log.info("registered on IRC as %s", self.nickname)
        elif command == "PRIVMSG" and len(rest) >= 2:
            if self.on_privmsg:
                self.on_privmsg(nick, rest[0], rest[1])
        elif command == "KICK" and len(rest) >= 2:
            if rest[1].lower() == self.nickname.lower():
                log.warning("kicked from %s; rejoining", rest[0])
                if self._registered.is_set():
                    self._raw(f"JOIN {rest[0]}")
        elif command == "433":
            fallback = f"{self.nickname}_"
            log.warning("nickname %s is taken; trying %s", self.nickname, fallback)
            self.nickname = fallback
            self._raw(f"NICK {fallback}")
        elif command == "ERROR":
            log.error("IRC error: %s", " ".join(rest))


class Bot:
    """Glues the IRC client to the llama server with a small job queue.

    Generation happens in one worker thread so the IRC connection (PING/PONG,
    reconnection) keeps running while the model is thinking.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.llama = LlamaClient(
            args.llama_url, args.model, args.temperature, args.max_tokens, args.llama_timeout
        )
        self.memory = Memory(args.system_prompt, args.history)
        self.jobs: "queue.Queue[Tuple[str, str, str, str, bool, int]]" = queue.Queue(
            maxsize=args.queue_size
        )
        self._irc: Optional[IrcClient] = None

    def start(self) -> None:
        threading.Thread(target=self._worker, daemon=True).start()
        self._wait_for_llama()
        while True:
            irc = IrcClient(
                self.args.server, self.args.port, self.args.nick,
                password=self.args.irc_password,
            )
            irc.on_privmsg = self._on_privmsg
            self._irc = irc
            try:
                irc.connect()
                if self.args.channel:
                    irc.join(self.args.channel)
                log.info(
                    "ready: nick=%s server=%s:%s channel=%s llama=%s",
                    irc.nickname, self.args.server, self.args.port,
                    self.args.channel or "-", self.args.llama_url,
                )
                irc.wait_closed()
            except Exception:
                log.exception("IRC connection failed")
            log.info("reconnecting in 5 seconds...")
            time.sleep(5)

    def shutdown(self) -> None:
        irc = self._irc
        if irc is not None:
            try:
                irc.quit("llamabot shutting down")
            except Exception:
                pass

    def _wait_for_llama(self) -> None:
        if self.llama.is_ready():
            log.info("llama server at %s is up", self.llama.base_url)
            return
        log.warning(
            "llama server at %s is not answering /health yet; "
            "requests will keep retrying until it is",
            self.llama.base_url,
        )

    def _effective_nick(self) -> str:
        return self._irc.nickname if self._irc is not None else self.args.nick

    def _on_privmsg(self, nick: str, target: str, text: str) -> None:
        if not text.strip() or nick.lower() == self._effective_nick().lower():
            return
        is_pm = target.lower() == self._effective_nick().lower()
        reply_to = nick if is_pm else target
        command = text.strip().lower().lstrip("!/")
        if command in ("reset", "clear"):
            self.memory.reset(reply_to.lower())
            ack = "conversation reset." if is_pm else f"{nick}: conversation reset."
            self._send(reply_to, ack)
            return
        if command == "help":
            self._send(reply_to, "ask me anything; say 'reset' to start a fresh conversation.")
            return
        self._enqueue(nick, reply_to, text, is_pm)

    def _enqueue(self, user_nick: str, reply_to: str, text: str, is_pm: bool) -> None:
        key = reply_to.lower()
        try:
            self.jobs.put_nowait((key, reply_to, user_nick, text, is_pm, self.memory.epoch_of(key)))
        except queue.Full:
            self._send(reply_to, "one sec, still thinking about the last question")

    def _worker(self) -> None:
        while True:
            key, target, user_nick, text, is_pm, job_epoch = self.jobs.get()
            log.info("thinking: %s asked %r", user_nick, text[:100])
            started = time.monotonic()
            try:
                messages, remember_allowed = self.memory.build_messages(key, text, job_epoch)
                reply = self.llama.chat(messages)
                if remember_allowed:
                    self.memory.remember(key, reply, job_epoch)
            except Exception:
                log.exception("generation failed")
                reply = "Sorry, I could not get an answer from the model server."
            log.info("answered %s in %.1fs", user_nick, time.monotonic() - started)
            prefix = "" if is_pm else f"{user_nick}: "
            chunks = split_message(reply, self.args.max_message_length - len(prefix))
            for index, chunk in enumerate(chunks):
                self._send(target, (prefix + chunk) if index == 0 else chunk)

    def _send(self, target: str, text: str) -> None:
        if self._irc is not None:
            self._irc.send(target, text)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IRC bot that answers privmsgs using a llama.cpp server."
    )
    irc = parser.add_argument_group("irc")
    irc.add_argument("--server", default="localhost", help="IRC server host (default: localhost)")
    irc.add_argument("--port", type=int, default=6667, help="IRC server port (default: 6667)")
    irc.add_argument(
        "--channel",
        default=None,
        help="channel to join (optional; the bot always answers private messages)",
    )
    irc.add_argument("--nick", default="llamabot", help="bot nickname (default: llamabot)")
    irc.add_argument("--irc-password", default=None, help="IRC server password, if any")
    llama = parser.add_argument_group("llama")
    llama.add_argument(
        "--llama-url",
        default="http://localhost:4444",
        help="base URL of the llama.cpp server (default: http://localhost:4444)",
    )
    llama.add_argument("--model", default="default", help="model name for the llama server")
    llama.add_argument("--temperature", type=float, default=0.7)
    llama.add_argument("--max-tokens", type=int, default=300, help="max tokens per reply (default: 300)")
    llama.add_argument(
        "--llama-timeout", type=int, default=300, help="seconds to wait for one reply (default: 300)"
    )
    chat = parser.add_argument_group("chat")
    chat.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    chat.add_argument("--history", type=int, default=12, help="past messages kept per conversation (default: 12)")
    chat.add_argument(
        "--max-message-length", type=int, default=400,
        help="split replies longer than this into several IRC messages (default: 400)",
    )
    chat.add_argument(
        "--queue-size", type=int, default=3,
        help="queued questions before the bot says it is busy (default: 3)",
    )
    chat.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    bot = Bot(args)
    try:
        bot.start()
    except KeyboardInterrupt:
        log.info("shutting down")
    finally:
        bot.shutdown()


if __name__ == "__main__":
    main()
