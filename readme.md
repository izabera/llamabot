llamabot is a simple irc bot written in python

it connects to an irc server and a llama.cpp instance, and answers any questions it receives via privmsg

## usage

no third-party dependencies — python 3.9+ standard library only.

```
python3 llamabot.py \
    --server localhost --port 6667 \
    --nick llamabot \
    --channel #somechannel \
    --llama-url http://localhost:4444
```

- `--channel` is optional: without it the bot only answers private messages
- `--model`, `--temperature`, `--max-tokens` are passed through to the llama server
- `--history N` keeps the last N messages of each conversation (default 12)
- `--max-message-length N` splits longer replies into several IRC messages (default 400)
- `-v` for debug logging

the bot answers every privmsg it sees (in its channel or in private message),
keeps a little per-conversation memory, serializes questions through a single
worker (one at a time, a few queued), and reconnects to the irc server after
5 seconds if the connection drops.

## commands

send one of these to the bot (with or without a leading `!` or `/`):

- `reset` — forget the conversation so far
- `help` — remind you of the above

## tests

self-contained, no dependencies:

```
python3 tests/test_mock.py   # fake irc + fake llama http server
python3 tests/test_logic.py  # in-process routing/memory/queue checks
```

## for testing
- a llama.cpp server is listening on localhost:4444
- an irc server is listening on localhost:6667
