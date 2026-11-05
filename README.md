# Home Library Policy Agent

A live voice and camera agent that helps a policyholder add what they own to their home-contents policy. They talk to
the agent and film the room with their camera; the agent measures the room, logs the objects it sees, reads every book
spine, identifies and prices the books, and writes a report with a structured packet (`claim_packet.json`).

Two processes run side by side:

| Process | Folder | Port | What it does |
|---|---|---|---|
| App server | `reference_app/` | 4177 | The web page, the Gemini Live voice agent and its tools, the report |
| Vision worker | `pipeline/` | 4180 | Object detection (Grounding DINO), spine reading (docTR), outlines (SAM), depth (Depth Pro), the books agent, pricing, room and spine measurement |

## What you need

- macOS (Apple silicon recommended) or Linux, with Python **3.12**. On a Mac: `brew install python@3.12`.
- About 10 GB of free disk space: the worker downloads its models (Grounding DINO, SAM, Depth Pro, docTR) on first use.
- Chrome or Firefox, with access to a camera and a microphone.
- The API keys below.

## API keys

Put keys only in the two `.env` files (created in Set up below), never in the code. Both files are ignored by git.

| Key | File | Needed? | Used for | Where to get it | Without it |
|---|---|---|---|---|---|
| `GOOGLE_API_KEY` | `reference_app/.env` | **Required** | The voice agent (Gemini Live) | [Google AI Studio](https://aistudio.google.com/apikey) → Create API key | The app cannot start a voice session |
| `GOOGLE_API_KEY` | `.env` (repository root) | Optional | The books agent's matching step (Gemini Flash), for readings the rules cannot match | Same place; a key in another project gives it its own quota | The worker uses the key from `reference_app/.env` |
| `GOOGLE_BOOKS_API_KEY` | `.env` | Recommended | Catalogue search and eBook prices | [Google Cloud console](https://console.cloud.google.com/apis/library/books.googleapis.com): enable the Books API, then Credentials → Create API key | Search falls back to Open Library (stricter, finds fewer books); no eBook prices |
| `EBAY_CLIENT_ID`, `EBAY_CLIENT_SECRET` | `.env` | Recommended | Replacement (new) and used prices | [developer.ebay.com](https://developer.ebay.com): create an application and use its **Production** keyset | Books are identified but not priced |

Free tiers are enough to try the app, but they run out quickly: the Gemini free tier allows about 20 matching calls a day per
model, and Google Books about 1,000 searches a day per project (a session uses a few hundred). For regular use, enable billing on
the Google project. The quotas are per project, so a second key in the same project adds nothing.

The Gemini key works for both Gemini Live and Gemini Flash; no separate key per model is needed. Exchange rates come from
the European Central Bank and Open Library needs no key.

## Set up

```bash
git clone <this repository> home-library-agent
cd home-library-agent

# 1. Vision worker environment (repository root)
python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

# 2. App server environment
python3.12 -m venv reference_app/.venv
reference_app/.venv/bin/pip install --upgrade pip
reference_app/.venv/bin/pip install -r reference_app/requirements.txt

# 3. Keys: copy the examples and fill them in
cp .env.example .env                              # Google Books, eBay, optional Gemini key for the books agent
cp reference_app/.env.example reference_app/.env  # Gemini key for the voice agent
```

See [API keys](#api-keys) for what goes in each file.

## Run

Start each process in its own terminal, from the repository root.

```bash
# Terminal 1: vision worker (the first start downloads the models and takes a few minutes)
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python -m uvicorn --app-dir . pipeline.vision_service:app --host 127.0.0.1 --port 4180

# Terminal 2: app server
reference_app/.venv/bin/python -m uvicorn --app-dir reference_app live_demo.server:app --host 127.0.0.1 --port 4177
```

Wait until `curl http://127.0.0.1:4180/health` shows `"ocr_ready":true`, then open <http://127.0.0.1:4177>, allow the
camera and microphone, and press **Talk**.

## A session

1. **Country.** The agent asks where you live; prices use that country's currency.
2. **Room size (optional).** Tape a sheet of plain A4 or US Letter paper on each of two walls that meet in a corner,
   stand in the opposite corner, point at each sheet when asked and say "ready".
3. **Items.** Show the room; detected objects are logged, and the agent asks what video cannot show.
4. **Books.** For each bookcase, hold a bank card flat against the books for two seconds (back side out), then film one
   shelf at a time, close and slow. The agent tells you which books it found and asks about any it is unsure of.
5. **Wrap up.** Say you are done; the report opens on the page.

Each session is saved in `sweeps/live-<date>-<time>-<id>/`: the frames, `claim_packet.json`, `report.html` and the
cost and timing of the session (`usage.json`).

## Tests

```bash
.venv/bin/python -m unittest discover -s pipeline/tests -t .
(cd reference_app && .venv/bin/python -m unittest discover -s tests -p "test_*.py")
node reference_app/tests/client-regressions.cjs     # needs Node.js
```

## Troubleshooting

- **"Live connection failed"** after a server restart: reload the page to start a new session.
- **"Live object detection and book reading are not running"**: the worker is not reachable on port 4180, or was busy
  for more than 15 s; check its terminal and `curl http://127.0.0.1:4180/health`.
- **Books found but few identified, or none priced**: a daily quota ran out (Google Books, Gemini or eBay). The worker
  log and `sweeps/<session>/usage.json` show the errors per service.
- **Port already in use**: stop the old process (`lsof -i :4177` or `:4180`) and start again.
