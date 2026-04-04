# AI Link Summarizer (Obsidian Community Plugin)

Right-click a URL in the editor and run **Summarize link**.
The plugin sends the URL to your selected provider (Gemini or OpenAI), then inserts a concise summary into the current note on a new line **before the detected link**.

## Features

- Detects URLs from:
  - selected raw URLs (`https://...`)
  - selected Markdown links (`[label](https://...)`)
  - Markdown/raw link under cursor on right-click
- Adds editor context menu action: **Summarize link**
- Provider support:
  - Gemini via `@google/genai` + `urlContext`
  - OpenAI via `openai` + `web_search_preview`
- Inserts summary before the detected link (optional timestamp prefix)
- Settings:
  - Provider (Gemini/OpenAI)
  - Separate settings blocks for Summary, Gemini, and OpenAI
  - Gemini API key + model
  - OpenAI API key + model
  - Gemini model presets: `gemini-3.1-flash-lite-preview`, `gemini-3.0-flash-preview`
  - OpenAI model presets: `gpt-5.3-chat-latest`, `gpt-5.2`
  - Summary length range (characters), format like `200-600` (minimum `200`)
  - Private-network URL policy toggle (advanced)
  - Request timeout (ms), range `5000-120000`
  - One-click clear stored API keys (and command: `ai-link-summarizer.clear-api-keys`)
  - Custom prompt (optional)
  - Include timestamp
- Notices + clean errors for:
  - no URL found
  - invalid URL
  - blocked private-network URL
  - provider request timeout
  - provider request failure
  - unsupported/unreadable page
  - no active editor

## Build

1. Install dependencies:
   ```bash
   npm install
   ```
2. Build once:
   ```bash
   npm run build
   ```
3. Or watch during development:
   ```bash
   npm run dev
   ```

## Pre-submit Checklist

Run this before tagging a release or updating your Obsidian Community Plugins PR:

```bash
npm run preflight
```

This runs:

- Type check (`npm run check`)
- Obsidian lint guardrails (`npm run lint:obsidian`)
- Production build (`npm run build`)

If your submission PR in `obsidianmd/obsidian-releases` references this plugin, keep the PR body **Repo URL** aligned with the current repository:

- `https://github.com/laroy-sh/obsidian-ai-link-summarizer`

## Install In Obsidian

1. Build the plugin so `main.js` exists.
2. Copy these files to your vault plugin folder:
   - `manifest.json`
   - `main.js`
   - `styles.css` (empty here, but included for compatibility)
3. Target folder:
   - `<your-vault>/.obsidian/plugins/ai-link-summarizer/`

4. In Obsidian:
   - **Settings -> Community plugins**
   - Enable **AI Link Summarizer**
   - Open plugin settings and set provider + API key + model

## How The Context Menu Hook Works

The plugin registers the official workspace event:

```ts
this.registerEvent(
  this.app.workspace.on("editor-menu", (menu, editor) => {
    // ...
  })
);
```

When you right-click in an editor, Obsidian provides the current `menu` and `editor`.
The plugin extracts a URL from the selection or link under cursor. If found, it adds **Summarize link** to that menu.

## Security Notes

- The plugin sends only the target URL and your prompt settings to the selected provider.
- API keys are stored locally in Obsidian plugin data and are not encrypted by this plugin.
- Private-network URLs (`localhost`, `*.local`, and private/link-local IPs) are blocked by default.
- You can clear all stored API keys from plugin settings or via the command **Clear stored API keys**.
- Non-overridable prompt constraints are enforced (single paragraph, full sentence ending, configured length range, no secrets disclosure).

## Known Limitations

- Model output is probabilistic and may be inaccurate or incomplete.
- The plugin inserts plain text only; it does not execute returned code or commands.

## Releases (GitHub Actions)

Pushing a version tag like `0.3.0` triggers a workflow that runs a security audit, builds the plugin, and uploads release assets:

- `main.js`
- `manifest.json`
- `versions.json`
- `styles.css`
- `obsidian-ai-link-summarizer-0.3.0.zip`

To publish:

```bash
git tag 0.3.0
git push origin 0.3.0
```

## Install From GitHub Release

1. Open the GitHub Release for your version tag.
2. Download either:
   - the zip (`obsidian-ai-link-summarizer-0.3.0.zip`), or
   - the individual files (`main.js`, `manifest.json`, `styles.css`, `versions.json`)
3. Extract/copy into:
   - `<your-vault>/.obsidian/plugins/ai-link-summarizer/`
4. Reload/enable the plugin in Obsidian.
