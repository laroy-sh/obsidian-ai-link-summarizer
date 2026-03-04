# Gemini Link Summarizer (Obsidian Community Plugin)

Right-click a URL in the editor and run **Summarize via Gemini**.  
The plugin sends the URL to Gemini using **URL Context**, then inserts a concise summary into the current note at the cursor.

## Features

- Detects URLs from:
  - selected raw URLs (`https://...`)
  - selected Markdown links (`[label](https://...)`)
  - Markdown/raw link under cursor on right-click
- Adds editor context menu action: **Summarize via Gemini**
- Uses official Google GenAI SDK (`@google/genai`) with URL Context (`tools: [{ urlContext: {} }]`)
- Inserts summary into note (optional timestamp prefix)
- Settings:
  - Gemini API key
  - Model name
  - Custom prompt (optional)
  - Include timestamp
- Notices + clean errors for:
  - no URL found
  - invalid URL
  - Gemini request failure
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

## Install In Obsidian

1. Build the plugin so `main.js` exists.
2. Copy these files to your vault plugin folder:
   - `manifest.json`
   - `main.js`
   - `styles.css` (not used here, optional)
3. Target folder:
   - `<your-vault>/.obsidian/plugins/gemini-link-summarizer/`
4. In Obsidian:
   - **Settings → Community plugins**
   - Enable **Gemini Link Summarizer**
   - Open plugin settings and set your Gemini API key/model

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
The plugin extracts a URL from the selection or link under cursor. If found, it adds **Summarize via Gemini** to that menu.
