import { App, Editor, EditorPosition, MarkdownView, Notice, Plugin, PluginSettingTab, Setting } from "obsidian";
import { GoogleGenAI } from "@google/genai";

interface GeminiLinkSummarizerSettings {
  apiKey: string;
  modelName: string;
  customPrompt: string;
  includeTimestamp: boolean;
}

interface UrlTarget {
  rawUrl: string;
  insertBefore: EditorPosition;
}

const DEFAULT_SETTINGS: GeminiLinkSummarizerSettings = {
  apiKey: "",
  modelName: "gemini-2.5-flash",
  customPrompt: "",
  includeTimestamp: false
};

const DEFAULT_SUMMARY_PROMPT =
  "Use URL Context to read the provided URL, then write one plain-text summary between 200 and 500 characters. Focus on the key point and important details. Do not use bullet points.";

const MENU_TITLE = "Summarize via Gemini";
const UNREADABLE_PAGE_ERROR = "UNREADABLE_PAGE";
const EMPTY_SUMMARY_ERROR = "EMPTY_SUMMARY";

export default class GeminiLinkSummarizerPlugin extends Plugin {
  settings: GeminiLinkSummarizerSettings = DEFAULT_SETTINGS;

  async onload(): Promise<void> {
    await this.loadSettings();
    this.addSettingTab(new GeminiLinkSummarizerSettingTab(this.app, this));

    this.registerEvent(
      this.app.workspace.on("editor-menu", (menu, editor) => {
        const target = this.extractUrlFromEditor(editor);
        if (!target) {
          return;
        }

        menu.addItem((item) => {
          item.setTitle(MENU_TITLE).onClick(async () => {
            await this.handleSummarizeClick(editor, target);
          });
        });
      })
    );
  }

  async loadSettings(): Promise<void> {
    this.settings = Object.assign({}, DEFAULT_SETTINGS, await this.loadData());
  }

  async saveSettings(): Promise<void> {
    await this.saveData(this.settings);
  }

  private async handleSummarizeClick(editorFromMenu: Editor, target: UrlTarget): Promise<void> {
    const activeEditor = this.getActiveEditor();
    if (!activeEditor) {
      new Notice("Gemini Link Summarizer: no active editor.");
      return;
    }

    const editor = editorFromMenu;
    const cleanedUrl = this.cleanExtractedUrl(target.rawUrl);

    if (!cleanedUrl) {
      new Notice("Gemini Link Summarizer: no URL found.");
      return;
    }

    if (!this.isValidHttpUrl(cleanedUrl)) {
      new Notice("Gemini Link Summarizer: invalid URL.");
      return;
    }

    if (!this.settings.apiKey.trim()) {
      new Notice("Gemini Link Summarizer: add your Gemini API key in plugin settings.");
      return;
    }

    try {
      const summary = await this.requestGeminiSummary(cleanedUrl);
      const output = `${this.formatOutput(summary)}\n`;
      editor.replaceRange(output, target.insertBefore);
      new Notice("Gemini Link Summarizer: summary inserted.");
    } catch (error: unknown) {
      new Notice(this.toNoticeMessage(error));
    }
  }

  private getActiveEditor(): Editor | null {
    const view = this.app.workspace.getActiveViewOfType(MarkdownView);
    return view?.editor ?? null;
  }

  private extractUrlFromEditor(editor: Editor): UrlTarget | null {
    const selection = editor.getSelection();
    if (selection.length > 0) {
      const fromSelection = this.extractUrlFromText(selection, editor.getCursor("from"));
      if (fromSelection) {
        return fromSelection;
      }
    }

    const cursor = editor.getCursor();
    const line = editor.getLine(cursor.line);
    return this.extractUrlFromLineAtPosition(line, cursor.line, cursor.ch);
  }

  private extractUrlFromText(text: string, basePos: EditorPosition): UrlTarget | null {
    const markdownLinkMatch = /\[[^\]]*]\(\s*<?(https?:\/\/[^\s)>]+)>?\s*\)/i.exec(text);
    if (markdownLinkMatch?.[1] && markdownLinkMatch.index !== undefined) {
      return {
        rawUrl: markdownLinkMatch[1],
        insertBefore: this.addTextOffset(basePos, text, markdownLinkMatch.index)
      };
    }

    const rawUrlMatch = /https?:\/\/[^\s<>"'`]+/i.exec(text);
    if (rawUrlMatch?.[0] && rawUrlMatch.index !== undefined) {
      return {
        rawUrl: rawUrlMatch[0],
        insertBefore: this.addTextOffset(basePos, text, rawUrlMatch.index)
      };
    }

    return null;
  }

  private extractUrlFromLineAtPosition(line: string, lineNumber: number, cursorCh: number): UrlTarget | null {
    const markdownRegex = /\[[^\]]*]\(\s*<?(https?:\/\/[^\s)>]+)>?\s*\)/gi;
    for (const match of line.matchAll(markdownRegex)) {
      const matchIndex = match.index ?? -1;
      const matchEnd = matchIndex + match[0].length;
      if (matchIndex <= cursorCh && cursorCh <= matchEnd) {
        return match[1]
          ? {
              rawUrl: match[1],
              insertBefore: { line: lineNumber, ch: matchIndex }
            }
          : null;
      }
    }

    const rawUrlRegex = /https?:\/\/[^\s<>"'`]+/gi;
    for (const match of line.matchAll(rawUrlRegex)) {
      const matchIndex = match.index ?? -1;
      const matchEnd = matchIndex + match[0].length;
      if (matchIndex <= cursorCh && cursorCh <= matchEnd) {
        return {
          rawUrl: match[0],
          insertBefore: { line: lineNumber, ch: matchIndex }
        };
      }
    }

    return null;
  }

  private addTextOffset(basePos: EditorPosition, text: string, offset: number): EditorPosition {
    const boundedOffset = Math.max(0, Math.min(offset, text.length));
    const before = text.slice(0, boundedOffset);
    const lines = before.split("\n");

    if (lines.length === 1) {
      return { line: basePos.line, ch: basePos.ch + lines[0].length };
    }

    return {
      line: basePos.line + lines.length - 1,
      ch: lines[lines.length - 1].length
    };
  }

  private cleanExtractedUrl(url: string): string {
    let cleaned = url.trim();
    if (cleaned.startsWith("<") && cleaned.endsWith(">")) {
      cleaned = cleaned.slice(1, -1);
    }

    cleaned = cleaned.replace(/[.,!?;:]+$/g, "");

    while (cleaned.endsWith(")") && !this.hasBalancedParentheses(cleaned)) {
      cleaned = cleaned.slice(0, -1);
    }

    return cleaned;
  }

  private hasBalancedParentheses(value: string): boolean {
    let open = 0;
    for (const char of value) {
      if (char === "(") {
        open += 1;
      } else if (char === ")") {
        if (open === 0) {
          return false;
        }
        open -= 1;
      }
    }

    return open === 0;
  }

  private isValidHttpUrl(value: string): boolean {
    try {
      const parsed = new URL(value);
      return parsed.protocol === "http:" || parsed.protocol === "https:";
    } catch {
      return false;
    }
  }

  private async requestGeminiSummary(url: string): Promise<string> {
    const ai = new GoogleGenAI({ apiKey: this.settings.apiKey.trim() });
    const model = this.settings.modelName.trim() || DEFAULT_SETTINGS.modelName;
    const userPrompt = this.settings.customPrompt.trim() || DEFAULT_SUMMARY_PROMPT;
    const prompt = `${userPrompt}\n\nURL: ${url}`;

    const response = await ai.models.generateContent({
      model,
      contents: prompt,
      config: {
        tools: [{ urlContext: {} }]
      }
    });

    const responseText = this.extractResponseText(response);
    if (!responseText) {
      throw new Error(EMPTY_SUMMARY_ERROR);
    }

    const normalized = responseText.replace(/\s+/g, " ").trim();
    if (!normalized) {
      throw new Error(UNREADABLE_PAGE_ERROR);
    }

    if (normalized.length > 500) {
      return `${normalized.slice(0, 497).trimEnd()}...`;
    }

    return normalized;
  }

  private extractResponseText(response: unknown): string {
    const responseObj = response as Record<string, unknown>;
    const textValue = responseObj.text;

    if (typeof textValue === "string") {
      return textValue.trim();
    }

    const candidates = responseObj.candidates;
    if (!Array.isArray(candidates) || candidates.length === 0) {
      return "";
    }

    const firstCandidate = candidates[0] as Record<string, unknown>;
    const content = firstCandidate.content as Record<string, unknown> | undefined;
    const parts = content?.parts;
    if (!Array.isArray(parts)) {
      return "";
    }

    return parts
      .map((part) => {
        const partObj = part as Record<string, unknown>;
        return typeof partObj.text === "string" ? partObj.text : "";
      })
      .filter((partText) => partText.length > 0)
      .join("\n")
      .trim();
  }

  private formatOutput(summary: string): string {
    if (!this.settings.includeTimestamp) {
      return summary;
    }

    const timestamp = new Date().toLocaleString();
    return `[${timestamp}] ${summary}`;
  }

  private toNoticeMessage(error: unknown): string {
    const message = error instanceof Error ? error.message : String(error);
    const lower = message.toLowerCase();

    if (message === UNREADABLE_PAGE_ERROR || message === EMPTY_SUMMARY_ERROR) {
      return "Gemini Link Summarizer: unsupported or unreadable page.";
    }

    if (lower.includes("api key") || lower.includes("unauth") || lower.includes("permission")) {
      return "Gemini Link Summarizer: Gemini request failed. Check API key and model settings.";
    }

    if (
      lower.includes("unsupported") ||
      lower.includes("unreadable") ||
      lower.includes("cannot fetch") ||
      lower.includes("unable to fetch") ||
      lower.includes("url context")
    ) {
      return "Gemini Link Summarizer: unsupported or unreadable page.";
    }

    return `Gemini Link Summarizer: Gemini request failure (${message}).`;
  }
}

class GeminiLinkSummarizerSettingTab extends PluginSettingTab {
  plugin: GeminiLinkSummarizerPlugin;

  constructor(app: App, plugin: GeminiLinkSummarizerPlugin) {
    super(app, plugin);
    this.plugin = plugin;
  }

  display(): void {
    const { containerEl } = this;
    containerEl.empty();
    containerEl.createEl("h2", { text: "Gemini Link Summarizer" });

    new Setting(containerEl)
      .setName("Gemini API key")
      .setDesc("API key used for Gemini requests.")
      .addText((text) =>
        text
          .setPlaceholder("AIza...")
          .setValue(this.plugin.settings.apiKey)
          .onChange(async (value) => {
            this.plugin.settings.apiKey = value.trim();
            await this.plugin.saveSettings();
          })
      );

    new Setting(containerEl)
      .setName("Model name")
      .setDesc("Gemini model to use.")
      .addText((text) =>
        text
          .setPlaceholder("gemini-2.5-flash")
          .setValue(this.plugin.settings.modelName)
          .onChange(async (value) => {
            this.plugin.settings.modelName = value.trim() || DEFAULT_SETTINGS.modelName;
            await this.plugin.saveSettings();
          })
      );

    new Setting(containerEl)
      .setName("Custom prompt (optional)")
      .setDesc("Overrides the default summarization prompt.")
      .addTextArea((textArea) =>
        textArea.setValue(this.plugin.settings.customPrompt).onChange(async (value) => {
          this.plugin.settings.customPrompt = value;
          await this.plugin.saveSettings();
        })
      );

    new Setting(containerEl)
      .setName("Include timestamp")
      .setDesc("Prepends current timestamp before inserted summary.")
      .addToggle((toggle) =>
        toggle.setValue(this.plugin.settings.includeTimestamp).onChange(async (value) => {
          this.plugin.settings.includeTimestamp = value;
          await this.plugin.saveSettings();
        })
      );
  }
}
