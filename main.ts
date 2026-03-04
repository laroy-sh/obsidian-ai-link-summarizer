import { App, Editor, EditorPosition, MarkdownView, Notice, Plugin, PluginSettingTab, Setting } from "obsidian";
import { GoogleGenAI } from "@google/genai";

interface GeminiLinkSummarizerSettings {
  apiKey: string;
  modelName: string;
  customPrompt: string;
  includeTimestamp: boolean;
  summaryLengthChars: number;
}

interface UrlTarget {
  rawUrl: string;
  insertBefore: EditorPosition;
}

const DEFAULT_SETTINGS: GeminiLinkSummarizerSettings = {
  apiKey: "",
  modelName: "gemini-2.5-flash",
  customPrompt: "",
  includeTimestamp: false,
  summaryLengthChars: 300
};

const MENU_TITLE = "Summarize via Gemini";
const NOTICE_PREFIX = "Gemini link summarizer";
const UNREADABLE_PAGE_ERROR = "UNREADABLE_PAGE";
const EMPTY_SUMMARY_ERROR = "EMPTY_SUMMARY";
const MIN_SUMMARY_LENGTH_CHARS = 120;
const MAX_SUMMARY_LENGTH_CHARS = 1200;
const FLASH_MODEL_PRESETS = ["gemini-3.1-flash-lite-preview", "gemini-3.0-flash-preview"] as const;

function clampSummaryLengthChars(value: number): number {
  if (!Number.isFinite(value)) {
    return DEFAULT_SETTINGS.summaryLengthChars;
  }

  return Math.min(MAX_SUMMARY_LENGTH_CHARS, Math.max(MIN_SUMMARY_LENGTH_CHARS, Math.round(value)));
}

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
    this.settings.summaryLengthChars = clampSummaryLengthChars(this.settings.summaryLengthChars);
    this.settings.modelName = this.settings.modelName.trim() || DEFAULT_SETTINGS.modelName;
  }

  async saveSettings(): Promise<void> {
    await this.saveData(this.settings);
  }

  private async handleSummarizeClick(editorFromMenu: Editor, target: UrlTarget): Promise<void> {
    const activeEditor = this.getActiveEditor();
    if (!activeEditor) {
      new Notice(`${NOTICE_PREFIX}: no active editor.`);
      return;
    }

    const editor = editorFromMenu;
    const cleanedUrl = this.cleanExtractedUrl(target.rawUrl);

    if (!cleanedUrl) {
      new Notice(`${NOTICE_PREFIX}: no URL found.`);
      return;
    }

    if (!this.isValidHttpUrl(cleanedUrl)) {
      new Notice(`${NOTICE_PREFIX}: invalid URL.`);
      return;
    }

    if (!this.settings.apiKey.trim()) {
      new Notice(`${NOTICE_PREFIX}: add your Gemini API key in plugin settings.`);
      return;
    }

    try {
      const summary = await this.requestGeminiSummary(cleanedUrl);
      const output = `${this.formatOutput(summary)}\n`;
      editor.replaceRange(output, target.insertBefore);
      new Notice(`${NOTICE_PREFIX}: summary inserted.`);
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
    const prompt = this.buildSummaryPrompt(url);

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

    return this.fitSummaryLength(normalized);
  }

  private buildSummaryPrompt(url: string): string {
    const customPrompt = this.settings.customPrompt.trim();
    const target = clampSummaryLengthChars(this.settings.summaryLengthChars);
    const margin = Math.max(40, Math.round(target * 0.2));
    const minLength = Math.max(MIN_SUMMARY_LENGTH_CHARS, target - margin);
    const maxLength = Math.min(MAX_SUMMARY_LENGTH_CHARS, target + margin);
    const basePrompt =
      customPrompt.length > 0
        ? customPrompt
        : "Use URL Context to read the provided URL and summarize the page.";
    const constraints = [
      "Write exactly one plain-text paragraph.",
      `Target length is about ${target} characters (aim for ${minLength} to ${maxLength}).`,
      "End at a full sentence boundary and do not cut off in the middle of a sentence.",
      "Do not use bullet points."
    ].join(" ");

    return `${basePrompt}\n\nAdditional requirements: ${constraints}\n\nURL: ${url}`;
  }

  private fitSummaryLength(summary: string): string {
    const target = clampSummaryLengthChars(this.settings.summaryLengthChars);
    const maxLength = Math.min(MAX_SUMMARY_LENGTH_CHARS, target + Math.max(60, Math.round(target * 0.25)));
    if (summary.length <= maxLength) {
      return summary;
    }

    const candidate = summary.slice(0, maxLength);
    const sentenceBoundaryIndex = this.findLastSentenceBoundaryIndex(candidate);
    if (sentenceBoundaryIndex > Math.round(target * 0.6)) {
      return candidate.slice(0, sentenceBoundaryIndex).trim();
    }

    return summary;
  }

  private findLastSentenceBoundaryIndex(text: string): number {
    let boundaryIndex = -1;
    const sentenceBoundaryRegex = /[.!?](?=\s|$)/g;
    let match = sentenceBoundaryRegex.exec(text);
    while (match) {
      boundaryIndex = match.index + 1;
      match = sentenceBoundaryRegex.exec(text);
    }
    return boundaryIndex;
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
      return `${NOTICE_PREFIX}: unsupported or unreadable page.`;
    }

    if (lower.includes("api key") || lower.includes("unauth") || lower.includes("permission")) {
      return `${NOTICE_PREFIX}: Gemini request failed. Check API key and model settings.`;
    }

    if (
      lower.includes("unsupported") ||
      lower.includes("unreadable") ||
      lower.includes("cannot fetch") ||
      lower.includes("unable to fetch") ||
      lower.includes("url context")
    ) {
      return `${NOTICE_PREFIX}: unsupported or unreadable page.`;
    }

    return `${NOTICE_PREFIX}: Gemini request failure (${message}).`;
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
    new Setting(containerEl).setName("Gemini link summarizer").setHeading();

    new Setting(containerEl)
      .setName("Gemini API key")
      .setDesc("API key used for Gemini requests.")
      .addText((text) =>
        text
          .setValue(this.plugin.settings.apiKey)
          .onChange(async (value) => {
            this.plugin.settings.apiKey = value.trim();
            await this.plugin.saveSettings();
          })
      );

    new Setting(containerEl)
      .setName("Model name")
      .setDesc("Gemini model to use. You can type any model name.")
      .addText((text) =>
        text
          .setValue(this.plugin.settings.modelName)
          .onChange(async (value) => {
            this.plugin.settings.modelName = value.trim() || DEFAULT_SETTINGS.modelName;
            await this.plugin.saveSettings();
          })
      );

    new Setting(containerEl)
      .setName("Flash model presets")
      .setDesc("Quickly choose a recent Flash preview model.")
      .addButton((button) =>
        button.setButtonText("3.1 flash lite preview").onClick(async () => {
          this.plugin.settings.modelName = FLASH_MODEL_PRESETS[0];
          await this.plugin.saveSettings();
          this.display();
        })
      )
      .addButton((button) =>
        button.setButtonText("3.0 flash preview").onClick(async () => {
          this.plugin.settings.modelName = FLASH_MODEL_PRESETS[1];
          await this.plugin.saveSettings();
          this.display();
        })
      );

    new Setting(containerEl)
      .setName("Summary length (characters)")
      .setDesc(`Target length for the summary paragraph (${MIN_SUMMARY_LENGTH_CHARS}-${MAX_SUMMARY_LENGTH_CHARS}).`)
      .addText((text) =>
        text
          .setPlaceholder(String(DEFAULT_SETTINGS.summaryLengthChars))
          .setValue(String(this.plugin.settings.summaryLengthChars))
          .onChange(async (value) => {
            const parsed = Number.parseInt(value, 10);
            if (Number.isNaN(parsed)) {
              return;
            }

            this.plugin.settings.summaryLengthChars = clampSummaryLengthChars(parsed);
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
      .setDesc("Prepends the current timestamp before the inserted summary.")
      .addToggle((toggle) =>
        toggle.setValue(this.plugin.settings.includeTimestamp).onChange(async (value) => {
          this.plugin.settings.includeTimestamp = value;
          await this.plugin.saveSettings();
        })
      );
  }
}
