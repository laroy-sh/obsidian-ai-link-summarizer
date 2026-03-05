import { defineConfig } from "eslint/config";
import tsParser from "@typescript-eslint/parser";
import json from "@eslint/json";
import obsidianmd from "eslint-plugin-obsidianmd";

export default defineConfig([
  {
    files: ["**/*.ts"],
    languageOptions: {
      parser: tsParser,
      parserOptions: {
        project: "./tsconfig.json",
        tsconfigRootDir: process.cwd(),
        sourceType: "module",
        ecmaVersion: 2022
      }
    },
    plugins: {
      obsidianmd
    },
    rules: obsidianmd.configs.recommendedWithLocalesEn
  },
  {
    files: ["manifest.json"],
    plugins: {
      json,
      obsidianmd
    },
    language: "json/json",
    rules: {
      "obsidianmd/validate-manifest": "error",
      "obsidianmd/ui/sentence-case-json": "error"
    }
  }
]);
