import { defineConfig, globalIgnores } from "eslint/config";
import obsidianmd from "eslint-plugin-obsidianmd";

export default defineConfig([
  // Build output and Node build scripts are not plugin source.
  globalIgnores(["node_modules", "dist", "main.js", "esbuild.config.mjs", ".remember"]),
  ...obsidianmd.configs.recommendedWithLocalesEn,
  {
    languageOptions: {
      parserOptions: {
        projectService: {
          allowDefaultProject: ["eslint.config.*"]
        }
      }
    },
    rules: {
      // getSettingDefinitions() is @since 1.13.0; minAppVersion is 1.5.0, so the
      // declarative settings API cannot be adopted without raising minAppVersion.
      "obsidianmd/settings-tab/prefer-setting-definitions": "off"
    }
  }
]);
