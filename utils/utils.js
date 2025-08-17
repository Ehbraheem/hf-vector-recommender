import { createRequire } from "module";
import { dirname } from "path";
import { fileURLToPath, pathToFileURL } from "url";

export function makeRequire(importMetaUrl) {
  // Create a `require` that resolves relative to the caller file
  const __dirname = dirname(fileURLToPath(importMetaUrl));
  
  return createRequire(pathToFileURL(__dirname + "/"));
}

/**
 * Native require (like CommonJS `require`).
 * Good for loading node_modules or global packages.
 */
export function nativeRequire(importMetaUrl) {
  return createRequire(importMetaUrl);
}