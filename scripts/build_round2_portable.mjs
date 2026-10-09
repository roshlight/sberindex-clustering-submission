#!/usr/bin/env node
// Native Data Analytics packaging with one recorded viewport/scrollbar compatibility fix.
// The installed plugin, artifact payload and compressed reader runtime remain unchanged.
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, mkdtempSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import assert from "node:assert/strict";

const STYLE_ID = "round2-scrollbar-compatibility";
const CSS = "html body .portable-page-header,html body .analytics-top-bar" +
  "{width:100%;margin-left:0;margin-right:0}";
const STYLE = `<style id="${STYLE_ID}">${CSS}</style>`;
const EMBEDDED_IDS = [
  "data-analytics-portable-artifact-payload-source",
  "data-analytics-portable-reader-runtime-source",
];
const hash = value => createHash("sha256").update(value).digest("hex");
const fileHash = path => hash(readFileSync(path));
const writeJson = (path, value) => writeFileSync(path, JSON.stringify(value, null, 2), "utf8");

export function applyHeaderCompatibilityPatch(html) {
  if (html.includes(STYLE_ID)) throw new Error("Compatibility style already present");
  if (html.split("</head>").length !== 2) throw new Error("Expected one native document head");
  const patched = html.replace("</head>", `${STYLE}</head>`);
  assert.equal(patched.replace(STYLE, ""), html);
  return patched;
}

function embeddedHashes(html) {
  return Object.fromEntries(EMBEDDED_IDS.map(id => {
    const block = html.match(new RegExp(`<template[^>]*\\bid="${id}"[^>]*>([\\s\\S]*?)<\\/template>`));
    if (!block) throw new Error(`Native compressed block missing: ${id}`);
    return [id, hash(block[1])];
  }));
}

function options(argv) {
  if (argv.includes("--self-test")) return { selfTest: true };
  const allowed = new Set(["plugin", "input", "output", "receipt", "work-dir"]);
  const values = {};
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i].slice(2), value = argv[i + 1];
    if (!argv[i].startsWith("--") || !allowed.has(key) || !value || value.startsWith("--") || key in values)
      throw new Error(`Invalid argument: ${argv[i]}`);
    values[key] = value;
  }
  for (const key of ["plugin", "input", "output", "work-dir"])
    if (!values[key]) throw new Error(`--${key} is required`);
  return values;
}

async function main() {
  const args = options(process.argv.slice(2));
  if (args.selfTest) {
    const original = "<!doctype html><html><head><title>Native</title></head><body>Keep all evidence" +
      EMBEDDED_IDS.map(id => `<template id="${id}" data-compression="gzip-base64">abc123</template>`).join("") +
      "</body></html>";
    const patched = applyHeaderCompatibilityPatch(original);
    assert.equal(patched.replace(STYLE, ""), original);
    assert.throws(() => applyHeaderCompatibilityPatch(patched), /already present/);
    assert.throws(() => applyHeaderCompatibilityPatch("<body>no head</body>"), /one native/);
    assert.deepEqual(embeddedHashes(patched), embeddedHashes(original));
    console.log(JSON.stringify({ ok: true, checks: ["exact one-style-only change", "duplicate guard", "head guard", "compressed block identity"] }));
    return;
  }
  const plugin = resolve(args.plugin), input = resolve(args.input), output = resolve(args.output);
  const receiptPath = resolve(args.receipt ?? join(dirname(output), "portable_delivery.json"));
  if (existsSync(output) || existsSync(receiptPath)) throw new Error("Choose new output/receipt paths");
  mkdirSync(dirname(output), { recursive: true });
  mkdirSync(dirname(receiptPath), { recursive: true });
  mkdirSync(resolve(args["work-dir"]), { recursive: true });
  const work = mkdtempSync(join(resolve(args["work-dir"]), "native-build-"));
  const candidate = join(work, "candidate.html");
  const pluginScripts = join(plugin, "skills/build-report/scripts");
  const moduleNames = ["build_portable_artifact.mjs", "extract_portable_chart_svgs.mjs", "verify_portable_artifact.mjs"];
  const [builder, extractor, verifier] = await Promise.all(moduleNames.map(name =>
    import(pathToFileURL(join(pluginScripts, name)).href)));
  const artifactHash = fileHash(input);
  const receipt = {
    ok: false,
    method: "installed native builder + native SVG extraction + native verifier",
    started_at: new Date().toISOString(),
    artifact: { path: relative(dirname(output), input).replaceAll("\\", "/"), sha256: artifactHash },
    wrapper_sha256: fileHash(fileURLToPath(import.meta.url)),
    plugin_files: Object.fromEntries(moduleNames.map(name => [name, fileHash(join(pluginScripts, name))])),
    plugin_modified: false,
  };
  try {
    const artifact = JSON.parse(readFileSync(input, "utf8"));
    let html = builder.buildPortableArtifact(artifact);
    writeFileSync(candidate, html, "utf8");
    if ((artifact.manifest?.blocks ?? []).some(block => block.type === "chart")) {
      const staticCharts = await extractor.extractPortableChartSvgs({ htmlPath: candidate });
      receipt.native_static_chart_entries = Object.keys(staticCharts).length;
      html = builder.buildPortableArtifact(artifact, { staticCharts });
      writeFileSync(candidate, html, "utf8");
    }
    const before = embeddedHashes(html);
    const verifyOptions = {
      artifactPath: input, htmlPath: candidate, timeoutMs: 15000,
      screenshotPath: join(work, "native-verification-failure.png"),
    };
    let verification;
    try {
      verification = await verifier.verifyPortableArtifact(verifyOptions);
      receipt.compatibility_patch = { applied: false, reason: "native verification passed" };
    } catch (error) {
      const failure = error.verificationResult ?? {};
      if ((failure.code ?? error.code) !== "horizontal_overflow") throw error;
      receipt.original_delivery_failure = {
        code: failure.code ?? error.code, error: failure.error ?? error.message,
        reason: "Native full-bleed headers use 100vw with a 15px classic vertical scrollbar; client1425 versus scroll1433 at viewport1440.",
        diagnostic_screenshot: relative(resolve(args["work-dir"]), verifyOptions.screenshotPath).replaceAll("\\", "/"),
      };
      html = applyHeaderCompatibilityPatch(html);
      assert.deepEqual(embeddedHashes(html), before);
      writeFileSync(candidate, html, "utf8");
      verification = await verifier.verifyPortableArtifact({
        ...verifyOptions, screenshotPath: join(work, "patched-verification-failure.png"),
      });
      receipt.compatibility_patch = { applied: true, style_id: STYLE_ID, css: CSS,
        scope: "Only document header container width and horizontal margins; no clipping or hidden content" };
    }
    if (fileHash(input) !== artifactHash) throw new Error("Artifact changed during packaging");
    assert.deepEqual(verification.viewports, [1440, 390]);
    assert.equal(verification.ok, true);
    assert.deepEqual(embeddedHashes(html), before);
    receipt.verification = { ...verification, html: relative(dirname(output), output).replaceAll("\\", "/") };
    receipt.embedded_payload_and_runtime_unchanged = true;
    receipt.compressed_block_sha256 = before;
    receipt.external_requests = "none; enforced by successful native browser verification";
    receipt.output_sha256 = hash(html);
    receipt.output_bytes = Buffer.byteLength(html, "utf8");
    renameSync(candidate, output);
    receipt.ok = true;
    receipt.finished_at = new Date().toISOString();
    writeJson(receiptPath, receipt);
    console.log(JSON.stringify({ ok: true, output, receipt: receiptPath, counts: verification.counts,
      viewports: verification.viewports, sourceDialog: verification.sourceDialog }));
  } catch (error) {
    receipt.error = { code: error.verificationResult?.code ?? error.code ?? "build_failed", message: error.message };
    receipt.finished_at = new Date().toISOString();
    writeJson(receiptPath, receipt);
    throw error;
  }
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  main().catch(error => { console.error(error.message); process.exitCode = 1; });
}
