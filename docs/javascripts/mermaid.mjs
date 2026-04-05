const getDocumentObservable = () => window["document$"];

const getMermaid = () => window.mermaid;

const getMermaidTheme = () =>
  document.body?.getAttribute("data-md-color-scheme") === "slate"
    ? "dark"
    : "default";

const resetRenderedDiagrams = () => {
  for (const block of document.querySelectorAll(".mermaid")) {
    if (!block.dataset.mermaidSource) {
      block.dataset.mermaidSource = block.textContent?.trim() ?? "";
    }

    if (block.getAttribute("data-processed") === "true") {
      block.removeAttribute("data-processed");
      block.innerHTML = block.dataset.mermaidSource;
    }
  }
};

const renderMermaid = async () => {
  const mermaid = getMermaid();
  if (!mermaid) {
    return;
  }

  resetRenderedDiagrams();
  mermaid.initialize({
    startOnLoad: false,
    theme: getMermaidTheme(),
  });
  await mermaid.run({ querySelector: ".mermaid" });
};

let paletteObserver;

const observePaletteChanges = () => {
  if (paletteObserver || !document.body) {
    return;
  }

  paletteObserver = new MutationObserver((mutations) => {
    for (const mutation of mutations) {
      if (mutation.attributeName === "data-md-color-scheme") {
        void renderMermaid();
      }
    }
  });

  paletteObserver.observe(document.body, {
    attributes: true,
    attributeFilter: ["data-md-color-scheme"],
  });
};

const onDocumentReady = () => {
  observePaletteChanges();
  void renderMermaid();
};

const documentObservable = getDocumentObservable();
if (documentObservable?.subscribe) {
  documentObservable.subscribe(onDocumentReady);
} else if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", onDocumentReady, { once: true });
} else {
  onDocumentReady();
}
