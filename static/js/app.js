(function () {
  "use strict";

  const ALLOWED = ["pdf", "docx", "txt"];

  /* -------------------------------------------------------------------------
     Category segments: toggle buttons that call onChange(category | null).
     Pressing the active segment again clears the filter.
     ----------------------------------------------------------------------- */
  function initSegments(onChange) {
    const group = document.getElementById("category-filter");
    if (!group) return;
    const buttons = Array.from(group.querySelectorAll(".segment"));
    buttons.forEach((btn) =>
      btn.addEventListener("click", () => {
        const wasActive = btn.getAttribute("aria-pressed") === "true";
        buttons.forEach((b) => b.setAttribute("aria-pressed", "false"));
        if (!wasActive) btn.setAttribute("aria-pressed", "true");
        onChange(wasActive ? null : btn.dataset.category);
      })
    );
  }

  /* -------------------------------------------------------------------------
     Analyze form
     ----------------------------------------------------------------------- */
  function initAnalyzeForm() {
    const form = document.getElementById("analyze-form");
    if (!form) return;

    const input = document.getElementById("resume");
    const zone = document.getElementById("dropzone");
    const nameEl = document.getElementById("file-name");
    const errorEl = document.getElementById("file-error");
    const submit = document.getElementById("submit-btn");
    const note = document.getElementById("progress-note");
    const select = document.getElementById("job_id");
    const maxMb = Number(input.dataset.maxMb || 5);
    const defaultLabel = nameEl.innerHTML;

    // --- file validation & drag/drop ---
    function showError(message) {
      errorEl.textContent = message;
      errorEl.hidden = !message;
    }

    function validateFile() {
      const file = input.files && input.files[0];
      zone.classList.toggle("has-file", Boolean(file));
      if (!file) {
        nameEl.innerHTML = defaultLabel;
        showError("");
        return false;
      }
      nameEl.textContent = file.name;
      const ext = file.name.split(".").pop().toLowerCase();
      if (!ALLOWED.includes(ext)) {
        showError("Unsupported file type. Upload a PDF, DOCX or TXT file.");
        return false;
      }
      if (file.size > maxMb * 1024 * 1024) {
        showError(`That file is larger than ${maxMb} MB.`);
        return false;
      }
      showError("");
      return true;
    }

    input.addEventListener("change", validateFile);
    ["dragenter", "dragover"].forEach((type) =>
      zone.addEventListener(type, (e) => {
        e.preventDefault();
        zone.classList.add("is-dragover");
      })
    );
    ["dragleave", "drop"].forEach((type) =>
      zone.addEventListener(type, () => zone.classList.remove("is-dragover"))
    );
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      if (e.dataTransfer && e.dataTransfer.files.length) {
        input.files = e.dataTransfer.files;
        validateFile();
      }
    });

    // --- role brief ---
    let jobs = {};
    try {
      jobs = JSON.parse(document.getElementById("jobs-data").textContent);
    } catch (_) {
      /* preview is optional */
    }
    const details = document.getElementById("job-details");
    const desc = document.getElementById("job-desc");
    const skills = document.getElementById("job-skills");

    function renderBrief() {
      const job = jobs[select.value];
      details.hidden = !job;
      if (!job) return;
      desc.textContent = job.description;
      skills.replaceChildren(
        ...job.skills.map((s) => {
          const tag = document.createElement("span");
          tag.className = "tag";
          tag.textContent = s;
          return tag;
        })
      );
    }
    select.addEventListener("change", renderBrief);
    renderBrief();

    // --- category filter: rebuild the option list (hiding <option> is not portable) ---
    const placeholder = select.options[0];
    const allOptions = Array.from(select.options).slice(1);
    initSegments((category) => {
      const current = select.value;
      const visible = allOptions.filter((o) => !category || o.dataset.category === category);
      select.replaceChildren(placeholder, ...visible);
      select.value = visible.some((o) => o.value === current) ? current : "";
      if (!select.value) placeholder.selected = true;
      renderBrief();
    });

    // --- submit state ---
    const label = submit.querySelector(".btn-label");
    const idleNote = note.textContent;
    form.addEventListener("submit", (e) => {
      if (!validateFile()) {
        e.preventDefault();
        if (!input.files.length) showError("Please choose a resume file.");
        return;
      }
      submit.disabled = true;
      submit.classList.add("is-loading");
      label.textContent = "Analyzing…";
      note.textContent = "reading · scoring · writing feedback";
    });
    window.addEventListener("pageshow", (e) => {
      if (!e.persisted) return;
      submit.disabled = false;
      submit.classList.remove("is-loading");
      label.textContent = "Run Analysis";
      note.textContent = idleNote;
    });
  }

  /* -------------------------------------------------------------------------
     Roles grid filter
     ----------------------------------------------------------------------- */
  function initRoleGrid() {
    const grid = document.getElementById("role-grid");
    if (!grid) return;
    const cards = Array.from(grid.querySelectorAll(":scope > [data-category]"));
    initSegments((category) => {
      cards.forEach((card) => {
        card.hidden = Boolean(category) && card.dataset.category !== category;
      });
    });
  }

  /* -------------------------------------------------------------------------
     Catalogue role vs pasted job posting
     ----------------------------------------------------------------------- */
  function initModeSwitch() {
    const radios = Array.from(document.querySelectorAll('input[name="mode"]'));
    if (!radios.length) return;
    const panels = Array.from(document.querySelectorAll(".mode-panel"));
    const select = document.getElementById("job_id");
    const description = document.getElementById("custom_description");

    function apply() {
      const mode = (radios.find((r) => r.checked) || radios[0]).value;
      panels.forEach((p) => (p.hidden = p.dataset.mode !== mode));
      select.required = mode === "catalogue";
      description.required = mode === "custom";
    }
    radios.forEach((r) => r.addEventListener("change", apply));
    apply();
  }

  /* -------------------------------------------------------------------------
     Pending report: poll until the background feedback is written
     ----------------------------------------------------------------------- */
  function initPolling() {
    const sheet = document.querySelector("[data-poll]");
    if (!sheet) return;
    const url = sheet.dataset.poll;
    const note = document.getElementById("pending-note");
    const started = Date.now();
    let delay = 2000;

    async function check() {
      try {
        const res = await fetch(url, { headers: { Accept: "application/json" } });
        if (res.ok && (await res.json()).ready) {
          window.location.reload();
          return;
        }
      } catch (_) {
        /* transient network error: keep trying */
      }
      const seconds = Math.round((Date.now() - started) / 1000);
      if (note && seconds > 45) {
        note.textContent = `Still working (${seconds}s). The primary model may be busy, so a fallback model is writing your review.`;
      }
      delay = Math.min(delay * 1.3, 8000);
      setTimeout(check, delay);
    }
    setTimeout(check, delay);
  }

  /* -------------------------------------------------------------------------
     Simple upload forms (best-fit ranking): show file name, loading state
     ----------------------------------------------------------------------- */
  function initUploadForms() {
    document.querySelectorAll("form[data-upload-form]").forEach((form) => {
      const input = form.querySelector('input[type="file"]');
      const label = form.querySelector("[data-file-name]");
      const zone = form.querySelector(".dropzone");
      const button = form.querySelector('button[type="submit"]');
      input.addEventListener("change", () => {
        const file = input.files && input.files[0];
        zone.classList.toggle("has-file", Boolean(file));
        if (file) label.textContent = file.name;
      });
      form.addEventListener("submit", () => {
        button.disabled = true;
        button.classList.add("is-loading");
        button.querySelector(".btn-label").textContent = "Ranking…";
      });
      window.addEventListener("pageshow", (e) => {
        if (!e.persisted) return;
        button.disabled = false;
        button.classList.remove("is-loading");
        button.querySelector(".btn-label").textContent = "Rank Roles";
      });
    });
  }

  function initActions() {
    document.querySelectorAll('[data-action="print"]').forEach((btn) =>
      btn.addEventListener("click", () => window.print())
    );
    document.querySelectorAll("form[data-confirm]").forEach((form) =>
      form.addEventListener("submit", (e) => {
        if (!window.confirm(form.dataset.confirm)) e.preventDefault();
      })
    );
  }

  initModeSwitch();
  initAnalyzeForm();
  initRoleGrid();
  initUploadForms();
  initPolling();
  initActions();
})();
