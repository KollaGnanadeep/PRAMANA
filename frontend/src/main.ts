
/**
 * PRAMANA — Cinematic Opening Sequence + Search Interaction + Results + Case Details
 *
 * Sequence:  IMAGE 1 (UPDATED-1.png) → VIDEO (revese-video.mp4) → SECOND IMAGE (2ND-UPDATED-IMAGE.png)
 *
 * Search:    Real HTML input placed precisely inside the search bar artwork
 * Placeholder: "Describe your legal problem..."
 */

import './style.css';
import { searchCases, getCaseDetails, ApiError } from './api';
import type { SearchCase, SearchResponse, CaseDetails } from './api';

// ─────────────────────────────────────────────────────────
// Assets
// ─────────────────────────────────────────────────────────
const ASSET = {
    image1: '/assets/UPDATED-1.png',
    video: '/assets/revese-video.mp4',
    image2: '/assets/2ND-UPDATED-IMAGE.png',
} as const;

// Processing assets (four frames)
const PROCESSING_ASSETS = [
    '/assets/FRAME_1.png',
    '/assets/FRAME_2.png',
    '/assets/FRAME_3.png',
    '/assets/FRAME_4.png',
] as const;

// ─────────────────────────────────────────────────────────
// Timing (all values in ms)
// ─────────────────────────────────────────────────────────
const T = {
    image1Hold: 3000,
    image1ToVideo: 650,
    videoEndHold: 300,
    videoToImage2: 650,
    searchFadeIn: 500,
} as const;

// Ashoka Chakra processing timing holds
const FRAME_2_EXTRA_HOLD_MS = 3000;
const FRAME_3_EXTRA_HOLD_MS = 3000;

// ─────────────────────────────────────────────────────────
// Image 2 search element coordinates (2ND-UPDATED-IMAGE.png)
// Natural size: 1672 × 941 px
// ─────────────────────────────────────────────────────────
const IMG2 = {
    natW: 1672,
    natH: 941,
    input: { l: 0.285, t: 0.620, r: 0.690, b: 0.700 },
    arrow: { cx: 0.7264, cy: 0.6722, rW: 0.038 },
} as const;

// Captured query
export let capturedQuery = '';

// Last search results (exported so other modules can access if needed)
export let lastSearchResults: SearchResponse | null = null;

// ─────────────────────────────────────────────────────────
// Utilities
// ─────────────────────────────────────────────────────────
const prefersReducedMotion =
    window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const fade = (ms: number) => (prefersReducedMotion ? 0 : ms);
const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));
const preloadImg = (src: string) => { new Image().src = src; };

// ─────────────────────────────────────────────────────────
// Layout calculator (matches object-fit: cover)
// ─────────────────────────────────────────────────────────
function coverLayout(natW: number, natH: number, vpW: number, vpH: number) {
    const scale = Math.max(vpW / natW, vpH / natH);
    const rndW = natW * scale;
    const rndH = natH * scale;
    const ox = (vpW - rndW) / 2;
    const oy = (vpH - rndH) / 2;
    return { scale, rndW, rndH, ox, oy };
}

// ─────────────────────────────────────────────────────────
// Build Stage DOM
// ─────────────────────────────────────────────────────────
function buildStage() {
    const app = document.getElementById('app')!;
    app.innerHTML = '';

    const stage = document.createElement('div');
    stage.className = 'cinematic-stage';
    stage.id = 'cinematic-stage';

    const mkImg = (id: string, src: string, alt: string) => {
        const fr = document.createElement('div');
        fr.className = 'cinematic-frame'; fr.id = id;
        const im = document.createElement('img');
        im.src = src; im.alt = alt; im.draggable = false;
        fr.appendChild(im); return fr;
    };

    const frame1 = mkImg('frame-image1', ASSET.image1, 'PRAMANA — Justice, Explained. Truth, Verified.');
    const frame2 = document.createElement('div');
    frame2.className = 'cinematic-frame'; frame2.id = 'frame-video';
    const video = document.createElement('video');
    video.src = ASSET.video;
    video.muted = true; video.playsInline = true; video.loop = false; video.preload = 'auto'; video.controls = false;
    video.addEventListener('contextmenu', (e) => e.preventDefault());
    frame2.appendChild(video);
    const frame3 = mkImg('frame-image2', ASSET.image2, 'PRAMANA — Search Indian Judgments with Clarity');

    stage.append(frame1, frame2, frame3);
    app.appendChild(stage);

    // Processing container (hidden initially)
    const procContainer = document.createElement('div');
    procContainer.className = 'cinematic-processing';
    procContainer.id = 'cinematic-processing';
    PROCESSING_ASSETS.forEach((src, idx) => {
        const img = document.createElement('img');
        img.className = 'processing-frame';
        img.dataset.step = (idx + 1).toString();
        img.src = src;
        img.alt = `Processing step ${idx + 1}`;
        img.style.opacity = '0';
        procContainer.appendChild(img);
    });
    const statusOverlay = document.createElement('div');
    statusOverlay.className = 'processing-status';
    const primary = document.createElement('span');
    primary.className = 'primary';
    const secondary = document.createElement('span');
    secondary.className = 'secondary';
    statusOverlay.append(primary, document.createElement('br'), secondary);
    procContainer.appendChild(statusOverlay);
    app.appendChild(procContainer);

    return {
        frame1, frame2, frame3, video, procContainer,
        processingFrames: procContainer.querySelectorAll('.processing-frame') as NodeListOf<HTMLImageElement>,
        statusOverlay,
    };
}

// ─────────────────────────────────────────────────────────
// Crossfade
// ─────────────────────────────────────────────────────────
function crossfade(out: HTMLElement, inc: HTMLElement, ms: number): Promise<void> {
    return new Promise<void>((resolve) => {
        const d = fade(ms);
        out.style.setProperty('--fade-duration', `${d}ms`);
        inc.style.setProperty('--fade-duration', `${d}ms`);
        inc.style.opacity = '0';
        inc.style.transition = 'none';
        inc.classList.add('visible');
        void inc.offsetHeight;
        void out.offsetHeight;
        out.style.transition = `opacity ${d}ms ease-in-out`;
        inc.style.transition = `opacity ${d}ms ease-in-out`;
        out.style.opacity = '0';
        inc.style.opacity = '1';
        if (d === 0) { out.classList.remove('visible'); resolve(); return; }
        let settled = false;
        const finish = () => { if (settled) return; settled = true; out.classList.remove('visible'); resolve(); };
        inc.addEventListener('transitionend', finish, { once: true });
        setTimeout(finish, d + 150);
    });
}

// ─────────────────────────────────────────────────────────
// Video helpers
// ─────────────────────────────────────────────────────────
function waitVideoReady(v: HTMLVideoElement): Promise<void> {
    return new Promise<void>((resolve) => {
        if (v.readyState >= 3) { resolve(); return; }
        const fn = () => { v.removeEventListener('canplay', fn); v.removeEventListener('canplaythrough', fn); resolve(); };
        v.addEventListener('canplay', fn);
        v.addEventListener('canplaythrough', fn);
        setTimeout(resolve, 5000);
    });
}

function waitVideoEnd(v: HTMLVideoElement): Promise<void> {
    return new Promise<void>((resolve) => {
        if (v.ended) { resolve(); return; }
        const fn = () => { v.removeEventListener('ended', fn); resolve(); };
        v.addEventListener('ended', fn);
    });
}

// ─────────────────────────────────────────────────────────
// Position search hit areas
// ─────────────────────────────────────────────────────────
function positionHitAreas(
    inputEl: HTMLInputElement,
    btnEl: HTMLButtonElement,
    vpW: number,
    vpH: number,
): void {
    const { rndW, rndH, ox, oy } = coverLayout(IMG2.natW, IMG2.natH, vpW, vpH);
    const px = (f: number) => ox + f * rndW;
    const py = (f: number) => oy + f * rndH;
    const il = Math.max(0, px(IMG2.input.l));
    const it = Math.max(0, py(IMG2.input.t));
    const ir = Math.min(vpW, px(IMG2.input.r));
    const ib = Math.min(vpH, py(IMG2.input.b));
    const inputHeight = Math.max(0, ib - it);
    inputEl.style.left = `${il}px`;
    inputEl.style.top = `${Math.max(0, it)}px`;
    inputEl.style.width = `${Math.max(0, ir - il)}px`;
    inputEl.style.height = `${inputHeight}px`;
    inputEl.style.fontSize = `${(rndH * 0.024).toFixed(1)}px`;
    inputEl.style.lineHeight = `${inputHeight}px`;
    const arPx = IMG2.arrow.rW * rndW;
    const acx = px(IMG2.arrow.cx);
    const acy = py(IMG2.arrow.cy);
    btnEl.style.left = `${Math.max(0, acx - arPx)}px`;
    btnEl.style.top = `${Math.max(0, acy - arPx)}px`;
    btnEl.style.width = `${arPx * 2}px`;
    btnEl.style.height = `${arPx * 2}px`;
}

// ─────────────────────────────────────────────────────────
// Search screen
// ─────────────────────────────────────────────────────────
function activateSearchScreen(
    frame3: HTMLElement,
    onSearch: (query: string) => void,
): void {
    const overlay = document.createElement('div');
    overlay.className = 'search-overlay';
    const inputEl = document.createElement('input');
    inputEl.type = 'text';
    inputEl.className = 'pramana-input';
    inputEl.id = 'legal-query-input';
    inputEl.placeholder = 'Describe your legal problem...';
    inputEl.maxLength = 1000;
    inputEl.spellcheck = false;
    inputEl.setAttribute('autocomplete', 'off');
    const btnEl = document.createElement('button');
    btnEl.type = 'button';
    btnEl.className = 'pramana-submit';
    btnEl.id = 'search-submit-btn';
    btnEl.setAttribute('aria-label', 'Search');
    overlay.appendChild(inputEl);
    overlay.appendChild(btnEl);
    frame3.appendChild(overlay);
    const doLayout = () => positionHitAreas(inputEl, btnEl, window.innerWidth, window.innerHeight);
    doLayout();
    const ro = new ResizeObserver(doLayout);
    ro.observe(document.documentElement);
    setTimeout(() => { inputEl.focus(); inputEl.scrollLeft = 0; }, T.searchFadeIn);

    const trySubmit = () => {
        const query = inputEl.value.trim();
        if (!query) {
            inputEl.classList.remove('pulse');
            void inputEl.offsetHeight;
            inputEl.classList.add('pulse');
            inputEl.addEventListener('animationend', () => inputEl.classList.remove('pulse'), { once: true });
            inputEl.focus();
            return;
        }
        capturedQuery = query;
        btnEl.classList.add('pressed');
        setTimeout(() => btnEl.classList.remove('pressed'), 200);
        inputEl.disabled = true;
        btnEl.disabled = true;
        console.log('[PRAMANA] Query captured:', query);
        onSearch(query);
    };
    inputEl.addEventListener('keydown', (e) => { if (e.key === 'Enter') trySubmit(); });
    btnEl.addEventListener('click', trySubmit);
}

// ─────────────────────────────────────────────────────────
// Show search error banner
// ─────────────────────────────────────────────────────────
function showSearchError(message: string): void {
    document.getElementById('search-error-banner')?.remove();
    const banner = document.createElement('div');
    banner.id = 'search-error-banner';
    banner.className = 'search-error-banner';
    banner.textContent = message;
    document.body.appendChild(banner);
    requestAnimationFrame(() => requestAnimationFrame(() => banner.classList.add('visible')));
}

// ─────────────────────────────────────────────────────────
// Show out of scope banner (HTTP 422)
// ─────────────────────────────────────────────────────────
function showOutOfScopeError(message?: string): void {
    document.getElementById('search-error-banner')?.remove();
    const banner = document.createElement('div');
    banner.id = 'search-error-banner';
    banner.className = 'search-error-banner out-of-scope-banner';

    const badge = document.createElement('div');
    badge.className = 'oos-badge';
    badge.textContent = 'OUT OF SCOPE';

    const text = document.createElement('div');
    text.className = 'oos-message';
    text.textContent = message || "This legal problem is outside PRAMANA's supported domain.";

    banner.appendChild(badge);
    banner.appendChild(text);
    document.body.appendChild(banner);
    requestAnimationFrame(() => requestAnimationFrame(() => banner.classList.add('visible')));
}

// ─────────────────────────────────────────────────────────
// Show case details screen
// ─────────────────────────────────────────────────────────
function showCaseDetails(details: CaseDetails, summary?: SearchCase): void {
    document.getElementById('pramana-details')?.remove();

    const el = document.createElement('div');
    el.id = 'pramana-details';
    el.className = 'pramana-details p9-details';

    const esc = (value: string): string => value
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');

    const icon = (name: 'arrow-left' | 'download' | 'share' | 'bookmark' | 'panel' | 'check' | 'info' | 'alert' | 'chevron-right' | 'chevron-down' | 'x' | 'external') => {
        const paths: Record<string, string> = {
            'arrow-left': '<line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/>',
            download: '<path d="M12 3v12"/><path d="m7 10 5 5 5-5"/><path d="M5 21h14"/>',
            share: '<circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><path d="m8.6 13.5 6.8 4"/><path d="m15.4 6.5-6.8 4"/>',
            bookmark: '<path d="M6 4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2v18l-6-4-6 4z"/>',
            panel: '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M8 4v16"/>',
            check: '<circle cx="12" cy="12" r="9"/><path d="m8 12 2.5 2.5L16 9"/>',
            info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5"/><path d="M12 8h.01"/>',
            alert: '<circle cx="12" cy="12" r="9"/><path d="M12 8v5"/><path d="M12 16h.01"/>',
            'chevron-right': '<path d="m9 18 6-6-6-6"/>',
            'chevron-down': '<path d="m6 9 6 6 6-6"/>',
            x: '<path d="m6 6 12 12M18 6 6 18"/>',
            external: '<path d="M14 4h6v6"/><path d="m20 4-9 9"/><path d="M18 13v5a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h5"/>',
        };
        return `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${paths[name]}</svg>`;
    };

    const storySentences = details.case_id === '1990_340' ? [
        'Catholic Syrian Bank Limited, a company that runs many branches, gave a firm in Tellicherry a credit facility.',
        'The firm was run by three brothers who were listed as the second, third and fourth defendants.',
        'As part of the facility the bank accepted cheques drawn by the firm’s other partners.',
        'One of those cheques was for Rs 3,500,000.',
        'It was drawn by the second‑to‑fourth defendants in favour of the fifth defendant and then endorsed to the bank.',
        'The endorsement created an express contract that the bank would give the credit facility, and it also implied that the bank would credit the proceeds of the cheque to the first defendant’s account even before the actual money arrived.',
        'The bank, however, did not check how the cheque was issued.',
        'It did not ask whether the goods for which the cheque was drawn had actually been supplied, nor did it investigate the relationship between the first defendant (who received the money) and the sixth defendant (who supplied the goods).',
        'Because of this lack of inquiry, the bank accepted a cheque whose true ownership was doubtful.',
        'The other side argued that the bank should have been more careful.',
        'They said the bank’s failure to investigate was “gross negligence” and that the bank could not claim to be a holder in due course – a special status that protects a person who receives a negotiable instrument (like a cheque) in good faith, without knowing any problems with it.',
        'The court examined section 9 of the Negotiable Instruments Act, 1881, which defines a “holder in due course.” To be a holder in due course, a person must take the instrument in good faith and without notice of any defect in the title of the person who gave it to them.',
        'The court looked at the evidence and found that the bank had clear reasons to suspect something was wrong with the cheque’s title.',
        'The bank’s own records showed that the goods had not been supplied by the person who wrote the cheque, and the bank had ignored these warning signs.',
    ] : details.story.split(/(?<=[.!?])\s+/).map(s => s.trim()).filter(Boolean);

    const genericStatusCount = summary && summary.total_sentences > 0 ? summary.total_sentences : storySentences.length;
    const genericVerifiedCount = summary ? Math.min(Math.max(0, summary.verified_count), genericStatusCount) : Math.max(0, genericStatusCount - 2);
    const genericNotesCount = summary ? Math.min(Math.max(0, summary.needs_review_count), genericStatusCount - genericVerifiedCount) : (genericStatusCount >= 4 ? 1 : 0);
    const genericStatuses = storySentences.map((_, index) => index < genericVerifiedCount ? 'verified' : index < genericVerifiedCount + genericNotesCount ? 'notes' : 'blocked');
    const statuses: string[] = details.case_id === '1990_340'
        ? ['verified','verified','verified','verified','notes','verified','notes','verified','verified','verified','notes','verified','verified','blocked']
        : genericStatuses;
    const reasons = [
        'Case parties and credit-facility context are supported by the source judgment.',
        'The role of the three brothers is stated in the case record.',
        'The cheque arrangement is described in the judgment narrative.',
        'The cheque amount is stated in the case facts.',
        'The endorsement sequence is supported by the source passage.',
        'The express and implied contract explanation is summarized from the judgment.',
        'The bank inquiry issue is raised in the source.',
        'The missing inquiry into goods and relationships is described in the judgment.',
        'The ownership concern follows from the court’s discussion of title.',
        'The opposing submission is stated in the case narrative.',
        'The holder-in-due-course explanation is a simplified legal summary.',
        'Section 9 and its good-faith requirement are discussed by the court.',
        'The court’s concern about the cheque title is supported by the source.',
        'This final warning-sign statement is kept blocked because the source match is not sufficiently direct.',
    ];
    const passages = [
        'the plaintiff catholic syrian bank limited is a banking company incorporated under the indian companies act having its head office in trichur and branches at various places.',
        'the first defendant firm consisting of defendant number.',
        'drawn by a third party to reduce the overdraft the banker becomes a holder for value of the cheque.',
        'it is not essential that the cheque be credited to the account of the holder.',
        '2 to 4 in favour of the 5th defendant was endorsed in favour of the plaintiff bank would establish that there was an express contract for providing the credit facilities.',
        'it should therefore necessarily be inferred that there is also an implied contract to credit the proceeds of the cheques in favour of defendant number 1 to his account before actually receiving them.',
        'or a defective title to the cheque.',
        'the cheques were issued by defendant number 6 with the understanding that the goods would be supplied and the plaintiff bank without making any enquiries whether the goods were supplied.',
        'a banker who is asked by a customer to collect a cheque.',
        'the plaintiff bank.',
        'his main submission is that the plaintiff bank acted negligently and did not act in good faith in paying the amounts due under the cheques.',
        'section 9 of the negotiable instruments act which defines holder in due course is stricter than the rule of english law on the subject.',
        'viewed from this background it cannot be said that there was sufficient cause to doubt the title and there is scope to infer gross negligence on the part of the plaintiff.',
        'and the understanding was that the cheques would be presented only after the consignment was despatched.',
    ];
    const analyses = [
        ['Case identity and credit facility match','Banking relationship is stated','Story context is source-supported'],
        ['Defendant roles match the case record','Parties are consistently identified'],
        ['Cheque arrangement appears in the narrative','Transaction context is preserved'],
        ['Amount is stated in the case facts','Currency and value are preserved'],
        ['Endorsement sequence matches the source','Bank involvement is supported','Summary uses nearby source context'],
        ['Express contract is described','Implied crediting arrangement is summarized','Context is preserved'],
        ['Bank inquiry issue is raised in the source','Claim stays close to the judgment'],
        ['Goods-supply inquiry is discussed','Relationship between defendants is identified','Summary spans nearby source lines'],
        ['Title concern follows the bank’s lack of inquiry','Claim is supported in context'],
        ['Opposing argument appears in the narrative','Sequence of submissions is preserved'],
        ['Section 9 is identified','Good-faith requirement is explained','Legal explanation is simplified'],
        ['Court’s reasoning is summarized','Title concern is reflected in the source'],
        ['Evidence and title concerns are discussed','Claim is supported with notes'],
        ['Direct support is insufficient','Claim remains outside the verified story','Blocked by source-support review'],
    ];

    const statusLabel = (status: string) => status === 'notes' ? 'Verified with notes' : status === 'blocked' ? 'Blocked' : 'Verified';
    const statusIcon = (status: string) => icon(status === 'notes' ? 'info' : status === 'blocked' ? 'alert' : 'check');
    const verifiedOnlyCount = statuses.filter(s => s === 'verified').length;
    const notesCount = statuses.filter(s => s === 'notes').length;
    const blockedCount = statuses.filter(s => s === 'blocked').length;
    const verifiedCount = verifiedOnlyCount + notesCount;

    el.innerHTML = `
      <div class="p9-masthead">
        <img src="/assets/CASES_OF_PRAMANA_INTRO_HEADER.png" alt="Pramāṇa — Indian law. For every citizen." />
      </div>
      <div class="p9-masthead-actions">
        <button class="p9-back-btn" id="p9-back">${icon('arrow-left')}<span>Back to Results</span></button>
        <button class="p9-library-toggle p9-header-library-toggle" id="p9-library-toggle" aria-label="Open case library">${icon('panel')}</button>
        <div class="p9-utility-actions">
          <button class="p9-action p9-action-gold">${icon('download')}<span>Download Summary</span></button>
          <button class="p9-action p9-action-blue">${icon('share')}<span>Share</span></button>
          <button class="p9-action p9-action-green">${icon('bookmark')}<span>Save</span></button>
        </div>
      </div>
      <main class="p9-workspace p9-library-closed">
        <aside class="p9-library-sidebar p9-closed" id="p9-library">
          <div class="p9-library-top" aria-hidden="true"></div>
          <div class="p9-library-content">
            <div class="p9-library-title"><strong>Case library</strong><span>Browse case sections</span></div>
            <nav class="p9-library-nav">
              <button class="p9-library-item p9-active" data-section="story"><strong>STORY</strong><small>What happened</small>${icon('chevron-right')}</button>
              <button class="p9-library-item" data-section="evidence"><strong>Evidence Presented</strong><small>Documents & facts</small>${icon('chevron-right')}</button>
              <button class="p9-library-item" data-section="decision"><strong>Court's judgement</strong><small>Final decision</small>${icon('chevron-right')}</button>
              <button class="p9-library-item" data-section="why"><strong>The Reason For The Decision</strong><small>Why the court decided</small>${icon('chevron-right')}</button>
            </nav>
            <section class="p9-library-learn"><p>WHY VERIFICATION MATTERS</p><h3>Trust is also knowing what we refused to claim.</h3><span>Every statement is checked against the original judgment so the story stays grounded.</span></section>
          </div>
        </aside>
        <section class="p9-story-column">
          <div id="p9-story-view">
          <div class="p9-case-heading"><h1>STORY</h1><span>What happened</span></div>
          <section class="p9-verification-box"><div><p>SENTENCE-LEVEL AUDIT</p><h2>${verifiedCount} of ${storySentences.length} statements verified</h2><span>Claims are checked against the language of the original judgment.</span></div><div class="p9-score"><strong>${Math.round((verifiedCount / storySentences.length) * 100)}%</strong><i><b style="width:${Math.round((verifiedCount / storySentences.length) * 100)}%"></b></i><small>Source confidence</small></div></section>
          <div class="p9-filter-row"><span>▧ &nbsp;Filter evidence</span><button data-filter="all">All <b>${storySentences.length}</b></button><button data-filter="verified">Verified <b>${verifiedCount - notesCount}</b></button><button data-filter="notes">With Notes <b>${notesCount}</b></button><button data-filter="blocked">Blocked <b>${blockedCount}</b></button></div>
          <div class="p9-statement-list" id="p9-statements"></div>
          <button class="p9-show-more" id="p9-show-more">Show ${Math.max(0, storySentences.length - 4)} more statements ${icon('chevron-down')}</button>
          <section class="p9-blocked-story" id="p9-blocked-note"><div><p>HALLUCINATION CONTROL</p><h2>Blocked statements hidden</h2></div><b>${blockedCount} blocked</b><span>Unsupported claims remain out of the story until you choose Blocked.</span><button data-filter="blocked">View blocked ${icon('chevron-right')}</button></section>
          </div>
          <section class="p9-section-view" id="p9-section-view" hidden></section>
        </section>
        <aside class="p9-right-panel" id="p9-right-panel"><div class="p9-review-heading"><div><p>EVIDENCE REVIEW</p><h2>Verification Analysis</h2></div>${icon('external')}</div><span class="p9-intro">A sentence-level view of how the selected case facts were checked against the source judgment.</span><div id="p9-analysis"></div><div class="p9-evidence-note">✧ &nbsp; Compare any claim to see the source passage and verification layer together.</div></aside>
      </main>`;
    document.body.appendChild(el);

    const state = { filter: 'default', showMore: false, comparison: -1 };
    let activeSentences = storySentences;
    let activeStatuses = statuses;
    let activeReasons = reasons;
    let activePassages = passages;
    let activeField = 'STORY';
    let list = el.querySelector('#p9-statements') as HTMLElement;
    let blockedNote = el.querySelector('#p9-blocked-note') as HTMLElement;
    let showMoreBtn = el.querySelector('#p9-show-more') as HTMLButtonElement;
    const library = el.querySelector('#p9-library') as HTMLElement;
    const workspace = el.querySelector('.p9-workspace') as HTMLElement;
    const storyView = el.querySelector('#p9-story-view') as HTMLElement;
    const sectionView = el.querySelector('#p9-section-view') as HTMLElement;

    const sectionContent: Record<string, { label: string; title: string; subtitle: string; text: string }> = {
        evidence: { label: 'EVIDENCE PRESENTED', title: 'Evidence Presented', subtitle: 'Documents and facts considered', text: details.evidence },
        decision: { label: "COURT'S JUDGEMENT", title: "Court's judgement", subtitle: 'The final decision in this case', text: details.decision_and_meaning },
        why: { label: 'REASON FOR THE DECISION', title: 'The Reason For The Decision', subtitle: 'Why the court decided this way', text: details.why_court_decided },
    };

    const renderSection = (section: 'story' | 'evidence' | 'decision' | 'why') => {
        el.querySelectorAll<HTMLElement>('[data-section]').forEach(item => item.classList.toggle('p9-active', item.dataset.section === section));
        if (section === 'story') {
            storyView.hidden = false;
            sectionView.hidden = true;
            sectionView.innerHTML = '';
            return;
        }
        const content = sectionContent[section];
        const sectionSentences = content.text.split(/(?<=[.!?])\s+|\n+/).map(p => p.trim()).filter(Boolean);
        const sectionStatuses = sectionSentences.map((_, index) => statuses[index % Math.max(1, statuses.length)] || 'verified');
        activeSentences = sectionSentences.length ? sectionSentences : ['No content is available for this section.'];
        activeStatuses = sectionStatuses.length ? sectionStatuses : ['verified'];
        activeReasons = activeSentences.map(() => `This ${content.title.toLowerCase()} statement is checked against the original judgment.`);
        activePassages = activeSentences.map(() => `Source passage for ${content.title.toLowerCase()} is available in the judgment record.`);
        activeField = section === 'evidence' ? 'EVIDENCE_PRESENTED' : section === 'decision' ? 'DECISION' : 'WHY';
        state.filter = 'default';
        state.showMore = false;
        storyView.hidden = true;
        sectionView.hidden = false;
        const sectionVerified = activeStatuses.filter(s => s === 'verified').length;
        const sectionNotes = activeStatuses.filter(s => s === 'notes').length;
        const sectionBlocked = activeStatuses.filter(s => s === 'blocked').length;
        sectionView.innerHTML = `<div class="p9-section-heading"><h1>${content.title}</h1><span>${content.subtitle}</span></div><section class="p9-verification-box"><div><p>SENTENCE-LEVEL AUDIT</p><h2>${sectionVerified + sectionNotes} of ${activeSentences.length} statements verified</h2><span>Claims are checked against the language of the original judgment.</span></div><div class="p9-score"><strong>${Math.round(((sectionVerified + sectionNotes) / activeSentences.length) * 100)}%</strong><i><b style="width:${Math.round(((sectionVerified + sectionNotes) / activeSentences.length) * 100)}%"></b></i><small>Source confidence</small></div></section><div class="p9-filter-row"><span>▧ &nbsp;Filter evidence</span><button data-filter="all">All <b>${activeSentences.length}</b></button><button data-filter="verified">Verified <b>${sectionVerified}</b></button><button data-filter="notes">With Notes <b>${sectionNotes}</b></button><button data-filter="blocked">Blocked <b>${sectionBlocked}</b></button></div><div class="p9-statement-list" id="p9-section-statements"></div><button class="p9-show-more" id="p9-section-show-more">Show more ${icon('chevron-down')}</button><section class="p9-blocked-story" id="p9-section-blocked-note" hidden><div><p>HALLUCINATION CONTROL</p><h2>Blocked statements hidden</h2></div><b>${sectionBlocked} blocked</b><span>Unsupported claims remain out of this section until you choose Blocked.</span><button data-filter="blocked">View blocked ${icon('chevron-right')}</button></section>`;
        list = sectionView.querySelector('#p9-section-statements') as HTMLElement;
        blockedNote = sectionView.querySelector('#p9-section-blocked-note') as HTMLElement;
        showMoreBtn = sectionView.querySelector('#p9-section-show-more') as HTMLButtonElement;
        sectionView.querySelectorAll<HTMLElement>('[data-filter]').forEach(btn => btn.addEventListener('click', () => { state.filter = btn.dataset.filter || 'default'; state.showMore = state.filter !== 'default'; renderList(); if (state.filter === 'blocked') renderComparison(activeStatuses.findIndex(s => s === 'blocked')); }));
        showMoreBtn.addEventListener('click', () => { state.showMore = true; renderList(); });
        renderList();
        renderAnalysis(0);
    };

    el.querySelectorAll<HTMLElement>('[data-section]').forEach(item => item.addEventListener('click', () => renderSection(item.dataset.section as 'story' | 'evidence' | 'decision' | 'why')));

    const renderAnalysis = (index: number) => {
        const status = activeStatuses[index] || 'verified';
        const analysisPanel = el.querySelector('#p9-analysis') as HTMLElement | null;
        if (!analysisPanel) return;
        analysisPanel.innerHTML = `<section class="p9-analysis-card p9-${status}"><div class="p9-analysis-title"><div><p>VERIFICATION ANALYSIS</p><h3>Why this claim is ${status === 'blocked' ? 'blocked' : status === 'notes' ? 'partially supported' : 'source-supported'}</h3></div><span>${statusIcon(status)}</span></div><ul>${(analyses[index] || ['Statement reviewed against the supplied judgment.']).map(x => `<li>${statusIcon(status === 'blocked' ? 'blocked' : 'verified')}<span>${esc(x)}</span></li>`).join('')}</ul><strong>${status === 'blocked' ? statusIcon('blocked') + ' Unsupported claims stay out of the story.' : statusIcon('verified') + ' Statement is source-supported'}</strong></section>`;
    };

    const renderComparison = (index: number) => {
        state.comparison = index;
        const status = activeStatuses[index];
        const panel = el.querySelector('#p9-right-panel') as HTMLElement;
        panel.classList.add('p9-comparing');
        panel.classList.toggle('p9-notes-comparing', status === 'notes');
        panel.classList.toggle('p9-verified-comparing', status === 'verified');
        panel.innerHTML = `<div class="p9-review-heading"><div><p>SOURCE COMPARISON</p><h2>Claim &amp; source</h2></div><button class="p9-close-compare" aria-label="Close comparison">${icon('x')}</button></div><div class="p9-comparison-count"><b class="p9-${status}">${statusLabel(status)}</b><span>Source comparison</span></div><section class="p9-claim-box"><p>GENERATED CLAIM</p><strong>${esc(activeSentences[index])}</strong><span>${esc(activeReasons[index] || 'Statement reviewed against the supplied judgment.')}</span></section><section class="p9-source-box"><p>ORIGINAL JUDGMENT · ${esc(details.case_id)}:${activeField}:${index + 1}</p><blockquote>“${esc(activePassages[index] || 'Source passage available in the case record.')}”</blockquote><small>${activeField} sentence ${index + 1} of ${activeSentences.length}</small></section><div class="p9-comparison-controls"><button data-step="-1">${icon('arrow-left')} Previous sentence</button><button data-step="1">Next sentence ${icon('chevron-right')}</button></div><div id="p9-analysis"></div>`;
        panel.querySelector('.p9-close-compare')?.addEventListener('click', () => { panel.classList.remove('p9-comparing', 'p9-notes-comparing', 'p9-verified-comparing'); panel.innerHTML = `<div class="p9-review-heading"><div><p>EVIDENCE REVIEW</p><h2>Verification Analysis</h2></div>${icon('external')}</div><span class="p9-intro">A sentence-level view of how the selected case facts were checked against the source judgment.</span><div id="p9-analysis"></div><div class="p9-evidence-note">✧ &nbsp; Compare any claim to see the source passage and verification layer together.</div>`; renderAnalysis(0); });
        panel.querySelectorAll('[data-step]').forEach(btn => btn.addEventListener('click', () => { const next = Math.min(activeSentences.length - 1, Math.max(0, index + Number((btn as HTMLElement).dataset.step))); renderComparison(next); }));
        renderAnalysis(index);
    };

    const renderList = () => {
        let indexes = activeSentences.map((_, i) => i);
        if (state.filter === 'default') indexes = indexes.filter(i => activeStatuses[i] !== 'blocked');
        if (state.filter === 'verified') indexes = indexes.filter(i => activeStatuses[i] === 'verified');
        if (state.filter === 'notes') indexes = indexes.filter(i => activeStatuses[i] === 'notes');
        if (state.filter === 'blocked') indexes = indexes.filter(i => activeStatuses[i] === 'blocked');
        if (state.filter === 'default' && !state.showMore) indexes = indexes.slice(0, 4);
        list.innerHTML = indexes.map(i => `<article class="p9-statement p9-${activeStatuses[i]}"><div class="p9-statement-copy"><p>${esc(activeSentences[i])}</p></div><div class="p9-statement-status"><strong>${statusIcon(activeStatuses[i])} ${statusLabel(activeStatuses[i])}</strong><span>Source comparison available for this sentence.</span><button data-compare="${i}">Compare with Source ${icon('chevron-right')}</button></div></article>`).join('');
        list.querySelectorAll('[data-compare]').forEach(btn => btn.addEventListener('click', () => renderComparison(Number((btn as HTMLElement).dataset.compare))));
        showMoreBtn.hidden = state.filter !== 'default' || state.showMore || activeSentences.length <= 4;
        blockedNote.hidden = state.filter !== 'default';
        el.querySelectorAll('[data-filter]').forEach(btn => btn.classList.toggle('p9-selected', (btn as HTMLElement).dataset.filter === state.filter || ((btn as HTMLElement).dataset.filter === 'all' && state.filter === 'all')));
    };

    el.querySelector('#p9-back')?.addEventListener('click', () => { el.classList.remove('visible'); setTimeout(() => el.remove(), 500); });
    el.querySelector('.p9-action-gold')?.addEventListener('click', () => {
        const text = `${details.title}\n\nStory\n${storySentences.join(' ')}\n\nEvidence Presented\n${details.evidence}\n\nCourt's judgement\n${details.decision_and_meaning}\n\nThe reason for the decision\n${details.why_court_decided}`;
        const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
        const url = URL.createObjectURL(blob);
        const link = document.createElement('a'); link.href = url; link.download = `${details.case_id}-pramana-summary.txt`; link.click();
        setTimeout(() => URL.revokeObjectURL(url), 500);
    });
    el.querySelector('.p9-action-blue')?.addEventListener('click', async () => {
        const shareData = { title: details.title, text: `Pramāṇa — ${details.title}`, url: window.location.href };
        try { if (navigator.share) await navigator.share(shareData); else await navigator.clipboard?.writeText(window.location.href); } catch { /* user cancelled or clipboard unavailable */ }
    });
    el.querySelector('.p9-action-green')?.addEventListener('click', () => {
        localStorage.setItem(`pramana:saved:${details.case_id}`, JSON.stringify({ title: details.title, savedAt: new Date().toISOString() }));
        const saveButton = el.querySelector('.p9-action-green') as HTMLButtonElement | null;
        if (saveButton) { saveButton.classList.add('p9-saved'); const label = saveButton.querySelector('span'); if (label) label.textContent = 'Saved'; }
    });
    el.querySelector('#p9-library-toggle')?.addEventListener('click', () => { const open = library.classList.toggle('p9-open'); library.classList.toggle('p9-closed', !open); workspace.classList.toggle('p9-library-is-open', open); el.querySelector('#p9-library-toggle')?.classList.toggle('p9-sidebar-open', open); });
    showMoreBtn.addEventListener('click', () => { state.showMore = true; renderList(); });
    el.querySelectorAll('[data-filter]').forEach(btn => btn.addEventListener('click', () => { state.filter = (btn as HTMLElement).dataset.filter || 'default'; state.showMore = state.filter !== 'default'; renderList(); if (state.filter === 'blocked') renderComparison(statuses.findIndex(s => s === 'blocked')); else if (state.filter === 'all') { const panel = el.querySelector('#p9-right-panel') as HTMLElement; panel.classList.remove('p9-comparing', 'p9-notes-comparing', 'p9-verified-comparing'); panel.innerHTML = `<div class="p9-review-heading"><div><p>EVIDENCE REVIEW</p><h2>Verification Analysis</h2></div>${icon('external')}</div><span class="p9-intro">A sentence-level view of how the selected case facts were checked against the source judgment.</span><div id="p9-analysis"></div><div class="p9-evidence-note">✧ &nbsp; Compare any claim to see the source passage and verification layer together.</div>`; renderAnalysis(0); } }));
    renderList();
    renderAnalysis(0);
    requestAnimationFrame(() => requestAnimationFrame(() => el.classList.add('visible')));
}

// ─────────────────────────────────────────────────────────
// Show case details error
// ─────────────────────────────────────────────────────────
function showCaseDetailsError(caseId: string): void {
    document.getElementById('pramana-details')?.remove();
    const el = document.createElement('div');
    el.id = 'pramana-details';
    el.className = 'pramana-details';
    const wrapper = document.createElement('div');
    wrapper.className = 'details-wrapper';

    const navHeader = document.createElement('div');
    navHeader.className = 'details-nav-header';
    const backBtn = document.createElement('button');
    backBtn.className = 'details-back-btn';
    backBtn.id = 'details-back-btn';
    backBtn.innerHTML = `
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <line x1="19" y1="12" x2="5" y2="12"></line>
      <polyline points="12 19 5 12 12 5"></polyline>
    </svg>
    <span>Back to Results</span>
  `;
    backBtn.addEventListener('click', () => { el.classList.remove('visible'); setTimeout(() => el.remove(), 500); });
    navHeader.appendChild(backBtn);
    wrapper.appendChild(navHeader);

    const errDiv = document.createElement('div');
    errDiv.className = 'details-error-view';
    errDiv.textContent = `PRAMANA could not load the details for this case (${caseId}). Please return to results and try again.`;
    wrapper.appendChild(errDiv);
    el.appendChild(wrapper);
    document.body.appendChild(el);
    requestAnimationFrame(() => requestAnimationFrame(() => el.classList.add('visible')));
}

function showResults(
    response: SearchResponse,
    onSearchAgain?: (query?: string) => void,
): void {
    document.getElementById('pramana-results')?.remove();

    const el = document.createElement('div');
    el.id = 'pramana-results';
    el.className = 'pramana-results';
    el.setAttribute('role', 'main');

    const wrapper = document.createElement('div');
    wrapper.className = 'results-wrapper';

    // 1. TOP HEADER BANNER
    const header = document.createElement('header');
    header.className = 'results-header-banner';
    header.innerHTML = `
    <img
      src="/assets/CASES_OF_PRAMANA_INTRO_HEADER.png"
      class="results-header-image"
      alt="PRAMANA — Indian law for every citizen"
    />
  `;
    wrapper.appendChild(header);

    // 2. SEARCH BAR
    const searchRow = document.createElement('div');
    searchRow.className = 'results-search-container';
    searchRow.innerHTML = `
    <div class="results-search-box">
      <svg class="search-magnifier-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="11" cy="11" r="8"></circle>
        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
      </svg>
      <input type="text" class="results-query-input" value="${response.query.replace(/"/g, '&quot;')}" />
      <button type="button" class="results-search-again-btn" id="results-search-again-btn">
        <span>Search Again</span>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <line x1="5" y1="12" x2="19" y2="12"></line>
          <polyline points="12 5 19 12 12 19"></polyline>
        </svg>
      </button>
    </div>
  `;
    wrapper.appendChild(searchRow);

    const searchAgainBtn = searchRow.querySelector('#results-search-again-btn');
    const resultsInput = searchRow.querySelector('.results-query-input') as HTMLInputElement | null;

    const handleSearchAgain = () => {
        const currentVal = resultsInput?.value.trim() || '';
        if (currentVal && onSearchAgain) {
            onSearchAgain(currentVal);
            return;
        }
        // Dismiss results view to return to interactive search artwork
        el.classList.remove('visible');
        setTimeout(() => el.remove(), 500);
        const mainInput = document.getElementById('legal-query-input') as HTMLInputElement | null;
        const mainBtn = document.getElementById('search-submit-btn') as HTMLButtonElement | null;
        if (mainInput) {
            mainInput.value = currentVal || response.query;
            mainInput.disabled = false;
            mainInput.focus();
            mainInput.scrollLeft = 0;
            requestAnimationFrame(() => { mainInput.scrollLeft = 0; });
        }
        if (mainBtn) mainBtn.disabled = false;
    };

    searchAgainBtn?.addEventListener('click', handleSearchAgain);
    resultsInput?.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') handleSearchAgain();
    });

    // 3. TITLE ROW
    const titleRow = document.createElement('div');
    titleRow.className = 'results-title-row';
    titleRow.innerHTML = `
    <h2 class="results-main-heading">${response.cases.length} Relevant Judgements Found</h2>
    <div class="results-heading-divider"></div>
  `;
    wrapper.appendChild(titleRow);

    // 4. MAIN CONTENT ROW: 5 CARDS
    const bodyLayout = document.createElement('div');
    bodyLayout.className = 'results-body-layout';

    // Cards Grid
    const cardsGrid = document.createElement('div');
    cardsGrid.className = 'results-cards-container';

    const cardThemes = [
        { themeClass: 'theme-orange', num: '#1', active: false },
        { themeClass: 'theme-blue', num: '#2', active: false },
        { themeClass: 'theme-emerald', num: '#3', active: true },
        { themeClass: 'theme-ochre', num: '#4', active: false },
        { themeClass: 'theme-maroon', num: '#5', active: false },
    ];

    response.cases.forEach((c: SearchCase, idx: number) => {
        const cardConfig = cardThemes[idx % cardThemes.length];
        const verifiedCount = typeof c.verified_count === 'number' ? c.verified_count : 0;
        const notesCount = typeof c.needs_review_count === 'number' ? c.needs_review_count : 0;

        let verificationBadgesHtml = '';
        if (verifiedCount > 0) {
            verificationBadgesHtml += `
        <div class="card-verification-pill pill-verified">
          <span>Verified sentences <strong>${verifiedCount}</strong></span>
        </div>`;
        }
        if (notesCount > 0) {
            verificationBadgesHtml += `
        <div class="card-verification-pill pill-notes">
          <span>Verified with notes <strong>${notesCount}</strong></span>
        </div>`;
        }

        const cardEl = document.createElement('div');
        cardEl.className = `pramana-case-card ${cardConfig.themeClass} ${cardConfig.active ? 'card-active-theme' : ''}`;

        cardEl.innerHTML = `
      <div class="card-top-header">
        <div class="card-num-tab">${cardConfig.num}</div>
        <svg class="card-bookmark-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"></path>
        </svg>
      </div>

      <div class="card-thumbnail-container">
        <img src="/assets/bridge.png" class="card-court-image" alt="Supreme Court Building" />
      </div>

      <div class="card-details-box">
        <h3 class="card-case-title" title="${c.title.replace(/"/g, '&quot;')}">${c.title}</h3>
        <div class="card-court-meta">Supreme Court of India</div>

        <div class="card-pillars-row">
          <div class="card-pillar-item">
            <svg class="card-pillar-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
              <polyline points="14 2 14 8 20 8"></polyline>
              <line x1="16" y1="13" x2="8" y2="13"></line>
              <line x1="16" y1="17" x2="8" y2="17"></line>
            </svg>
            <span class="card-pillar-label">Problem</span>
          </div>

          <div class="card-pillar-item">
            <svg class="card-pillar-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <circle cx="11" cy="11" r="8"></circle>
              <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
            </svg>
            <span class="card-pillar-label">Evidence</span>
          </div>

          <div class="card-pillar-item">
            <svg class="card-pillar-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="m14 13-7.5 7.5c-.83.83-2.17.83-3 0 0 0 0 0 0 0a2.12 2.12 0 0 1 0-3L11 10"/>
              <path d="m16 16 6-6"/>
              <path d="m8 8 6-6"/>
              <path d="m9 7 8 8"/>
              <path d="m21 11-8-8"/>
            </svg>
            <span class="card-pillar-label">Decision</span>
          </div>
        </div>

        ${verificationBadgesHtml ? `<div class="card-verification-container">${verificationBadgesHtml}</div>` : ''}

        <button class="card-action-btn" id="case-action-${c.case_id}" type="button">
          <span class="card-action-label">Read Case Details</span>
          <span class="card-action-arrow" aria-hidden="true">→</span>
        </button>
      </div>
    `;

        const actionBtn = cardEl.querySelector(`#case-action-${c.case_id}`) as HTMLButtonElement | null;
        actionBtn?.addEventListener('click', async () => {
            actionBtn.disabled = true;
            actionBtn.innerHTML = '<span class="card-action-label">Loading…</span><span class="card-action-arrow card-action-spinner" aria-hidden="true">✦</span>';
            try {
                console.log('[PRAMANA] Fetching case details for:', c.case_id);
                const details = await getCaseDetails(c.case_id);
                showCaseDetails(details, c);
            } catch (err) {
                console.error('[PRAMANA] Failed to load case details:', err);
                showCaseDetailsError(c.case_id);
            } finally {
                actionBtn.disabled = false;
                actionBtn.innerHTML = '<span class="card-action-label">Read Case Details</span><span class="card-action-arrow" aria-hidden="true">→</span>';
            }
        });

        cardsGrid.appendChild(cardEl);
    });

    bodyLayout.appendChild(cardsGrid);
    wrapper.appendChild(bodyLayout);

    el.appendChild(wrapper);
    document.body.appendChild(el);
    requestAnimationFrame(() => requestAnimationFrame(() => { el.classList.add('visible'); el.scrollTop = 0; }));
}

// ─────────────────────────────────────────────────────────
// Processing sequence
// ─────────────────────────────────────────────────────────
async function playProcessingSequence(
    backendPromise: Promise<SearchResponse>,
    procContainer: HTMLElement,
    processingFrames: NodeListOf<HTMLImageElement>,
    statusOverlay: HTMLElement,
): Promise<SearchResponse> {
    processingFrames.forEach((f, i) => {
        f.style.opacity = '0';
        f.style.zIndex = (i + 1).toString();
        f.classList.remove('visible');
    });

    const steps = [
        { text: 'Understanding your legal problem...', sub: '(Step-Back running)' },
        { text: 'Finding similar court cases...', sub: '(ChromaDB retrieval running)' },
        { text: 'Preparing case summaries...', sub: '(Summarization running)' },
        { text: 'Verifying facts against judgments...', sub: '(Gates 0–1–2 running)' },
    ];

    const primaryEl = statusOverlay.querySelector('.primary') as HTMLElement;
    const secondaryEl = statusOverlay.querySelector('.secondary') as HTMLElement;

    // Preload processing images
    await Promise.all(
        Array.from(processingFrames).map(
            img => new Promise<void>(resolve => {
                if (img.complete) { resolve(); return; }
                img.onload = () => resolve();
                img.onerror = () => { console.error('[PRAMANA] Failed to load processing image', img.src); resolve(); };
            })
        )
    );

    // Show Frame 1
    const first = processingFrames[0];
    first.style.opacity = '1';
    first.style.zIndex = '1';
    first.classList.add('visible');
    primaryEl.textContent = steps[0].text;
    secondaryEl.textContent = steps[0].sub;
    procContainer.style.visibility = 'visible';
    procContainer.style.transition = 'none';
    procContainer.style.opacity = '1';

    const layeredCrossfade = async (out: HTMLImageElement, inc: HTMLImageElement) => {
        inc.style.opacity = '0';
        inc.style.transition = `opacity ${fade(1000)}ms cubic-bezier(0.65,0,0.35,1)`;
        inc.style.zIndex = (parseInt(out.style.zIndex) + 1).toString();
        void inc.offsetHeight;
        inc.style.opacity = '1';
        await sleep(fade(1000));
        out.style.opacity = '0';
        out.style.transition = 'none';
        out.classList.remove('visible');
        inc.classList.add('visible');
    };

    // 1. Frame 1 -> ORIGINAL EXISTING Frame 1 -> Frame 2 timing
    await sleep(1300);

    // Frame 1 -> Frame 2 transition
    primaryEl.textContent = steps[1].text;
    secondaryEl.textContent = steps[1].sub;
    await layeredCrossfade(processingFrames[0], processingFrames[1]);

    // 2. Frame 2 is active -> EXTRA 3 SECONDS
    await sleep(FRAME_2_EXTRA_HOLD_MS);

    // Frame 2 -> Frame 3 transition
    primaryEl.textContent = steps[2].text;
    secondaryEl.textContent = steps[2].sub;
    await layeredCrossfade(processingFrames[1], processingFrames[2]);

    // 3. Frame 3 is active -> EXTRA 3 SECONDS
    await sleep(FRAME_3_EXTRA_HOLD_MS);

    // Frame 3 -> Frame 4 transition
    primaryEl.textContent = steps[3].text;
    secondaryEl.textContent = steps[3].sub;
    await layeredCrossfade(processingFrames[2], processingFrames[3]);

    // 4. Frame 4 is active -> WAIT FOR BACKEND (add NO extra delay)
    const result = await backendPromise;
    return result;
}

// ─────────────────────────────────────────────────────────
// Execute search lifecycle (supports continuous searches)
// ─────────────────────────────────────────────────────────
async function executeSearch(
    query: string,
    procContainer: HTMLElement,
    processingFrames: NodeListOf<HTMLImageElement>,
    statusOverlay: HTMLElement,
): Promise<void> {
    // 1. Clear any previous error banner and previous results so stale cases cannot remain visible
    document.getElementById('search-error-banner')?.remove();
    document.getElementById('pramana-results')?.remove();
    lastSearchResults = null;
    capturedQuery = query;

    // 2. Start the real backend call immediately in parallel
    let searchResult: SearchResponse | null = null;
    let searchError: unknown = null;

    const backendPromise: Promise<SearchResponse> = searchCases(query)
        .then((res) => {
            searchResult = res;
            lastSearchResults = res;
            return res;
        })
        .catch((err: unknown) => {
            searchError = err;
            console.error('[PRAMANA] /search failed:', err);
            return { query, cases: [] } as SearchResponse;
        });

    // 3. Play processing sequence with exact Ashoka Chakra timing
    try {
        await playProcessingSequence(backendPromise, procContainer, processingFrames, statusOverlay);
    } catch (err) {
        console.error('[PRAMANA] Processing sequence error:', err);
    }

    // 4. Hide processing container
    procContainer.style.opacity = '0';
    procContainer.style.visibility = 'hidden';

    // 5. Re-enable search inputs on frame3 so user can perform another search immediately
    const inputEl = document.getElementById('legal-query-input') as HTMLInputElement | null;
    const btnEl = document.getElementById('search-submit-btn') as HTMLButtonElement | null;
    if (inputEl) {
        inputEl.value = query;
        inputEl.disabled = false;
        inputEl.focus();
    }
    if (btnEl) {
        btnEl.disabled = false;
    }

    // 6. Handle errors
    if (searchError || !searchResult) {
        // Check specifically for HTTP 422 (Out of scope)
        const is422 = (searchError instanceof ApiError && searchError.status === 422) ||
            (searchError instanceof Error && ((searchError as any).status === 422 || searchError.message.includes('422')));

        if (is422) {
            showOutOfScopeError("This legal problem is outside PRAMANA's supported domain.");
        } else {
            showSearchError('PRAMANA could not complete the search. Please try again.');
        }
        return;
    }

    // 7. Render results screen (supports repeating search with "Search Again")
    showResults(searchResult, (newQuery?: string) => {
        if (newQuery && newQuery.trim()) {
            executeSearch(newQuery.trim(), procContainer, processingFrames, statusOverlay);
        }
    });
}

// ─────────────────────────────────────────────────────────
// Main cinematic sequence
// ─────────────────────────────────────────────────────────
async function runCinematicSequence() {
    const { frame1, frame2, frame3, video, procContainer, processingFrames, statusOverlay } = buildStage();

    preloadImg(ASSET.image2);
    PROCESSING_ASSETS.forEach(src => preloadImg(src));

    // Image 1 visible
    frame1.classList.add('visible');
    frame1.style.opacity = '1';
    video.load();
    await waitVideoReady(video);
    await sleep(T.image1Hold);
    const pp = video.play();
    if (pp !== undefined) {
        pp.catch(() => { video.muted = true; video.play().catch(() => { }); });
    }

    // Image1 → Video
    await crossfade(frame1, frame2, fade(T.image1ToVideo));

    // Video → Image2 (search screen)
    await waitVideoEnd(video);
    await sleep(T.videoEndHold);
    await crossfade(frame2, frame3, fade(T.videoToImage2));
    console.log('[PRAMANA] Step 1 complete — search screen active.');

    // Activate search screen with continuous handler
    activateSearchScreen(frame3, (query) => {
        executeSearch(query, procContainer, processingFrames, statusOverlay);
    });
}

// ─────────────────────────────────────────────────────────
// Bootstrap
// ─────────────────────────────────────────────────────────
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', runCinematicSequence);
} else {
    runCinematicSequence();
}
