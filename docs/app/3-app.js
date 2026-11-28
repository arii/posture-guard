// Author: Ariel Anders
// Copyright (c) 2026 Ariel Anders
// SPDX-License-Identifier: MIT
//
// Browser shell for BackUp: webcam + MediaPipe Pose Landmarker in the
// page, posture-core.js for scoring/state/needle/calibration, Web Audio for
// chimes, recorded Kokoro clips (speechSynthesis fallback) for speech, localStorage for settings,
// calibration and daily totals. No frame or landmark leaves the browser.

import { ACTIVITY, ActivityMachine, OwedReminder } from "./activity.js";
import { CUE_SHORT, DEFAULTS, LEFT_EAR, Nudge, PostureEngine, Reason, State, TRACKED } from "./posture-core.js";
import {
  BREAK_LINES,
  BREATHE_LINES,
  DEFAULT_PROMPT,
  PRAISE,
  RECOVERY_PRAISE,
  RELAX_LINES,
  TEST_LINE,
  VOICES,
} from "./voice-lines.js";
import { BracingMonitor, BreathCheck, neckRatio, SHOULDER_OFF_REASON, shoulderChecksOn } from "./relax.js";
import { createMoveNetDetector } from "./movenet-adapter.js";
import { LoadSteps, downloadAll, nextPaint } from "./loading.js";
import { SOUND_PACKS, playSpooky } from "./sound-packs.js";
import { BREAKS, BreakTracker } from "./breaks.js";
import { BreatheGuide, BreathingMoments, GUIDE_SAFETY } from "./breathing.js";
import { PromptScheduler } from "./prompts.js";
import { angleTransform, gaugeView, glideAngle, tileStates, ZONE_COLOR } from "./mini-view.js";
import { SessionTimeline } from "./timeline.js";
import { demoPose, practicePose } from "./demo-pose.js";
import {
  calibrateNote,
  DIAGNOSIS_TEXT,
  Diagnosis,
  markOnboarded,
  needsAckOnly,
  ONBOARDING_STEPS,
  PRACTICE_STEPS,
  RELAX_SAFETY,
  SAFETY_POINTS,
  SetupCheck,
  shouldShowOnboarding,
} from "./onboarding.js";
import { consistencyScore } from "./consistency.js";
import {
  applyImport,
  dayKeyFor,
  exportJson,
  IMPORT_MAX_BYTES,
  migrateStore,
  PROGRESS,
  countEpisodes,
  orDetached,
  parseImport,
  previewImport,
  summarize,
  summarizeWeeks,
  tallyDay,
} from "./progress.js";
import {
  CACHE_PREFIX,
  cacheNameFor,
  isOfflineReady,
  META_URL,
  offlineMessage,
  offlineState,
  progressPercent,
} from "./offline.js";
import { addWater, DEMO_DONE, demoAt, demoLength, demoStaticText, toolSafety } from "./wellness.js";
import {
  NEEDS_TRACKING_TIP,
  cameraConstraints,
  cameraErrorMessage,
  cameraProblem,
  isDeviceGoneError,
  isPauseShortcut,
  minutesLabel,
  nextOption,
  resolveSettings,
} from "./settings.js";
import { BUG_LINK_TIP, buildBugReportBody, clearStoredData, sanitizeError, webglSummary } from "./privacy.js";
import {
  CONSENT_CHECKBOX,
  CONSENT_POINTS,
  CONSENT_TITLE,
  hasConsent,
  hasShareConsent,
  recordConsent,
  recordShareConsent,
  SHARE_CONSENT_CHECKBOX,
  SHARE_CONSENT_POINTS,
  SHARE_CONSENT_TITLE,
  withdrawConsent,
  withdrawShareConsent,
} from "./consent.js";
import {
  SHARE_TEXT,
  cardAltText,
  cleanNickname,
  peerLink,
  peerRecord,
  readPeerLink,
  drawScoreCard,
  nativeSharePayload,
  scoreCardData,
  normalizeInstance,
  shareLinks,
  shareUrl,
} from "./share.js";
import {
  SPONSOR,
  SPONSOR_URL,
  answerSponsor,
  markSponsorShown,
  refundSponsorShown,
  recordSession,
  shouldOfferSponsor,
  sponsorText,
} from "./sponsor.js";
import {
  completeStep,
  nextSetupStep,
  SETUP_STEPS,
  setupProgress,
  skipStep,
  undoStep,
  viewSetupStep,
} from "./onboarding.js";
import {
  nextUnlockCard,
  RevealStages,
  STAGES,
  trackedDays,
  trackedSecondsFromDays,
  unlockLevel,
  visibleRows,
} from "./reveal.js";
import { FeedbackPolicy, MILESTONE_TEXT, normalizeLevel, recordAnnoyance, tabTitleFor } from "./feedback.js";

// Self-hosted under docs/app/vendor/ (see vendor/README.md for versions and hashes).
const TASKS_VISION = new URL("./vendor", import.meta.url).href;
const MODEL_URL = `${TASKS_VISION}/pose_landmarker_lite.task`;
// Fallback for browsers without WebGL: TF.js MoveNet on the WASM backend, fetched only when needed.
const MOVENET_DIR = `${TASKS_VISION}/movenet`;
const VISIBLE_FPS = 15; // detection rate while the tab is visible
const HIDDEN_TICK_MS = 200; // ~5 fps from a worker timer while hidden
const STORE = "posture-guard:v1";

const $ = (id) => document.getElementById(id);
// Elements added in the week/episodes release: if a cached older index.html is paired with this script,
// writes go to a detached node instead of throwing and breaking the rest of the render.
const $soft = (id) => orDetached($(id), () => document.createElement("div"));
const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");

// -- persistence --------------------------------------------------------------

function loadStore() {
  try {
    return JSON.parse(localStorage.getItem(STORE)) || {};
  } catch {
    return {};
  }
}
const store = loadStore();
migrateStore(store); // schema 2: day keys are local dates from now on (old UTC keys stay as they are)
if (store.firstRunDone) store.setupDone = true; // set up before the sequence existed: nothing to show again
let deleted = false; // "Delete my data" ran: never write the store back
function save() {
  if (deleted) return;
  try {
    localStorage.setItem(STORE, JSON.stringify(store));
  } catch {
    /* quota: keep in memory */
  }
}
const todayKey = () => dayKeyFor(store, Date.now());
function todayRecord() {
  store.days = store.days || {};
  store.days[todayKey()] = store.days[todayKey()] || { good: 0, bad: 0 };
  return store.days[todayKey()];
}

// -- settings -----------------------------------------------------------------

const settings = {
  mode: store.mode || "chime_and_voice",
  voiceText: store.voiceText || DEFAULT_PROMPT,
  specificCues: store.specificCues !== false, // default on
  spokenPraise: store.spokenPraise === true, // default off: opt in
  soundPack: SOUND_PACKS[store.soundPack] ? store.soundPack : "clean",
  voiceDelay: DEFAULTS.state.voice_delay_s,
  voiceRepeat: DEFAULTS.state.voice_repeat_cooldown_s,
};

// -- activity ---------------------------------------------------------------------
// One answer to "what is the user doing, and what may the app do right now?"
// (activity.js). Pause, silent test, practice, breath check, breathing guide, eye
// break and demos all go through it; the per-frame code asks rules() instead of
// reading flags. `test` and `practice` below only hold their progress data.

const activity = new ActivityMachine();
const clockS = () => performance.now() / 1000;
const rules = () => activity.policy(clockS());
const isPaused = () => activity.isActive(ACTIVITY.PAUSED);
const MODE_STATES = new Set([ACTIVITY.PAUSED, ACTIVITY.SILENT_TEST, ACTIVITY.PRACTICE]);
/** A pause, silent test or practice is running: the prompt and relax loops stand still. */
const inMode = () => MODE_STATES.has(activity.current);
const BREAK_STATES = new Set([ACTIVITY.EYE_BREAK, ACTIVITY.DEMO]);
/** A new foreground activity may begin: nothing is running, or only a break that it replaces. */
const canBegin = (replaceBreaks = false) =>
  activity.current === ACTIVITY.TRACKING || (replaceBreaks && BREAK_STATES.has(activity.current));
const MODE_RANK = { chime_and_voice: 2, chime_only: 1, silent: 0 };
settings.coachingLevel = normalizeLevel(store.coachingLevel);
const policy = new FeedbackPolicy(settings.coachingLevel);
// Muting counts toward the Light suggestion (feedback-design 7.4); Light is now a fixed default
// for whoever already chose it, so the log is kept but no suggestion is shown (#98 v3).
function noteAnnoyance() {
  store.annoyLog = recordAnnoyance(store.annoyLog, Date.now());
  save();
}

function allowed(nudge) {
  if (settings.mode === "silent") return false;
  return !(settings.mode === "chime_only" && nudge === Nudge.VOICE);
}

// -- engine -------------------------------------------------------------------

const engine = new PostureEngine(DEFAULTS, {
  voiceTimings: () => [settings.voiceDelay, settings.voiceRepeat],
});
if (store.calibration) engine.calibration.load(store.calibration);
let savedCalVersion = engine.calibration.version;

// -- audio --------------------------------------------------------------------

let audioCtx = null;
const custom = { bad: null, good: null }; // AudioBuffers

function ensureAudio() {
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === "suspended") audioCtx.resume();
  return audioCtx;
}

// harmonics: [[k, amplitude], ...] for a custom timbre; omitted = pure sine.
function tones(freqs, seconds, harmonics = null) {
  const ac = ensureAudio();
  let wave = null;
  if (harmonics) {
    const n = Math.max(...harmonics.map(([k]) => k)) + 1;
    const real = new Float32Array(n);
    const imag = new Float32Array(n); // sine terms
    for (const [k, amp] of harmonics) imag[k] = amp;
    wave = ac.createPeriodicWave(real, imag); // peak-normalized
  }
  let t = ac.currentTime + 0.02;
  freqs.forEach((f, i) => {
    const dur = seconds[i];
    const osc = ac.createOscillator();
    const gain = ac.createGain();
    if (wave) osc.setPeriodicWave(wave);
    else osc.type = "sine";
    osc.frequency.value = f;
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.exponentialRampToValueAtTime(0.5, t + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + dur); // ~exp(-3t) decay
    osc.connect(gain).connect(ac.destination);
    osc.start(t);
    osc.stop(t + dur + 0.02);
    t += dur;
  });
}

function playBuffer(buffer) {
  const ac = ensureAudio();
  const src = ac.createBufferSource();
  src.buffer = buffer;
  src.connect(ac.destination);
  src.start();
}

// A falling "uh-oh" reminder (E5 -> B-flat 4, a tritone
// down, reedy odd harmonics) and a rising C-major arpeggio reward.
const BAD_HARMONICS = [
  [1, 1.0],
  [3, 0.35],
  [5, 0.15],
];
// An uploaded sound wins, then the chosen pack (clean by default).
const playBad = () =>
  custom.bad
    ? playBuffer(custom.bad)
    : settings.soundPack === "spooky"
      ? playSpooky(ensureAudio(), "bad")
      : tones([659.25, 466.16], [0.18, 0.42], BAD_HARMONICS);
const playGood = () =>
  custom.good
    ? playBuffer(custom.good)
    : settings.soundPack === "spooky"
      ? playSpooky(ensureAudio(), "good")
      : tones([523.25, 659.25, 783.99, 1046.5], [0.2, 0.2, 0.2, 0.2]);

// -- voice --------------------------------------------------------------------
// Default: clips pre-rendered with Kokoro-82M (docs/app/voice, see
// voice-lines.js), which sound like a person. Lines without a clip (a custom
// prompt) and the "System:" choices use speechSynthesis; for those, rank the
// installed voices and avoid OS fallbacks like eSpeak, which sound robotic.

const ROBOTIC = /espeak|e-speak|mbrola|festival|pico|flite|rhvoice/i;
function voiceScore(v, lang) {
  let s = 0;
  if (/natural|neural|online/i.test(v.name)) s += 8;
  if (/^google/i.test(v.name)) s += 6;
  if (/premium|enhanced|siri/i.test(v.name)) s += 5;
  if (/samantha|ava|allison|karen|daniel|serena|aria|jenny|guy|libby|sonia/i.test(v.name)) s += 2;
  if (!v.localService) s += 1; // network voices are usually the neural ones
  if (ROBOTIC.test(v.name)) s -= 10;
  if (v.lang && v.lang.toLowerCase() === lang.toLowerCase()) s += 3;
  else if (v.lang && v.lang.slice(0, 2).toLowerCase() === lang.slice(0, 2).toLowerCase()) s += 2;
  else s -= 20; // wrong language: never pick it automatically
  return s;
}

const KOKORO = "kokoro:";
let clipManifest = null; // { voices: { af_heart: { text: "af_heart/x.mp3" } } }
const clipCache = new Map(); // url -> Promise<AudioBuffer>
let clipSource = null; // the clip playing now, so a new line can cut it off
let voices = [];

fetch("./voice/manifest.json")
  .then((r) => (r.ok ? r.json() : null))
  .then((m) => {
    clipManifest = m?.voices ? m : null;
    refreshVoices();
  })
  .catch(() => {});

const kokoroVoices = () => Object.keys(VOICES).filter((id) => clipManifest?.voices[id]);
/** Selected voice: "kokoro:<id>" or a system voice name. Default: first Kokoro voice. */
function selectedVoice() {
  const want = store.voiceName;
  if (want?.startsWith(KOKORO) && kokoroVoices().includes(want.slice(KOKORO.length))) return want;
  if (want && voices.some((v) => v.name === want)) return want;
  return kokoroVoices().length ? KOKORO + kokoroVoices()[0] : (voices[0]?.name ?? "");
}
function chosenVoice() {
  // best or selected system voice (for fallback speech)
  const sel = selectedVoice();
  return voices.find((v) => v.name === sel) || voices[0] || null;
}

function refreshVoices() {
  const lang = navigator.language || "en-US";
  voices =
    "speechSynthesis" in window
      ? window.speechSynthesis
          .getVoices()
          .slice()
          .sort((a, b) => voiceScore(b, lang) - voiceScore(a, lang))
      : [];
}
if ("speechSynthesis" in window) {
  refreshVoices();
  window.speechSynthesis.addEventListener?.("voiceschanged", refreshVoices);
}

function loadClip(url) {
  if (!clipCache.has(url)) {
    clipCache.set(
      url,
      fetch(url)
        .then((r) => r.arrayBuffer())
        .then((b) => ensureAudio().decodeAudioData(b)),
    );
  }
  return clipCache.get(url);
}

let postureSpeech = false; // the line playing now is a posture reminder, not a break instruction

function stopSpeech() {
  postureSpeech = false;
  if ("speechSynthesis" in window) window.speechSynthesis.cancel();
  if (clipSource) {
    try {
      clipSource.stop();
    } catch {
      /* already ended */
    }
    clipSource = null;
  }
}

function speakSystem(text) {
  if (!("speechSynthesis" in window)) return;
  const u = new SpeechSynthesisUtterance(text);
  const v = chosenVoice();
  if (v) {
    u.voice = v;
    u.lang = v.lang;
  }
  window.speechSynthesis.speak(u);
}

function speak(text, posture = false) {
  if (!rules().speak) return;
  postureSpeech = posture; // a line queued just before a pause, silent test or practice must not play
  stopSpeech();
  const sel = selectedVoice();
  const file = sel.startsWith(KOKORO) ? clipManifest?.voices[sel.slice(KOKORO.length)]?.[text] : null;
  document.dispatchEvent(new CustomEvent("pg:speak", { detail: { text, engine: file ? "kokoro" : "system" } }));
  if (!file) return speakSystem(text);
  loadClip(`./voice/${file}`)
    .then((buffer) => {
      if (!rules().speak) return;
      const ac = ensureAudio();
      const src = ac.createBufferSource();
      src.buffer = buffer;
      src.connect(ac.destination);
      src.onended = () => {
        if (clipSource === src) clipSource = null;
      };
      clipSource = src;
      src.start();
    })
    .catch(() => speakSystem(text)); // clip missing or undecodable
}

let praiseIndex = 0;
const nextPraise = () => PRAISE[praiseIndex++ % PRAISE.length];
let recoveryIndex = 0;
const nextRecovery = () => RECOVERY_PRAISE[recoveryIndex++ % RECOVERY_PRAISE.length];

const BANNERS = {
  [Nudge.POSITIVE_CHIME]: "Nice posture!",
  [Nudge.RECOVERY]: "Back to good posture",
};
let reminderCount = 0;
let bannerTimer = null;
let praiseTimer = null; // the spoken half of a praise line, queued after the arpeggio

/** Banner text for a reminder: what is off, when the engine has a cue for this
 *  frame, "Sit up" otherwise. */
function reminderText(nudge, cue) {
  // One short cue on the video (K5, #98); the voice says the full fix.
  if (nudge === Nudge.CHIME || nudge === Nudge.VOICE) return (cue && CUE_SHORT[cue.metric]) || "Sit up";
  return BANNERS[nudge];
}

let episode = 0; // counts slouch episodes, for the coaching policy
let lastBad = false;

// The hourly reminder cap (feedback-design) still applies; its note went with the
// Reminders & sounds panel (#98 v3), so there is nothing to show.
function showReminderNote() {}

function fire(nudge, cue) {
  const reminder = nudge === Nudge.CHIME || nudge === Nudge.VOICE;
  const allow = rules();
  if (!(reminder ? allow.reminders : allow.praise)) return; // silent test, practice, breath check, a break or its grace
  const nowS = performance.now() / 1000;
  if (reminder) {
    const verdict = policy.onReminder(nowS, { episodeId: episode, voice: nudge === Nudge.VOICE });
    if (verdict.capped) {
      showReminderNote(true);
      return;
    } // too many in an hour: stay quiet
    if (nudge === Nudge.VOICE && !verdict.voice) nudge = Nudge.CHIME; // light: chime instead of voice
    reminderCount += 1;
    prompts.notePosture(pclock); // a posture reminder owns the screen: other prompts wait
  } else if (nudge === Nudge.POSITIVE_CHIME && !policy.onPraise("timed")) {
    return; // light coaching: no timed praise
  }
  const praise = nudge === Nudge.POSITIVE_CHIME ? nextPraise() : nudge === Nudge.RECOVERY ? nextRecovery() : null;
  const text = praise || reminderText(nudge, cue);
  const banner = $("banner");
  banner.textContent = text;
  banner.classList.remove("cut");
  banner.classList.toggle("positive", !reminder);
  banner.classList.add("show");
  clearTimeout(bannerTimer);
  bannerTimer = setTimeout(() => banner.classList.remove("show"), cue ? 5000 : 2500);
  if (!allowed(nudge)) return; // silent / chime-only: still a visible event
  if (nudge === Nudge.VOICE) speak(settings.specificCues && cue ? cue.say : settings.voiceText, true);
  else if (nudge === Nudge.CHIME) playBad();
  else {
    playGood();
    // Spoken praise rides on "Chime + voice": after the arpeggio (~0.8 s).
    if (praise && settings.spokenPraise && settings.mode === "chime_and_voice") {
      clearTimeout(praiseTimer);
      praiseTimer = setTimeout(
        () => {
          if (rules().praise) speak(praise); // a break may have started since
        },
        custom.good ? 400 : 850,
      );
    }
  }
}

// -- calibration UI -----------------------------------------------------------

// "Save my posture again" (Settings) records the good posture; the first-run card does the
// same. While it records, and once it is saved, a short line shows on the video.
let calibWasActive = false;
function renderCalibration() {
  const c = engine.calibration.status;
  // #107: a countdown while it records, then what changed.
  if (c.active && !firstRun.open) {
    const left = Math.max(1, Math.ceil(c.remaining_s ?? 0));
    const text = `Hold still, saving your posture… ${left}`;
    if ($("banner").textContent !== text || !calibWasActive) showRelaxBanner(text);
  }
  if (!c.active && calibWasActive && !firstRun.open) {
    showRelaxBanner(
      c.error
        ? "Couldn't see you well enough. Try again"
        : demo
          ? "Posture saved for this demo only"
          : "Posture saved. Reminders now measure from it; your stats are kept",
    );
  }
  calibWasActive = !!c.active;
  if (engine.calibration.version !== savedCalVersion) {
    savedCalVersion = engine.calibration.version;
    // A posture saved on the demo's scripted poses is not the person's: keep it for this
    // session only, so a later camera session doesn't measure from it.
    if (c.good || c.bad) {
      if (!demo)
        store.calibration = { good: c.good, bad: c.bad, good_samples: c.good_samples, bad_samples: c.bad_samples };
    } else delete store.calibration;
    save();
  }
}

// -- session stats ------------------------------------------------------------

const session = { good: 0, bad: 0, streak: 0, best: 0, lastT: null };
store.days = store.days || {};

// -- relax, don't brace -----------------------------------------------------------
// Good posture should be a relaxed stack, not a squeeze. relax.js watches the
// ears-to-shoulders gap for shoulders creeping up while posture reads good,
// and runs the breath check ("if your shoulders rise when you breathe in, you
// are bracing"). See relax.js for the measurement.

// -- consent for shoulder and breathing features -------------------------------------
// Breath check, "Breathe with me", breathing moments and shoulder-creep
// reminders read shoulder, neck and nose landmarks. They stay off until the
// user opts in (a dated record in the store, see consent.js). Posture tracking
// itself does not need it.

let consentThen = null; // runs after the user agrees
/** True when consent exists. Otherwise opens the consent dialog and runs `then` after the user agrees. */
function requireConsent(then) {
  if (hasConsent(store)) return true;
  consentThen = then || null;
  $("consent-check").checked = false;
  $("consent-agree").disabled = true;
  $("consent-dialog").showModal();
  return false;
}
$("consent-title").textContent = CONSENT_TITLE;
$("consent-label").append(CONSENT_CHECKBOX);
$("consent-list").replaceChildren(
  ...CONSENT_POINTS.map((t) => {
    const li = document.createElement("li");
    li.textContent = t;
    return li;
  }),
);
$("consent-check").addEventListener("change", (e) => {
  $("consent-agree").disabled = !e.target.checked;
});
$("consent-agree").addEventListener("click", () => {
  recordConsent(store);
  save();
  $("consent-dialog").close();
  renderRows();
  const then = consentThen;
  consentThen = null;
  if (then) then();
});
$("consent-cancel").addEventListener("click", () => {
  consentThen = null;
  $("consent-dialog").close();
  renderRows();
});
$("consent-dialog").addEventListener("close", () => {
  if (!hasConsent(store)) renderRows();
});
/** Settings → Shoulders & breathing off: withdraw the OK and stop anything that uses it. */
function withdrawShoulders() {
  withdrawConsent(store);
  save();
  guide.cancel();
  breath.cancel?.();
  activity.stop(ACTIVITY.BREATH_CHECK, clockS(), "cancelled");
  if (card) hideCard();
  prompts.dismiss(pclock, "done");
  renderRows();
}

const bracing = new BracingMonitor();
const breath = new BreathCheck();
const AUTO_BREATH_S = 20 * 60; // coached good-posture time between automatic checks
const BREATH_INTRO_GOOD_S = 60; // good posture needed before the first breath check is offered
let goodSinceBreath = 0;
let relaxLastT = null;
Object.assign(settings, resolveSettings(store)); // relaxReminders and pauseShortcut: on for new users, stored choices kept

/** Relax lines are spoken unless the mode is silent (chime-only still talks
 *  here: a breath check without words is not a check). */
function sayRelax(line) {
  if (settings.mode !== "silent") speak(line);
}

function showRelaxBanner(text) {
  const banner = $("banner");
  banner.textContent = text;
  banner.classList.remove("positive", "cut");
  banner.classList.add("show", "relax");
  clearTimeout(bannerTimer);
  bannerTimer = setTimeout(() => banner.classList.remove("show", "relax"), 6000);
}

// The exercise safety line goes on the video with the check itself (Settings no longer has a panel for it).
const RELAX_SAFETY_SHORT = RELAX_SAFETY.slice(RELAX_SAFETY.indexOf("Stop"));

// One short line on the video after a breath check; the voice says the full verdict.
const VERDICT_SHORT = {
  rose: "Let your shoulders drop",
  held: "That lower spot is right",
  relaxed: "Relaxed and stacked",
  unclear: "Couldn't see your shoulders",
};

function startBreathCheck() {
  if (!loopsStarted || !canBegin()) return;
  if (!shoulderChecksOn(lowAccuracy)) {
    showRelaxBanner(SHOULDER_OFF_REASON); // the fallback model has no reliable shoulder signal
    return;
  }
  if (!requireConsent(startBreathCheck)) return;
  ensureAudio();
  runBreathCheck();
}

/** Begin the check itself (consent is settled). The breathing guide hands off to it. */
function runBreathCheck() {
  if (!activity.start(ACTIVITY.BREATH_CHECK, clockS())) return;
  goodSinceBreath = 0;
  breath.start(performance.now() / 1000, bracing.value);
  sayRelax(RELAX_LINES.inhale);
  showRelaxBanner(`Breathe in slowly. ${RELAX_SAFETY_SHORT}`);
}

function relaxStep(landmarks, state, now) {
  if (!shoulderChecksOn(lowAccuracy)) {
    relaxLastT = null;
    return;
  }
  if (inMode()) {
    relaxLastT = null;
    return;
  }
  if (!hasConsent(store)) {
    relaxLastT = null;
    return;
  }
  const ratio = neckRatio(landmarks);
  const allow = rules();
  const good = state === State.GOOD && allow.engine === "track"; // the held state says nothing about posture
  const flagged = bracing.update(ratio, now, good && !breath.active);
  if (flagged && allow.score === "day" && !breath.active) {
    const day = todayRecord();
    day.brace_events = (day.brace_events || 0) + 1;
  }
  if (flagged && settings.relaxReminders && allow.reminders) {
    showRelaxBanner(RELAX_LINES.bracing);
    sayRelax(RELAX_LINES.bracing);
    store.bracingCount = (store.bracingCount || 0) + 1;
    save();
    prompts.enqueue("offer", pclock); // offer "Breathe with me" (scheduler rules apply)
  }
  if (breath.active) {
    const step = breath.update(bracing.value, now);
    if (step === "exhale") {
      sayRelax(RELAX_LINES.exhale);
      showRelaxBanner("Breathe out, let your shoulders drop");
    } else if (step) {
      activity.stop(ACTIVITY.BREATH_CHECK, now, "completed");
      if (step.kind !== "unclear" && step.kind !== "rose") bracing.pinRelaxed(step.relaxed);
      sayRelax(RELAX_LINES[step.kind]);
      showRelaxBanner(VERDICT_SHORT[step.kind]); // the voice says the full verdict
    }
  } else if (good && settings.relaxReminders && relaxLastT !== null) {
    goodSinceBreath += Math.min(1, Math.max(0, now - relaxLastT));
    if (goodSinceBreath >= AUTO_BREATH_S) prompts.enqueue("breath", pclock); // the scheduler decides when
  }
  relaxLastT = now;
}

// Controls that need frames from the camera or the demo stay disabled until tracking starts.
// Settings rows that start something need frames from the camera or the demo.
const NEEDS_TRACKING = ["row-posture", "row-break", "break-btn", "test-start"];
function setTrackingControls(on) {
  for (const id of NEEDS_TRACKING) {
    const el = $(id);
    el.disabled = !on;
    if (!on) el.title = NEEDS_TRACKING_TIP;
    else el.removeAttribute("title");
  }
}

/** Settings → Shoulders & breathing on: ask first (consent dialog), then turn on the
 *  shoulder reminders and the automatic breath check. */
function setShoulders(on) {
  if (!on) {
    settings.relaxReminders = false;
    store.relaxReminders = false;
    withdrawShoulders();
    return;
  }
  const apply = () => {
    settings.relaxReminders = true;
    store.relaxReminders = true;
    save();
    renderRows();
  };
  if (!requireConsent(apply)) {
    renderRows(); // the switch stays off until the user agrees
    return;
  }
  apply();
}

// -- breaks and prompts ---------------------------------------------------------
// Everything that is not a posture reminder goes through one PromptScheduler
// (prompts.js): eye break (20-20-20), move break, breathing moments, the auto
// breath check and the "Breathe with me" offer. One card at a time, none in the
// first 5 min, at most one per 10 min, none while paused, in a silent test or
// during a breath check. Prompt timers run on a prompt clock that only advances
// while tracking is running (paused or hidden gaps never count).
//
// Testing hook: ?clockScale=N (1..1000) makes that prompt clock run N times
// faster, so an end-to-end test can see a 20 minute eye break in seconds. It
// scales only the prompt timers: tracking, reminders and countdowns stay real.

const CLOCK_SCALE = (() => {
  const n = Number(new URLSearchParams(location.search).get("clockScale"));
  return Number.isFinite(n) && n >= 1 && n <= 1000 ? n : 1;
})();
const DISMISS_KEY = {
  eye: "eye",
  move: "move",
  stretch: "stretch",
  water: "water",
  jaw: "jaw",
  breathing: "breathing",
  offer: "breathing",
};
const dismissed = settings.promptsDismissed;

function promptEnabled(id) {
  if (dismissed[DISMISS_KEY[id]]) return false;
  if (id === "eye") return settings.eyeBreaks;
  if (id === "move") return settings.moveBreakMinutes > 0;
  if (id === "stretch") return settings.stretchMinutes > 0;
  if (id === "water") return settings.waterMinutes > 0;
  if (id === "jaw") return settings.jawMinutes > 0;
  if (id === "breathing") return settings.breathingMoments && hasConsent(store);
  if (id === "breath" || id === "offer") return settings.relaxReminders && hasConsent(store);
  if (id === "breath_intro") return settings.relaxReminders && !store.breathIntroShown; // asks for consent itself
  return true;
}

const prompts = new PromptScheduler({ isEnabled: promptEnabled });
const breaks = new BreakTracker(settings);
const moments = new BreathingMoments();
const guide = new BreatheGuide();
let pclock = 0; // prompt clock (scaled by clockScale)
let preal = 0; // real seconds of running tracking (card timers, guide)
let plast = null;
let pstreak = 0; // good-posture streak on the prompt clock
let card = null; // the prompt on screen: { id, text, t, finished }

// Break reminders (Settings, from 2 min): one switch for the eye and move reminders. The
// intervals, stretch, water, jaw and breathing-moment options are fixed defaults (#98 v3).
function breakRemindersOn() {
  return settings.eyeBreaks || settings.moveBreakMinutes > 0;
}
function setBreakReminders(on) {
  settings.eyeBreaks = on;
  settings.moveBreakMinutes = on ? store.lastMoveMinutes || 60 : 0;
  if (!on && breaks.moveMinutes > 0) store.lastMoveMinutes = breaks.moveMinutes;
  if (on) {
    delete dismissed.eye;
    delete dismissed.move;
  }
  breaks.setEyeBreaks(settings.eyeBreaks);
  breaks.setMoveMinutes(settings.moveBreakMinutes);
  store.eyeBreaks = settings.eyeBreaks;
  store.moveBreakMinutes = settings.moveBreakMinutes;
  store.promptsDismissed = { ...dismissed };
  save();
  renderRows();
}

function breaksTakenToday() {
  const b = store.breaksToday;
  return b && b.day === todayKey() ? b.n : 0;
}
/** A finished eye, move, stretch or jaw break: reset its timer and count it for today. */
function tookBreak(kind) {
  breaks.taken(kind);
  countBreak();
}
function countBreak() {
  store.breaksToday = { day: todayKey(), n: breaksTakenToday() + 1 };
  const day = todayRecord();
  day.breaks = store.breaksToday.n;
  save();
}

/** Prompts speak only on "Chime + voice" (they are never required to be heard). */
function sayPrompt(line) {
  if (settings.mode === "chime_and_voice") speak(line);
}

const CARD_UI = {
  eye: { bar: true, snooze: true, never: true },
  move: { action: true, done: true, snooze: true, never: true },
  stretch: { action: true, actionLabel: "Show me", done: true, snooze: true, never: true },
  water: { action: true, actionLabel: "I drank", snooze: true, never: true },
  jaw: { done: true, snooze: true, never: true },
  demo: { bar: true, stop: "End break" },
  breathing: { action: true, never: true },
  offer: { action: true, never: true },
  breath_intro: { action: true, actionLabel: "Try it" },
  guide: { bar: true, stop: "End breathing" },
};

function renderCard() {
  const el = $("prompt-card");
  if (!card) {
    el.classList.remove("show");
    $("prompt-figure").hidden = true;
    el.dataset.prompt = "";
    return;
  }
  const ui = card.ui || CARD_UI[card.id];
  if (card.id !== "demo") $("prompt-figure").hidden = true;
  $("prompt-text").textContent = card.text;
  const safety = toolSafety(card.id, card.kind);
  $("prompt-safety").textContent = safety;
  $("prompt-safety").hidden = !safety;
  $("prompt-bar").hidden = !ui.bar;
  $("prompt-action").hidden = !ui.action;
  $("prompt-action").textContent = ui.actionLabel || "Breathe with me";
  $("prompt-done").hidden = !ui.done || card.finished;
  $("prompt-snooze").hidden = !ui.snooze || card.finished;
  $("prompt-never").hidden = !ui.never || card.finished;
  // A finished water card keeps a dismiss button, so "I didn't drink" ends it without counting a drink.
  $("prompt-notnow").hidden = card.finished && !ui.dismissible;
  $("prompt-notnow").textContent = ui.stop || "Not now"; // #111: it ends the break, not tracking
  el.dataset.prompt = card.id;
  el.classList.add("show");
}

function setCardFill(f) {
  const pct = Math.round(Math.min(1, Math.max(0, f)) * 100);
  $("prompt-fill").style.width = `${pct}%`;
  $("prompt-bar").setAttribute("aria-valuenow", String(pct));
}

const CARD_ACTIVITY = { eye: ACTIVITY.EYE_BREAK, demo: ACTIVITY.DEMO, guide: ACTIVITY.BREATHE_GUIDE };

/** Every card removal comes through here, so the activity a card stands for can never outlive it.
 *  Finished activities were already stopped as "completed"; this records the rest as cancelled. */
function hideCard() {
  const act = card && CARD_ACTIVITY[card.id];
  if (act) activity.stop(act, clockS(), "cancelled");
  card = null;
  renderCard();
}

/** The user (or a timeout) ends the card. how: done | snooze | notnow | never. */
function endCard(how) {
  if (card && card.id === "guide") {
    guide.cancel();
    hideCard();
    return;
  }
  if (card && card.id === "demo") {
    prompts.dismiss(pclock, "done");
    hideCard();
    return;
  }
  const id = prompts.dismiss(pclock, how);
  if (how === "never" && DISMISS_KEY[id]) {
    dismissed[DISMISS_KEY[id]] = true;
    savePromptSettings();
  }
  hideCard();
}

/** Drop whatever prompt is on screen (pause, silent test, practice). */
function cancelPrompts() {
  hideSponsor();
  guide.cancel();
  if (card) endCard("done");
  prompts.dismiss(pclock, "done");
}

function showPrompt(item) {
  if (item.id === "share") {
    showSponsor();
    return;
  }
  if (item.id === "breath") {
    // the auto breath check has its own UI
    prompts.dismiss(pclock, "done");
    startBreathCheck();
    return;
  }
  if (item.id === "breath_intro") {
    showBreathIntro();
    return;
  }
  const text =
    item.id === "eye" || item.id === "move" || item.id === "stretch" || item.id === "water" || item.id === "jaw"
      ? BREAK_LINES[item.id]
      : item.id === "offer"
        ? BREATHE_LINES.offer
        : moments.take(pclock);
  if (item.id === "eye" && !activity.start(ACTIVITY.EYE_BREAK, clockS())) {
    prompts.dismiss(pclock, "done");
    return;
  }
  card = { id: item.id, text, t: 0, finished: false };
  setCardFill(0);
  renderCard();
  if (item.id !== "offer") sayPrompt(text);
}

/** The first breath check offer. Asked at most 3 times; it never starts a check without consent. */
function showBreathIntro() {
  if (lowAccuracy) {
    prompts.dismiss(pclock, "done"); // the fallback model has no reliable shoulder signal
    return;
  }
  store.breathIntroOffers = (store.breathIntroOffers || 0) + 1;
  if (store.breathIntroOffers >= 3) store.breathIntroShown = true; // the last time it asks
  save();
  const note = hasConsent(store) ? "" : " You will be asked to agree before anything is measured.";
  card = { id: "breath_intro", text: RELAX_LINES.introOffer + note, t: 0, finished: false };
  setCardFill(0);
  renderCard();
  sayPrompt(RELAX_LINES.introOffer);
}

function acceptBreathIntro() {
  endCard("snooze"); // if the consent dialog is declined, the offer rests for 10 minutes
  const go = () => {
    store.breathIntroShown = true;
    save();
    startBreathCheck();
  };
  if (hasConsent(store)) go();
  else requireConsent(go);
}

function startGuide() {
  if (!loopsStarted || !canBegin(true)) return;
  if (!requireConsent(startGuide)) return;
  ensureAudio();
  if (card) endCard("done");
  if (!activity.start(ACTIVITY.BREATHE_GUIDE, clockS())) return;
  guide.start(preal);
  card = { id: "guide", text: `${BREATHE_LINES.intro}`, t: 0, finished: false };
  setCardFill(0);
  renderCard();
  if (settings.breatheCues && settings.mode !== "silent") speak(BREATHE_LINES.intro);
}

function guideStep() {
  const g = guide.update(preal);
  if (!g) return;
  if (g.phase === "done") {
    if (!shoulderChecksOn(lowAccuracy)) {
      activity.stop(ACTIVITY.BREATHE_GUIDE, clockS(), "completed"); // timed only: no hand-off to a verdict
      hideCard();
      return;
    }
    if (settings.breatheCues && settings.mode !== "silent") speak(BREATHE_LINES.done);
    runBreathCheck(); // hand-off, no grace: ends with the existing breath check (do the shoulders stay down?)
    hideCard();
    return;
  }
  setCardFill(g.fill);
  if (g.phaseStart) {
    card.text = `${g.phase === "in" ? "Breathe in..." : "Breathe out slowly..."} ${GUIDE_SAFETY}`;
    $("prompt-text").textContent = card.text;
    if (settings.breatheCues && settings.mode !== "silent" && g.cycle < 2) {
      speak(g.phase === "in" ? BREATHE_LINES.in : BREATHE_LINES.out);
    }
  }
}

/** The first breath check offer: tracking has settled, posture was good for a minute, no setup card is open. */
function breathIntroDue() {
  return (
    pstreak >= BREATH_INTRO_GOOD_S &&
    currentStage() >= STAGES.settled &&
    !firstRun.open &&
    !lowAccuracy &&
    !card &&
    !breath.active &&
    !guide.active &&
    !store.breathIntroShown
  );
}

/** The eye break countdown. The activity ends when the 20 s are up, not after the "well done" text. */
function eyeCardStep(now) {
  setCardFill(card.t / BREAKS.eye_look_s);
  if (!card.finished && card.t >= BREAKS.eye_look_s) {
    card.finished = true;
    activity.stop(ACTIVITY.EYE_BREAK, now, "completed");
    card.text = BREAK_LINES.eyeDone;
    tookBreak("eye");
    renderCard();
    sayPrompt(BREAK_LINES.eyeDone);
  }
  if (card.finished && card.t >= BREAKS.eye_look_s + 3) endCard("done");
}

/** Advance the card on screen by one frame: countdowns, demo steps and timeouts. */
function promptCardStep(dt, now) {
  card.t += dt;
  if (card.id === "eye") eyeCardStep(now);
  else if (card.id === "demo" && !card.static) {
    demoStep();
    if (card.finished && card.t >= demoLength(card.kind) + 8) endCard("done");
  } else if (card.id === "demo") {
    if (card.t >= 120) {
      activity.stop(ACTIVITY.DEMO, now, "timeout");
      endCard("done");
    }
  } else if (card.t >= 60) {
    endCard("notnow"); // ignoring a prompt counts as "Not now"
  }
}

/** Walking away is not an answer: it ends an offer card and takes back a sponsor card. */
function promptAbsence(present) {
  if (present) return;
  if (card && !CARD_ACTIVITY[card.id]) endCard("done"); // no prompt while away (breaks and the guide end through the machine)
  if (sponsorCard) {
    refundSponsorShown(store);
    save();
    hideSponsor();
  }
}

/** Queue whatever is due; the scheduler decides when it may show. */
function promptEnqueueDue(present) {
  if (
    shouldOfferSponsor(store, {
      nowMs: Date.now(),
      good: pstreak >= SPONSOR.good_streak_s,
      silent: !rules().prompts,
    })
  ) {
    prompts.enqueue("share", pclock);
  }
  if (breathIntroDue()) prompts.enqueue("breath_intro", pclock);
  for (const kind of breaks.due(present)) prompts.enqueue(kind, pclock);
  if (settings.breathingMoments && moments.due(pstreak, pclock)) prompts.enqueue("breathing", pclock);
}

/** Runs every tracked frame (real `now` in seconds). present: someone is in view. */
function promptStep(state, now, present) {
  if (inMode()) {
    plast = null;
    return;
  }
  const dt = plast === null ? 0 : Math.min(1, Math.max(0, now - plast));
  plast = now;
  preal += dt;
  pclock += dt * CLOCK_SCALE;
  // The reveal clock must keep running to open setup. Only optional prompts wait.
  if (nextSetupStep(store, STAGES.settled)) return;
  const allow = rules();
  if (breaks.update(present, pclock)) countBreak();
  if (allow.engine === "track") pstreak = state === State.GOOD ? pstreak + dt * CLOCK_SCALE : 0; // a held state says nothing about posture
  promptAbsence(present);
  promptEnqueueDue(present);
  if (sponsorCard) {
    sponsorCard.t += dt;
    if (sponsorCard.t >= 120) answerSponsorCard("later"); // ignoring it counts as "Maybe later"
  } else if (guide.active) guideStep();
  else if (card) promptCardStep(dt, now);
  else {
    const item = prompts.poll(pclock, { suppressed: !allow.prompts });
    if (item) showPrompt(item);
  }
}

$("prompt-action").addEventListener("click", () => {
  if (card && card.id === "breath_intro") acceptBreathIntro();
  else if (card && (card.id === "water" || card.act === "drink")) drank();
  else if (card && card.id === "stretch") {
    endCard("done");
    startDemo("stretch");
  } else startGuide();
});
$("prompt-done").addEventListener("click", () => {
  if (card && (card.id === "move" || card.id === "stretch" || card.id === "jaw")) tookBreak(card.id);
  // Water resets only through "I drank" (drank()), never through Done.
  if (card && card.static && card.kind !== "water") tookBreak(card.kind);
  endCard("done");
});
$("prompt-snooze").addEventListener("click", () => {
  noteAnnoyance();
  endCard("snooze");
});
$("prompt-notnow").addEventListener("click", () => {
  if (card && card.id !== "demo" && card.id !== "guide") noteAnnoyance(); // Stop on a demo is not a dismissal
  endCard("notnow");
});
$("prompt-never").addEventListener("click", () => {
  noteAnnoyance();
  endCard("never");
});

// -- "Take a break" menu: guided demos and the water count -----------------------
// Demos are text steps with a small outline, 10 to 20 s, no camera analysis.
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");

function showFigure(pose) {
  const fig = $("prompt-figure");
  fig.hidden = !pose;
  if (pose && !reducedMotion.matches) fig.dataset.pose = pose;
  else if (pose) fig.dataset.pose = pose === "sip" ? "cup" : "";
}

function startDemo(kind) {
  if (card) endCard("done");
  if (!activity.start(ACTIVITY.DEMO, clockS())) return;
  const still = reducedMotion.matches;
  const finalAct = kind === "water" ? "drink" : "";
  card = { id: "demo", kind, text: "", t: 0, finished: false, act: finalAct, ui: { bar: !still, stop: "End break" } };
  if (still) {
    // Reduced motion: every step at once, no timer, the person ends it.
    card.text = demoStaticText(kind);
    card.finished = false;
    card.ui = { done: true, action: kind === "water", actionLabel: "I drank" };
    card.static = true;
  }
  setCardFill(0);
  renderCard();
  showFigure(still ? "" : demoAt(kind, 0).pose);
  if (!still) demoStep(true);
}

function demoStep(first = false) {
  const d = demoAt(card.kind, card.t);
  if (d.index === card.demoIndex && !first && !d.done) {
    setCardFill(card.t / demoLength(card.kind));
    return;
  }
  card.demoIndex = d.index;
  setCardFill(Math.min(1, card.t / demoLength(card.kind)));
  if (d.done) {
    if (!card.finished) {
      card.finished = true;
      activity.stop(ACTIVITY.DEMO, clockS(), "completed");
      if (card.kind !== "water") tookBreak(card.kind);
      if (DEMO_DONE[card.kind]) {
        card.text = DEMO_DONE[card.kind];
        sayPrompt(card.text);
      }
      if (card.kind === "water") {
        card.ui = { action: true, actionLabel: "I drank", stop: "Didn't drink", dismissible: true };
      }
      renderCard();
      showFigure("");
    }
    return;
  }
  card.text = d.line;
  showFigure(d.pose);
  renderCard();
  sayPrompt(d.line);
}

function openBreak(kind) {
  if (!loopsStarted || !canBegin(true)) return;
  if (kind === "eye") {
    if (card) endCard("done");
    if (!activity.start(ACTIVITY.EYE_BREAK, clockS())) return;
    card = { id: "eye", text: BREAK_LINES.eye, t: 0, finished: false };
    setCardFill(0);
    renderCard();
    sayPrompt(BREAK_LINES.eye);
    return;
  }
  ensureAudio();
  startDemo(kind);
}

function drank() {
  const n = addWater(store, todayKey());
  breaks.taken("water");
  save();
  renderProgress();
  showRelaxBanner(`${n} ${n === 1 ? "drink" : "drinks"} today`);
  endCard("done");
}

// -- sponsor card (lowest priority prompt, never spoken, at most 3 times ever) -----
// A small non-modal card under the main screen, never over the video. Gating is in sponsor.js; the
// counts it uses live only in this browser's store.

let sponsorCard = null; // { t } while on screen
$("sponsor-link").href = SPONSOR_URL;

function showSponsor() {
  const still = shouldOfferSponsor(store, {
    nowMs: Date.now(),
    good: pstreak >= SPONSOR.good_streak_s,
    silent: !rules().prompts,
  });
  if (!still) {
    prompts.dismiss(pclock, "done");
    return;
  } // conditions changed while it waited
  markSponsorShown(store);
  save();
  sponsorCard = { t: 0 };
  $("sponsor-text").textContent = sponsorText(store);
  $("sponsor-card").hidden = false;
}
function hideSponsor() {
  if (!sponsorCard) return;
  sponsorCard = null;
  $("sponsor-card").hidden = true;
  prompts.dismiss(pclock, "done");
}
function answerSponsorCard(how) {
  if (how !== "done") {
    answerSponsor(store, how, Date.now());
    save();
  }
  hideSponsor();
}
$("sponsor-later").addEventListener("click", () => answerSponsorCard("later"));
$("sponsor-never").addEventListener("click", () => answerSponsorCard("never"));
$("sponsor-link").addEventListener("click", () => answerSponsorCard("never")); // they acted: do not ask again
$("sponsor-share").addEventListener("click", () => {
  answerSponsorCard("later");
  openShare();
});

// -- share (Web Share API, else a menu of plain links; nothing third-party is loaded) --
function renderShareLinks() {
  const links = shareLinks(shareUrl(location.href), { instance: $("share-instance").value });
  for (const [id, href] of Object.entries(links)) {
    const a = $(`share-${id}`);
    if (a) a.href = href;
  }
  $("share-mastodon").hidden = !links.mastodon;
}
function openShare() {
  const url = shareUrl(location.href);
  $("share-native").hidden = typeof navigator.share !== "function";
  $("share-link").value = url;
  $("share-text").textContent = SHARE_TEXT;
  $("share-msg").textContent = "";
  $("share-card-wrap").hidden = true;
  renderShareLinks();
  $("share-dialog").showModal();
}
$("share-native").addEventListener("click", async () => {
  try {
    await navigator.share(nativeSharePayload(shareUrl(location.href)));
  } catch (e) {
    if (e?.name !== "AbortError") $("share-msg").textContent = "The share menu was blocked. Use the links below.";
  }
});
$("share-close").addEventListener("click", () => $("share-dialog").close());
// The score card is drawn from this session's two numbers only. No video frame is read.
let cardBlob = null;
$("share-card-btn").addEventListener("click", () => {
  const d = scoreCardData(session);
  if (!d) {
    $("share-msg").textContent = "Track at least a minute of posture first, then your score card is ready.";
    return;
  }
  const canvas = $("share-card");
  drawScoreCard(canvas.getContext("2d"), d);
  canvas.setAttribute("aria-label", cardAltText(d));
  $("share-card-wrap").hidden = false;
  $("share-msg").textContent = "";
  cardBlob = null;
  canvas.toBlob((b) => {
    cardBlob = b;
    const file = b && new File([b], "backup-score.png", { type: "image/png" });
    $("share-card-share").hidden = !(file && navigator.canShare?.({ files: [file] }));
  }, "image/png");
});
$("share-card-download").addEventListener("click", () => {
  if (!cardBlob) return;
  const a = document.createElement("a");
  a.href = URL.createObjectURL(cardBlob);
  a.download = "backup-score.png";
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});
$("share-card-share").addEventListener("click", async () => {
  if (!cardBlob) return;
  try {
    await navigator.share({
      files: [new File([cardBlob], "backup-score.png", { type: "image/png" })],
      text: SHARE_TEXT,
    });
  } catch (e) {
    if (e?.name !== "AbortError")
      $("share-msg").textContent = "Sharing the image was blocked. Use Download PNG instead.";
  }
});
$("share-instance").addEventListener("input", () => {
  renderShareLinks();
  const v = $("share-instance").value.trim();
  $("share-msg").textContent = v && !normalizeInstance(v) ? "Enter just the server name, such as mastodon.social." : "";
});
$("share-copy").addEventListener("click", async () => {
  const text = `${SHARE_TEXT} ${$("share-link").value}`;
  try {
    await navigator.clipboard.writeText(text);
    $("share-msg").textContent = "Copied.";
  } catch {
    $("share-link").select();
    $("share-msg").textContent = "Copy was blocked. The link is selected: press Ctrl+C (or Cmd+C).";
  }
});

function resetPromptSession() {
  prompts.startSession(pclock);
  breaks.reset();
  moments.last = null;
  pstreak = 0;
  plast = null;
}

const DAY_SAVE_S = 15;
let lastDaySave = 0;

/** Effort-based praise at 5, 15, 30 and 60 continuous good minutes (banner and chime;
 *  spoken only when spoken praise is on, using the existing praise clips). */
function fireMilestone(minute) {
  if (!rules().praise || !policy.onPraise("milestone")) return;
  const banner = $("banner");
  banner.textContent = MILESTONE_TEXT[minute];
  banner.classList.remove("relax", "cut");
  banner.classList.add("positive", "show");
  clearTimeout(bannerTimer);
  bannerTimer = setTimeout(() => banner.classList.remove("show"), 6000);
  if (!allowed(Nudge.POSITIVE_CHIME)) return;
  playGood();
  if (settings.spokenPraise && settings.mode === "chime_and_voice") {
    const line = nextPraise();
    setTimeout(() => speak(line), custom.good ? 400 : 850);
  }
}

// Episode bookkeeping (see countEpisodes): a slouch counts once when a bad state begins, a good run
// once when the streak reaches PROGRESS.good_run_s. The flags reset when the day changes.
const episodeFlags = { dayKey: null, inBadRun: false, goodRunCounted: false };

/** Coached time: session, daily totals and the coaching policy. */
function accountDay(state, now, dt) {
  const dayKey = todayKey();
  store.days[dayKey] = store.days[dayKey] || { good: 0, bad: 0, slouch_events: 0, good_runs: 0 };
  const day = store.days[dayKey];
  if (state === State.GOOD) {
    session.good += dt;
    session.streak += dt;
    session.best = Math.max(session.best, session.streak);
    // relaxed seconds need the shoulder consent (the monitor only runs with it)
    tallyDay(day, "good", dt, {
      relaxed: hasConsent(store) && shoulderChecksOn(lowAccuracy) ? bracing.relaxed : null,
      streak: session.streak,
    });
    countEpisodes(episodeFlags, day, dayKey, "good", session.streak);
    const milestone = policy.onGood(now, dt);
    if (milestone !== null) fireMilestone(milestone);
  } else if (state === State.BAD_CHIME || state === State.BAD_VOICE) {
    session.bad += dt;
    tallyDay(day, "bad", dt);
    countEpisodes(episodeFlags, day, dayKey, "bad", 0);
    session.streak = 0;
    policy.onBad();
  }
  if (now - lastDaySave >= DAY_SAVE_S) {
    lastDaySave = now;
    save();
  } // keep daily totals across reloads
}

/** Route the frame's time by the activity: "day" tallies, "test" feeds the silent test,
 *  "none" (practice, the eye break, demos, the breathing guide) records nothing and
 *  leaves the streak where it is. */
function account(state, now) {
  if (session.lastT !== null) {
    const dt = Math.min(1, Math.max(0, now - session.lastT)); // cap gaps (sleep, hidden stalls)
    const route = rules().score;
    if (route === "test" && test) {
      accountTest(state, dt);
      session.lastT = now;
      if (test.elapsed >= test.minutes * 60) finishTest(false);
      else if (!$("test-view").hidden) renderTest();
      return;
    }
    if (route === "day") accountDay(state, now, dt);
  }
  session.lastT = now;
}

const mmss = (s) => `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, "0")}`;

// The Today panel (frame 04, #98): good % today, best streak, the next break, reminders, and a
// colour timeline of this session. The timeline lives in memory only.
const timeline = new SessionTimeline();
const RING_LEN = 2 * Math.PI * 16;
let lastTimelineRender = 0;
function renderStats() {
  const day = store.days[todayKey()];
  const tracked = day ? day.good + day.bad : 0;
  const pct = tracked > 0 ? Math.round((100 * day.good) / tracked) : null;
  $("good-pct").textContent = pct === null ? "-" : `${pct}%`;
  $soft("good-today-label").textContent = tracked > 0 ? `good of ${minutesText(tracked)} today` : "good today";
  $("best-streak").textContent = mmss(Math.max(session.best, day?.best || 0));
  $("reminder-count").textContent = String(reminderCount);
  // Episodes today: counted since this build, so a day tracked before it shows a dash, not 0.
  const hasEpisodes = day && Number.isFinite(day.slouch_events);
  $soft("today-slouch-events").textContent = !day
    ? "-"
    : hasEpisodes
      ? String(day.slouch_events)
      : day.bad > 0
        ? "-"
        : "0";
  $soft("today-slouch-label").textContent = day?.slouch_events === 1 ? "slouch episode" : "slouch episodes";
  $soft("today-good-runs").textContent = !day
    ? "-"
    : Number.isFinite(day.good_runs)
      ? String(day.good_runs)
      : day.good > 0
        ? "-"
        : "0";
  const next = breaks.nextIn();
  $("break-ring-arc").setAttribute("stroke-dashoffset", String(RING_LEN * (next ? 1 - next.fraction : 1)));
  // The number in the ring is breaks taken today (restored after #98 v3 dropped it).
  const taken = breaksTakenToday();
  $("breaks-today").textContent = String(taken);
  const when = !next ? "breaks off" : next.seconds < 60 ? "break now" : `break in ${Math.round(next.seconds / 60)} min`;
  $("break-in").textContent = when;
  $("break-in").title = `${taken} ${taken === 1 ? "break" : "breaks"} taken today`;
  if (performance.now() - lastTimelineRender < 1000) return;
  lastTimelineRender = performance.now();
  $("timeline").replaceChildren(
    ...timeline.segments().map((seg) => {
      const el = document.createElement("div");
      el.className = seg.zone;
      el.style.flex = `${seg.fraction} 0 0`;
      return el;
    }),
  );
  if (timeline.start !== null && !$("timeline-start").textContent) {
    const t = new Date(Date.now() - (performance.now() / 1000 - timeline.start) * 1000);
    $("timeline-start").textContent = t.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  }
}

// -- silent test ("Test me", under Help & more) ---------------------------------------
// A retention test in motor-learning terms: same tracking and state machine, but every cue
// is withheld (no sounds, speech, banners, needle or fill), so the score reflects the habit
// rather than the reminders. Results are kept per test; coached time keeps feeding store.days.
store.tests = store.tests || [];
let test = null; // { minutes, elapsed, good, bad } while a test runs
const pctOf = (r) => (r.good + r.bad > 0 ? Math.round((100 * r.good) / (r.good + r.bad)) : null);

function startTest() {
  if (test) return finishTest(true);
  if (practice) stopPractice();
  if (!activity.start(ACTIVITY.SILENT_TEST, clockS())) {
    $("test-status").textContent = "Resume tracking first.";
    return;
  }
  test = { minutes: Number($("test-minutes").dataset.value) || 10, elapsed: 0, good: 0, bad: 0 };
  breath.cancel();
  cancelPrompts();
  stopSpeech();
  document.body.classList.add("testing");
  $("banner").classList.remove("show");
  setSettingsOpen(false); // the test runs on the main screen; Settings shows its progress
  renderTest();
}

function finishTest(stoppedEarly) {
  const t = test;
  test = null;
  activity.stop(ACTIVITY.SILENT_TEST, clockS(), stoppedEarly ? "cancelled" : "completed");
  document.body.classList.remove("testing");
  const tracked = t.good + t.bad;
  if (!stoppedEarly && tracked >= 30) {
    store.tests.push({
      at: new Date().toISOString(),
      minutes: t.minutes,
      good: Math.round(t.good),
      bad: Math.round(t.bad),
    });
    store.tests = store.tests.slice(-50);
    save();
    $("test-status").textContent = `Done: ${pctOf(t)}% good posture with no reminders.`;
    showRelaxBanner(`Silent test: ${pctOf(t)}% good`);
    renderProgress();
    playGood();
  } else {
    $("test-status").textContent = stoppedEarly
      ? "Test stopped, not saved."
      : "Not enough tracked time to score; not saved.";
  }
  renderTest();
}

function renderTest() {
  setRowValue("test-minutes", $("test-minutes").dataset.value || 10, `${$("test-minutes").dataset.value || 10} min`);
  $("test-start").textContent = test ? "Stop test" : "Start silent test";
  $("test-progress-wrap").style.display = test ? "" : "none";
  $("test-value").textContent = test ? "Running" : "";
  if (test) {
    const left = Math.max(0, test.minutes * 60 - test.elapsed);
    $("test-status").textContent = `Running: ${mmss(left)} left. Sit and work as usual.`;
    $("test-progress").style.width = `${Math.min(100, (100 * test.elapsed) / (test.minutes * 60)).toFixed(1)}%`;
  }
  const tests = store.tests;
  const coached = Object.values(store.days).reduce((a, d) => ({ good: a.good + d.good, bad: a.bad + d.bad }), {
    good: 0,
    bad: 0,
  });
  const parts = [];
  if (tests.length) {
    const first = tests[0];
    const last = tests[tests.length - 1];
    parts.push(
      tests.length === 1
        ? `Last test: ${pctOf(last)}% good`
        : `First test ${pctOf(first)}%, latest ${pctOf(last)}% (${tests.length} tests)`,
    );
    if (coached.good + coached.bad > 0) parts.push(`with coaching: ${pctOf(coached)}%`);
  }
  $("test-compare").textContent = parts.join(" \u00b7 ");
}
$("test-minutes").addEventListener("click", () => {
  if (test) return; // the length is fixed once a test runs
  const v = cycleValue($("test-minutes"));
  setRowValue("test-minutes", v, `${v} min`);
});
$("test-start").addEventListener("click", () => {
  ensureAudio();
  startTest();
});

/** Silent test time: counted to the test only, never to the daily totals. */
function accountTest(state, dt) {
  test.elapsed += dt;
  if (state === State.GOOD) test.good += dt;
  else if (state === State.BAD_CHIME || state === State.BAD_VOICE) test.bad += dt;
}

// -- rendering ----------------------------------------------------------------

const STATE_LABELS = {
  [State.GOOD]: "good posture",
  [State.BAD_CHIME]: "sit up",
  [State.BAD_VOICE]: "sit up",
  [State.NO_PERSON]: "no one in view",
  [State.UNRELIABLE]: "tracking unsure",
};

const SKELETON = [
  [7, 8],
  [7, 11],
  [8, 12],
  [11, 12],
  [11, 23],
  [12, 24],
  [23, 24],
];

// Demo only: a synthetic cartoon figure (not a real person), the provided avatar drawn at its
// own proportions in two joined pieces: the body (hair behind, neck, sweater) and the head
// (ears, face, front hair, glasses). The scripted pose uses the same proportions, so the body
// sits on the shoulder landmarks and the head on the ear landmarks. A slouch moves the head
// down over the collar, as a real head-forward slouch looks from the front.
const AVATAR = { w: 640, sh: [319, 410], ear: [320, 226], shDist: 264 };
function updateDemoAvatar(lm) {
  const wrap = $("demo-avatar");
  const on = Boolean(demo && lm);
  wrap.hidden = !on;
  wrap.parentElement.classList.toggle("demo-on", on);
  if (!on) return;
  const W = canvas.clientWidth;
  const H = canvas.clientHeight;
  const s = Math.hypot((lm[11].x - lm[12].x) * W, (lm[11].y - lm[12].y) * H) / AVATAR.shDist;
  const tiltOf = (a, b) => {
    let deg = (Math.atan2((b.y - a.y) * H, (a.x - b.x) * W) * 180) / Math.PI;
    if (deg > 90) deg -= 180;
    if (deg < -90) deg += 180;
    return deg;
  };
  const pin = (img, a, b, anchor) => {
    img.style.width = `${AVATAR.w * s}px`;
    img.style.left = `${(1 - (a.x + b.x) / 2) * W - anchor[0] * s}px`;
    img.style.top = `${((a.y + b.y) / 2) * H - anchor[1] * s}px`;
    img.style.transformOrigin = `${anchor[0] * s}px ${anchor[1] * s}px`;
    img.style.transform = `rotate(${tiltOf(a, b).toFixed(2)}deg)`;
  };
  pin($("demo-avatar-body"), lm[11], lm[12], AVATAR.sh);
  pin($("demo-avatar-head"), lm[7], lm[8], AVATAR.ear);
}

function drawSkeleton(landmarks, color, cv = canvas, c = ctx, scale = 1) {
  cv.width = cv.clientWidth;
  cv.height = cv.clientHeight;
  const w = cv.width;
  const h = cv.height;
  c.clearRect(0, 0, w, h);
  if (!landmarks) return;
  c.strokeStyle = color;
  c.fillStyle = color;
  c.lineWidth = 4 * scale;
  for (const [a, b] of SKELETON) {
    const p = landmarks[a],
      q = landmarks[b];
    if (![p.x, p.y, q.x, q.y].every(Number.isFinite)) continue;
    c.beginPath();
    c.moveTo(p.x * w, p.y * h);
    c.lineTo(q.x * w, q.y * h);
    c.stroke();
  }
  const le = landmarks[7],
    re = landmarks[8];
  if ([le.x, le.y, re.x, re.y].every(Number.isFinite)) {
    // head outline between the ears
    const r = Math.hypot((re.x - le.x) * w, (re.y - le.y) * h) * 0.75;
    c.beginPath();
    c.arc(((le.x + re.x) / 2) * w, ((le.y + re.y) / 2) * h - r * 0.35, r, 0, 2 * Math.PI);
    c.stroke();
  }
  for (const i of TRACKED) {
    const p = landmarks[i];
    if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) continue;
    c.beginPath();
    c.arc(p.x * w, p.y * h, 6 * scale, 0, 2 * Math.PI);
    c.fill();
  }
}

// A real camera that still finds no one after FRAMING_HELP_MS gets a short checklist and a way into the demo.
const FRAMING_HELP_MS = 10000;
let noPoseSince = null;
function updateFramingHelp(result) {
  const none = !demo && result.kind === "unreliable" && result.reason === Reason.NO_POSE;
  if (!none) noPoseSince = null;
  else noPoseSince ??= performance.now();
  $("framing-help").hidden = !(none && performance.now() - noPoseSince >= FRAMING_HELP_MS);
}
$("framing-demo").addEventListener("click", () => {
  stopCamera(new Error("camera stopped"));
  showError("");
  $("framing-help").hidden = true;
  noPoseSince = null;
  $("start-demo").click();
});

function causeFor(result, landmarks) {
  if (landmarks && landmarks[LEFT_EAR].y < 0.1) return "head near the top edge - lower the camera or sit lower";
  if (result.kind !== "unreliable") return "";
  switch (result.reason) {
    case Reason.NO_POSE:
      return "no one in view - sit facing the camera";
    case Reason.LOW_VISIBILITY:
      return `low confidence - ${result.detail}`;
    case Reason.FRAMING_OUT_OF_BAND:
      return result.detail.startsWith("too close") ? "too close - move back a little" : "too far - move closer";
    case Reason.NON_FINITE:
      return "tracking glitch - holding the last state";
    default:
      return result.detail;
  }
}

// -- load marks -----------------------------------------------------------------
// performance.now() at each start-up milestone, kept for the bug-report
// diagnostics and for e2e (window.__loadMarks). Only the first hit counts.
const loadMarks = {};
window.__loadMarks = loadMarks;
function markLoad(name) {
  if (loadMarks.click !== undefined && loadMarks[name] === undefined) loadMarks[name] = Math.round(performance.now());
}

// The tab title mirrors an active reminder so it shows in a background tab.
// Set only when the text changes (no blinking).
const BASE_TITLE = document.title;
function syncTabTitle() {
  const t = tabTitleFor($("state-badge").className, BASE_TITLE);
  if (document.title !== t) document.title = t;
}

/** The badge: the state, or the activity while one holds posture feedback. */
function renderBadge(transition, result, held, label, zone, hold) {
  const badge = $("state-badge");
  // NO_PERSON is also the start state: it lasts until posture is clearly good
  // or clearly bad, so a tracked person sitting in the in-between band would
  // otherwise read "no one in view" while being scored.
  badge.textContent =
    transition.state === State.NO_PERSON && result.kind === "scored"
      ? "settling in"
      : STATE_LABELS[transition.state] || transition.state;
  badge.className = `badge ${transition.state}`;
  // #111: right after a break the restarted engine settles; say why, without a banner (grace keeps the video quiet).
  if (badge.textContent === "settling in" && activity.inGrace(clockS())) badge.textContent = "back from break";
  if (zone === "drift") {
    badge.textContent = "slipping"; // amber with the needle and skeleton (frame 05, #98)
    badge.className = "badge drift";
  }
  const settling = !test && !practice && !held && badge.textContent === "settling in";
  // #105: while good posture is being confirmed the note counts down and a small bar
  // fills; before any timer runs it says why the badge waits.
  const confirming = settling && hold && hold.kind === "confirm";
  const noteText = confirming
    ? `Settling in: hold still ~${Math.max(1, Math.ceil(hold.remaining))}s`
    : "Settling in: waiting for a clear good or slouched posture.";
  $("settling-note").hidden = !settling;
  $("settling-text").textContent = noteText;
  badge.title = settling ? noteText : "";
  $("settling-bar").hidden = !confirming;
  if (confirming) $("settling-fill").style.width = `${(hold.fraction * 100).toFixed(0)}%`;
  if (held) {
    badge.textContent = label;
    badge.className = "badge activity";
  }
  if (test) {
    badge.textContent = "silent test";
    badge.className = "badge testing";
  }
  if (practice) {
    badge.textContent = "practice";
    badge.className = "badge practice";
  }
  syncTabTitle();
}

/** Body-part tiles (frames 04 to 06, #98): the angle, and the icon colour from tileStates.
 *  Both use the angles from the saved posture, the same ones the reminders use, so a tile
 *  can't be amber while the gauge is green because of where the camera sits. */
const LEAN_UNAVAILABLE_NOTE =
  "Lean needs both hips in view. They were not visible enough to measure it, so Lean is left out of your score.";

function renderMetrics(relative, zone, cue) {
  const values = {};
  for (const key of ["head_forward_deg", "head_tilt_deg", "shoulder_level_deg", "trunk_lean_deg"]) {
    let v = relative ? relative[key] : null;
    if (key === "trunk_lean_deg" && relative && !relative.trunk_lean_available) v = null;
    values[key] = v;
    $(`m-${key}`).textContent = v === null ? "-" : `${Math.abs(Math.round(v))}\u00b0`; // how far off, either way
  }
  // Lean is measured from shoulders to hips. When the hips are not visible enough, there is no
  // value, so the tile says why (hover, focus or long-press) instead of a bare dash.
  const leanTile = $("t-trunk_lean");
  const leanMissing = Boolean(relative) && !relative.trunk_lean_available;
  if (leanMissing) {
    leanTile.title = LEAN_UNAVAILABLE_NOTE;
    leanTile.setAttribute("aria-label", `Lean: no value. ${LEAN_UNAVAILABLE_NOTE}`);
    leanTile.tabIndex = 0;
  } else {
    leanTile.removeAttribute("title");
    leanTile.removeAttribute("aria-label");
    leanTile.removeAttribute("tabindex");
  }
  leanTile.classList.toggle("na", leanMissing);
  const states = tileStates(zone, cue?.metric ?? null, values, DEFAULTS.scoring);
  for (const [metric, st] of Object.entries(states)) {
    const tile = $(`t-${metric}`);
    tile.classList.toggle("warn", st === "warn");
    tile.classList.toggle("bad", st === "bad");
    tile.classList.toggle("cued", st !== "ok" && cue?.metric === metric);
  }
}

/** Needle, label and skeleton colour from one zone (mini-view.js), so they always agree (#97). */
let shownAngle = null;
let shownAngleAt = 0;
function renderNeedle(view, held) {
  const rotor = $("needle-rotor");
  // No zone (settling in, not tracked): no needle, so it can't point into an arc the label
  // and skeleton don't show, and no leftover "slouching" from before.
  const zoned = view.zone !== "none" && view.angle !== null;
  rotor.style.visibility = zoned || held ? "" : "hidden";
  if (zoned) {
    // Glide to the target angle: a zone change sweeps across the dial instead of snapping.
    const now = performance.now();
    shownAngle = glideAngle(shownAngle, view.angle, (now - shownAngleAt) / 1000);
    shownAngleAt = now;
    rotor.setAttribute("transform", angleTransform(shownAngle));
  } else {
    shownAngle = null; // hidden: the next appearance starts at its target
  }
  if (held) setGaugeLabel("posture paused");
  else setGaugeLabel(view.label ?? "settling in"); // one word; the cue itself is the banner on the video
}

// Under the gauge (K1, K2 in #98): while good, "Looking good" and a green fill toward the
// next praise; while slipping, an amber fill that is full when the chime plays. The bar
// stays full for a moment on the chime frame, then hides: no fill while slouching.
const CHIME_FULL_MS = 300;
let chimeFullUntil = -Infinity;
function renderGaugeStatus(zone, hold, nudge) {
  const now = performance.now();
  if (nudge === Nudge.CHIME) chimeFullUntil = now + CHIME_FULL_MS;
  let kind = null;
  let fraction = 0;
  if (now < chimeFullUntil) {
    kind = "warn";
    fraction = 1;
  } else if (hold && zone === "good" && hold.kind === "praise") {
    kind = "good";
    fraction = hold.fraction;
  } else if (hold && zone === "drift" && hold.kind === "slipping") {
    kind = "warn";
    fraction = hold.fraction;
  }
  $("gauge-praise").hidden = kind !== "good";
  const bar = $("gauge-bar");
  bar.hidden = kind === null;
  if (kind === null) return;
  bar.className = `gauge-bar ${kind}`;
  bar.setAttribute("aria-label", kind === "good" ? "Toward the next praise" : "Toward the reminder");
  $("gauge-fill").style.width = `${(fraction * 100).toFixed(1)}%`;
  bar.setAttribute("aria-valuenow", String(Math.round(fraction * 100)));
}

// The gauge is role="img", so screen readers read its aria-label, not the <text> inside.
// Both carry the same words, so the state never depends on colour or the needle angle (#95).
function setGaugeLabel(label) {
  if ($("needle-text").textContent === label) return;
  $("needle-text").textContent = label;
  $("gauge-svg").setAttribute("aria-label", `Posture: ${label}`);
}

function render(out, landmarks) {
  const { result, transition, needle } = out;
  const held = out.held === true;
  const label = rules().label;
  if (result.kind === "scored") {
    markLoad("firstScored");
    if (transition.state !== State.NO_PERSON) markLoad("settled");
  }
  if (practice) renderPractice(result);
  const view = gaugeView(transition.state, result.kind === "scored", out.hold?.kind === "slipping", needle);
  renderBadge(transition, result, held, label, view.zone, out.hold);
  $("warnings").textContent = causeFor(result, landmarks);
  updateFramingHelp(result);
  renderMetrics(out.relative, held ? "none" : view.zone, held ? null : out.cue);
  if (result.kind === "scored" || held) timeline.add(held ? "break" : view.zone, performance.now() / 1000);
  renderNeedle(view, held);
  // Praise on the video goes as soon as posture leaves the green zone (frame 05, #98).
  const banner = $("banner");
  // "cut" drops the fade, so no faint pill is left on the video (frame 05).
  if (view.zone !== "good" && banner.classList.contains("positive") && banner.classList.contains("show")) {
    banner.classList.add("cut");
    banner.classList.remove("show");
  }
  renderGaugeStatus(view.zone, held || activity.inGrace(clockS()) || practice ? null : out.hold, out.transition.nudge);
  // #105/#110: the specific cue while slipping or slouching, so the offending angle is
  // named on the main screen, not only inside Learn the cues.
  const cueLine = $("gauge-cue");
  const cueVisible = !held && !test && !practice && out.cue && (view.zone === "drift" || view.zone === "bad");
  cueLine.hidden = !cueVisible;
  if (cueVisible) cueLine.textContent = CUE_SHORT[out.cue.metric] || "Sit up";
  drawSkeleton(landmarks, held ? ZONE_COLOR.none : view.color);
  updateDemoAvatar(landmarks);
  renderCalibration();
  updateFirstRunLive();
  renderStats();
  if (Date.now() - lastProgressRender > 10000) renderProgress();
}

// -- frame loop ----------------------------------------------------------------

let poseLandmarker = null;
let demo = false;
let lastProcessed = 0;
let lastVideoMs = -1;
let fpsEma = 0;
let lastFrameT = null;

// -- pause ----------------------------------------------------------------------
// Paused: no pose tracking, no reminders or praise, no stats, and the camera
// track is disabled (the picture goes black; most browsers turn the camera
// light off). Resuming starts fresh so paused time can never fire a reminder.

function setPaused(on) {
  if (!loopsStarted || on === isPaused()) return;
  owed.clear(); // step() returns early while paused, so a reminder owed from grace would otherwise outlive the pause
  if (on) {
    if (practice) stopPractice();
    activity.start(ACTIVITY.PAUSED, clockS());
  } else activity.stop(ACTIVITY.PAUSED, clockS());
  save(); // flush daily totals when pausing
  const stream = video.srcObject;
  if (stream)
    stream.getVideoTracks().forEach((t) => {
      t.enabled = !on;
    });
  if (on) {
    breath.cancel();
    cancelPrompts();
    stopSpeech();
    if (firstRun.open) $("first-run").hidden = true; // the setup card waits; the next frame after Resume shows it again
    $("banner").classList.remove("show");
    const badge = $("state-badge");
    badge.textContent = "paused";
    badge.className = "badge paused";
    syncTabTitle();
    document.body.dataset.fps = "-";
    // #104: the frozen gauge and skeleton dim (CSS on body.is-paused), the gauge says so, and
    // the badge counts the paused time. The count is display only, never read by scoring.
    setGaugeLabel("paused");
    pausedAt = performance.now();
    clearInterval(pausedTimer);
    pausedTimer = setInterval(() => {
      if (isPaused()) badge.textContent = `paused ${mmss((performance.now() - pausedAt) / 1000)}`;
    }, 1000);
  } else {
    clearInterval(pausedTimer);
    engine.restart();
    bracing.reset();
    relaxLastT = null;
    plast = null;
    session.lastT = null;
    lastFrameT = null;
    fpsEma = 0;
    setup.reset();
  }
  document.body.classList.toggle("is-paused", on);
  $("pause").setAttribute("aria-label", on ? "Resume" : "Pause");
  $("pause").setAttribute("aria-pressed", String(on));
  setPauseTitle(on);
}
let pausedAt = 0;
let pausedTimer = null;

function setPauseTitle(paused) {
  const text = paused ? RESUME_TITLE : PAUSE_TITLE;
  $("pause").title = settings.pauseShortcut ? `${text} (shortcut: P)` : text;
}

$("pause").addEventListener("click", () => setPaused(!isPaused()));
// The P shortcut is a fixed default since #98 v3 (store.pauseShortcut is still honoured).
const PAUSE_TITLE = "Pause tracking and all sounds";
const RESUME_TITLE = "Resume tracking and sounds";
setPauseTitle(false);
document.addEventListener("keydown", (e) => {
  if (isPauseShortcut(e, settings.pauseShortcut)) setPaused(!isPaused());
});

// -- staged reveal ----------------------------------------------------------------
// Panels carry data-stage="N" in index.html and start hidden (the hidden attribute, so
// they leave the tab order). RevealStages (reveal.js) decides the stage from tracking
// time on the prompt clock, so ?clockScale speeds it up too. A stage never goes back
// down internally; only the first two stages are automatic. Optional panels need
// an explicit choice (Explore more or Show all settings), even on repeat visits.
// Every element stays in the DOM, so settings renderers and tests keep working on
// hidden panels. Stages are not announced: the panels appearing is the news.

const reveal = new RevealStages();

function currentStage() {
  // Time and repeat visits never opt someone into a more complicated workspace.
  return Math.min(reveal.stage, STAGES.settled);
}

function applyReveal() {
  const stage = currentStage();
  for (const el of document.querySelectorAll("[data-stage]")) el.hidden = Number(el.dataset.stage) > stage;
  // "Finding you" holds the gauge's place until tracking settles (frame 02).
  $("finding").hidden = stage !== STAGES.tracking;
  // Learn the cues uses the main gauge and its tiles, even before tracking has settled.
  if (document.body.classList.contains("practicing")) {
    $("gauge").hidden = $("body-panel").hidden = false;
    $("finding").hidden = true;
  }
  return stage;
}

function revealStep(out) {
  const scored = out.result.kind === "scored";
  const { changed } = reveal.update(
    {
      scored,
      // Any scored person counts, whatever their posture: someone sitting in the band between
      // good and slouched (often the camera angle, before Save my posture) stays in the
      // NO_PERSON start state, and used to wait forever for the "Sit comfortably" card.
      present: scored,
      paused: isPaused(),
      silentTest: activity.isActive(ACTIVITY.SILENT_TEST),
      practice: activity.isActive(ACTIVITY.PRACTICE),
    },
    pclock,
  );
  if (changed) applyReveal();
  store.trackedS = trackedTotalS(); // saved with the next save(): unlocks survive reloads
  const level = currentUnlockLevel();
  if (level !== shownUnlockLevel) {
    shownUnlockLevel = level;
    renderRows();
  }
  unlockStep();
}

applyReveal();

// -- settings side panel and unlocks (#96, storyboard v3 frames 07 to 12) --------------
// Settings is one grouped list in a side panel over the dimmed main screen. Rows unlock as
// the app is used (unlockLevel in reveal.js, from the stored tracked time and days); each new
// row gets one card on the main screen, at most one per session, and the gear gets a dot until
// Settings is opened. Tracking keeps running while the panel is open.
const sessionStartTrackedS = store.trackedS ?? trackedSecondsFromDays(store.days);
function trackedTotalS() {
  return sessionStartTrackedS + reveal.tracked;
}
function currentUnlockLevel() {
  return unlockLevel({ trackedS: trackedTotalS(), days: trackedDays(store.days) });
}
function rowsShown() {
  return visibleRows(currentUnlockLevel(), !!store.revealAll);
}
if (!store.rowsSeen) store.rowsSeen = [...visibleRows(0, false)];
let shownUnlockLevel = -1;

const MODE_FOR = (chime, sayFix) => (!chime ? "silent" : sayFix ? "chime_and_voice" : "chime_only");
function setMode(mode) {
  if (MODE_RANK[mode] < MODE_RANK[settings.mode]) noteAnnoyance(); // muting counts toward the Light suggestion
  settings.mode = mode;
  store.mode = mode;
  save();
  renderRows();
}

/** Show the unlocked rows, mark the new ones, and set every switch from the settings. */
function renderRows() {
  const shown = new Set(rowsShown());
  for (const row of document.querySelectorAll("#settings-list [data-row]")) {
    const key = row.dataset.row;
    row.hidden = !shown.has(key);
    row.classList.toggle("new", shown.has(key) && !store.rowsSeen.includes(key));
  }
  for (const group of document.querySelectorAll("#settings-list .row-group")) {
    group.hidden = ![...group.querySelectorAll("[data-row]")].some((r) => !r.hidden);
  }
  $("row-chime").checked = settings.mode !== "silent";
  $("row-sayfix").checked = settings.mode === "chime_and_voice";
  $("row-breaks").checked = breakRemindersOn();
  $("row-shoulders").checked = settings.relaxReminders && hasConsent(store);
  $("breaks-value").textContent = breakRemindersOn() ? "On" : "Off";
  $("shoulders-value").textContent = settings.relaxReminders && hasConsent(store) ? "On" : "Off";
  $("settings-dot").hidden = [...shown].every((k) => store.rowsSeen.includes(k));
  $("sound-btn").setAttribute("aria-pressed", String(settings.mode === "silent"));
  $("sound-btn").setAttribute("aria-label", settings.mode === "silent" ? "Sound off" : "Sound on");
}

function setSettingsOpen(open, view = "settings-list") {
  if (open === !$("settings-view").hidden) {
    if (open) showView(view);
    return;
  }
  if (!open) {
    // Closing marks what was on show as seen: the green rows and the dot go.
    store.rowsSeen = [...new Set([...store.rowsSeen, ...rowsShown()])];
    save();
  }
  $("settings-view").hidden = !open;
  $("settings-scrim").hidden = !open;
  $("settings-btn").setAttribute("aria-expanded", String(open));
  showView(open ? view : "settings-list", false);
  renderRows();
  if (open && view === "settings-list") $("settings-title").focus();
  if (!open) $("settings-btn").focus({ preventScroll: true });
}

// One page of the side panel at a time: the list, or a sub-page reached from a row with an
// arrow (data-view). A sub-page's back arrow returns to the page and row that opened it.
let currentView = "settings-list";
function showView(id, focus = true) {
  currentView = id;
  for (const v of document.querySelectorAll("#settings-view .side-view")) v.hidden = v.id !== id;
  if (id === "week-view") renderProgress();
  if (id === "breaks-view" || id === "shoulders-view") renderPromptSettings();
  if (id === "more-view") renderMore();
  if (id === "test-view") renderTest();
  if (id === "break-view") renderBreakMenu();
  if (focus) $(id).querySelector("h2")?.focus();
}
function backFrom(id) {
  const opener = $($(id).dataset.back);
  const parent = opener?.closest(".side-view")?.id || "settings-list";
  showView(parent, false);
  opener?.focus();
}
for (const row of document.querySelectorAll("#settings-view [data-view]"))
  row.addEventListener("click", () => showView(row.dataset.view));
for (const back of document.querySelectorAll("#settings-view .side-view .back"))
  back.addEventListener("click", () => backFrom(back.closest(".side-view").id));
/** Kept for callers from before the sub-pages: My week on or off. */
function showWeek(on) {
  showView(on ? "week-view" : "settings-list", on);
}
$("settings-btn").addEventListener("click", () => setSettingsOpen($("settings-view").hidden));
$("settings-close").addEventListener("click", () => setSettingsOpen(false));
$("settings-scrim").addEventListener("click", () => setSettingsOpen(false));
document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape" || $("settings-view").hidden || document.querySelector("dialog[open]")) return;
  if (currentView !== "settings-list") backFrom(currentView);
  else setSettingsOpen(false);
});

// Rows tapped to change their value (no dropdowns, frames 08 to 12): the value shows on the right.
function cycleValue(el) {
  return nextOption(el.dataset.options.split(","), el.dataset.value);
}
function setRowValue(id, value, text) {
  const el = $(id);
  el.dataset.value = String(value);
  el.querySelector(".value").textContent = text;
  el.setAttribute("aria-label", `${el.querySelector("span").textContent}: ${text}. Tap to change.`);
}

$("row-chime").addEventListener("change", (e) => setMode(MODE_FOR(e.target.checked, $("row-sayfix").checked)));
$("row-sayfix").addEventListener("change", (e) =>
  setMode(MODE_FOR(e.target.checked || $("row-chime").checked, e.target.checked)),
);
$("row-breaks").addEventListener("change", (e) => {
  setBreakReminders(e.target.checked);
  renderPromptSettings();
});
$("row-shoulders").addEventListener("change", (e) => {
  setShoulders(e.target.checked);
  renderPromptSettings();
});
$("row-posture").addEventListener("click", () => {
  setSettingsOpen(false);
  engine.calibration.start("good");
});
// Header icons (frame 04): sound on/off is the Chime row; the cup opens Take a break.
$("sound-btn").addEventListener("click", () => setMode(MODE_FOR(settings.mode === "silent", true)));
$("break-btn").addEventListener("click", () => setSettingsOpen(true, "break-view"));
$("row-help").addEventListener("click", () => {
  setSettingsOpen(false);
  openHelp();
});

// -- Break reminders sub-page ---------------------------------------------------------
function renderPromptSettings() {
  $("row-breaks").checked = breakRemindersOn();
  $("breaks-detail").hidden = !breakRemindersOn();
  $("eye-breaks").checked = settings.eyeBreaks && !dismissed.eye;
  setRowValue(
    "move-minutes",
    dismissed.move ? 0 : settings.moveBreakMinutes,
    minutesLabel(dismissed.move ? 0 : settings.moveBreakMinutes),
  );
  for (const [id, key, d] of [
    ["stretch-minutes", "stretchMinutes", "stretch"],
    ["water-minutes", "waterMinutes", "water"],
    ["jaw-minutes", "jawMinutes", "jaw"],
  ]) {
    const v = dismissed[d] ? 0 : settings[key];
    setRowValue(id, v, minutesLabel(v));
  }
  $("prompts-reset").hidden = !Object.keys(dismissed).length;
  const consent = hasConsent(store);
  $("row-shoulders").checked = settings.relaxReminders && consent;
  $("breathing-moments").checked = settings.breathingMoments && !dismissed.breathing && consent;
  $("breathe-cues").checked = settings.breatheCues;
  $("consent-withdraw").hidden = !consent;
  renderRows();
}
function savePromptSettings() {
  store.eyeBreaks = settings.eyeBreaks;
  store.moveBreakMinutes = settings.moveBreakMinutes;
  store.stretchMinutes = settings.stretchMinutes;
  store.waterMinutes = settings.waterMinutes;
  store.jawMinutes = settings.jawMinutes;
  store.breathingMoments = settings.breathingMoments;
  store.breatheCues = settings.breatheCues;
  store.promptsDismissed = { ...dismissed };
  save();
  renderPromptSettings();
}
$("eye-breaks").addEventListener("change", (e) => {
  settings.eyeBreaks = e.target.checked;
  if (e.target.checked) delete dismissed.eye;
  breaks.setEyeBreaks(settings.eyeBreaks);
  savePromptSettings();
});
const INTERVAL_ROWS = [
  ["move-minutes", "moveBreakMinutes", "move", (v) => breaks.setMoveMinutes(v), () => breaks.moveMinutes],
  ["stretch-minutes", "stretchMinutes", "stretch", (v) => breaks.setStretchMinutes(v), () => breaks.stretchMinutes],
  ["water-minutes", "waterMinutes", "water", (v) => breaks.setWaterMinutes(v), () => breaks.waterMinutes],
  ["jaw-minutes", "jawMinutes", "jaw", (v) => breaks.setJawMinutes(v), () => breaks.jawMinutes],
];
for (const [id, key, d, set, get] of INTERVAL_ROWS) {
  $(id).addEventListener("click", () => {
    set(cycleValue($(id)));
    settings[key] = get();
    if (key === "moveBreakMinutes" && settings[key] > 0) store.lastMoveMinutes = settings[key];
    if (settings[key] > 0) delete dismissed[d];
    savePromptSettings();
  });
}
$("prompts-reset").addEventListener("click", () => {
  for (const k of Object.keys(dismissed)) delete dismissed[k];
  breaks.setEyeBreaks(settings.eyeBreaks);
  savePromptSettings();
});

// -- Shoulders & breathing sub-page ------------------------------------------------------
$("breathing-moments").addEventListener("change", (e) => {
  const on = e.target.checked;
  const apply = () => {
    settings.breathingMoments = on;
    if (on) delete dismissed.breathing;
    savePromptSettings();
  };
  if (on && !requireConsent(apply)) {
    renderPromptSettings(); // stays off until the dialog is agreed
    return;
  }
  apply();
});
$("breathe-cues").addEventListener("change", (e) => {
  settings.breatheCues = e.target.checked;
  savePromptSettings();
});
$("breath-check").addEventListener("click", () => {
  setSettingsOpen(false);
  startBreathCheck(); // asks for consent first when needed
});
$("consent-withdraw").addEventListener("click", () => {
  settings.relaxReminders = false;
  store.relaxReminders = false;
  settings.breathingMoments = false;
  store.breathingMoments = false;
  withdrawShoulders();
  renderPromptSettings();
});

// -- Take a break sub-page (the cup icon): pick the kind of break -----------------------------
function renderBreakMenu() {
  const n = breaksTakenToday();
  const water = store.water?.[todayKey()] || 0;
  const parts = [];
  if (n) parts.push(`${n} ${n === 1 ? "break" : "breaks"} today`);
  if (water) parts.push(`${water} ${water === 1 ? "drink" : "drinks"}`);
  $("break-menu-note").textContent = parts.join(" \u00b7 ");
  const off = !loopsStarted;
  for (const id of ["break-eye", "break-move", "break-stretch", "breathe-with-me", "break-water", "breath-check"]) {
    $(id).disabled = off;
    if (off) $(id).title = NEEDS_TRACKING_TIP;
    else $(id).removeAttribute("title");
  }
}
for (const kind of ["eye", "move", "stretch", "water"]) {
  $(`break-${kind}`).addEventListener("click", () => {
    setSettingsOpen(false);
    openBreak(kind);
  });
}
$("breathe-with-me").addEventListener("click", () => {
  setSettingsOpen(false);
  startGuide(); // asks for consent first when needed
});

// -- Help & more sub-page: sounds and voice, coaching level, offline copy -------------------
const PACK_NAME = { clean: "Clean", spooky: "Spooky" };
/** Voices to step through: the recorded ones, then the best few system voices. */
function voiceChoices() {
  return [...kokoroVoices().map((id) => KOKORO + id), ...voices.slice(0, 4).map((v) => v.name)];
}
function voiceName(v) {
  if (v?.startsWith(KOKORO)) return VOICES[v.slice(KOKORO.length)] || "Natural";
  return v ? `${v} (system)` : "None";
}
function renderMore() {
  setRowValue("sound-pack", settings.soundPack, PACK_NAME[settings.soundPack] || settings.soundPack);
  const v = selectedVoice();
  $("voice-pick").dataset.options = voiceChoices().join(",");
  setRowValue("voice-pick", v, voiceName(v));
  $("voice-pick").disabled = voiceChoices().length < 2;
  $("spoken-praise").checked = settings.spokenPraise;
  setRowValue("coaching-level", settings.coachingLevel, settings.coachingLevel === "light" ? "Light" : "Standard");
  $("test-value").textContent = test ? "Running" : "";
  renderOffline();
}
$("sound-pack").addEventListener("click", () => {
  const next = cycleValue($("sound-pack"));
  settings.soundPack = SOUND_PACKS[next] ? next : "clean";
  store.soundPack = settings.soundPack;
  save();
  renderMore();
  playBad(); // a preview of the new pack's reminder
});
$("voice-pick").addEventListener("click", () => {
  const next = cycleValue($("voice-pick"));
  if (!next) return;
  store.voiceName = next;
  save();
  renderMore();
  speak(TEST_LINE); // hear the new voice
});
$("spoken-praise").addEventListener("change", (e) => {
  settings.spokenPraise = e.target.checked;
  store.spokenPraise = settings.spokenPraise;
  save();
});
function setCoachingLevel(level) {
  settings.coachingLevel = normalizeLevel(level);
  policy.setLevel(settings.coachingLevel);
  store.coachingLevel = settings.coachingLevel;
  save();
  if (!$("more-view").hidden) renderMore();
}
$("coaching-level").addEventListener("click", () => setCoachingLevel(cycleValue($("coaching-level"))));

// The unlock card (frame 07): one line, one switch (or an arrow for My week), x. One per session,
// and x or using it means it is not shown again. The row stays in Settings either way.
const UNLOCK_CARD_TITLE = { breaks: "Break reminders", shoulders: "Shoulders & breathing", week: "My week" };
let unlockShownThisSession = false;
let unlockCardKey = null;
function finishUnlockCard() {
  if (!unlockCardKey) return;
  store.unlockCardsDone = [...new Set([...(store.unlockCardsDone || []), unlockCardKey])];
  save();
  unlockCardKey = null;
  $("unlock-card").hidden = true;
}
function unlockStep() {
  if (unlockShownThisSession || firstRun.open || !loopsStarted || !rules().setupCards) return;
  const key = nextUnlockCard(currentUnlockLevel(), store.unlockCardsDone || []);
  if (!key) return;
  unlockShownThisSession = true;
  unlockCardKey = key;
  $("unlock-title").textContent = UNLOCK_CARD_TITLE[key];
  const sw = $("unlock-switch");
  sw.hidden = key === "week";
  $("unlock-open").hidden = key !== "week";
  sw.checked = key === "breaks" ? breakRemindersOn() : settings.relaxReminders && hasConsent(store);
  $("unlock-card").hidden = false;
  renderRows();
}
$("unlock-switch").addEventListener("change", (e) => {
  const key = unlockCardKey;
  finishUnlockCard();
  if (key === "breaks") setBreakReminders(e.target.checked);
  else if (key === "shoulders") setShoulders(e.target.checked);
});
$("unlock-open").addEventListener("click", () => {
  finishUnlockCard();
  setSettingsOpen(true);
  showWeek(true);
});
$("unlock-close").addEventListener("click", finishUnlockCard);
renderRows();

let wasHeld = false; // the previous frame was scored by hold()
let settleUntil = -Infinity; // after a held run the restarted engine reads NO_PERSON until posture is clear

/** A suppressing activity just began: take back what was already on screen or queued. */
function clearCoaching() {
  clearTimeout(bannerTimer);
  clearTimeout(praiseTimer);
  $("banner").classList.remove("show", "relax");
}

/** Someone is in view. While held the machine state is frozen, so read the frame itself. */
function framePresent(out, held) {
  if (!held) return out.transition.state !== State.NO_PERSON;
  return !(out.result.kind === "unreliable" && out.result.reason === Reason.NO_POSE);
}

const owed = new OwedReminder(); // a reminder the engine emitted during grace, delivered when grace ends

function step() {
  if (!rules().frames) return;
  const nowMs = performance.now();
  const now = nowMs / 1000;
  let landmarks = null;
  if (demo) {
    // #103: during Learn the cues the demo shows the step being taught, not its slouch cycle.
    landmarks = practice ? practicePose(PRACTICE_STEPS[practice.i].metric, now) : demoPose(now);
  } else {
    if (!poseLandmarker || video.readyState < 2) return;
    // detectForVideo needs strictly increasing timestamps.
    const ts = Math.max(nowMs, lastVideoMs + 1);
    lastVideoMs = ts;
    let res;
    try {
      res = poseLandmarker.detectForVideo(video, ts);
    } catch (err) {
      stopCamera(err); // e.g. the GPU context was lost: stop instead of throwing every frame
      return;
    }
    const first = res.landmarks?.[0];
    // Some tasks-vision builds omit visibility; absent means "detected".
    landmarks = first ? first.map((p) => ({ x: p.x, y: p.y, visibility: p.visibility ?? 1 })) : null;
  }
  const held = activity.policy(now).engine === "hold";
  if (!wasHeld && held) {
    clearCoaching(); // a reminder banner or queued praise from just before the break
    owed.clear();
    if (postureSpeech) stopSpeech(); // a reminder that began just before the break must not talk over it
  }
  if (wasHeld && !held) {
    engine.restart(); // the activity ended: start fresh (calibration kept), nothing carries over
    settleUntil = now + activity.leave_s;
  }
  wasHeld = held;
  const out = held ? engine.hold(landmarks, now) : engine.process(landmarks, now);
  if (out.transition.state !== State.NO_PERSON) settleUntil = -Infinity;
  // While held, and while the restarted engine settles, read the frame itself: NO_PERSON there is not an absence.
  const settling = now < settleUntil;
  const present = settling || framePresent(out, held);
  if (activity.observePresence(present, now) && card) endCard("done"); // left the frame during a break
  const slouching = out.transition.state === State.BAD_CHIME || out.transition.state === State.BAD_VOICE;
  if (slouching && !lastBad) episode += 1; // a new slouch episode
  lastBad = slouching;
  if (inMode()) owed.clear();
  for (const nudge of out.nudges) {
    if (activity.inGrace(now) && (nudge === Nudge.CHIME || nudge === Nudge.VOICE)) owed.defer(nudge);
    else fire(nudge, out.cue);
  }
  // The engine advanced its reminder clock while grace kept the cue quiet: deliver the first one now if still slouching.
  const due = owed.take(activity.inGrace(now), slouching);
  if (due) fire(due, out.cue);
  account(out.transition.state, now);
  relaxStep(landmarks, out.transition.state, now);
  promptStep(out.transition.state, now, present);
  revealStep(out);
  setupStep();
  if (lastFrameT !== null) {
    const inst = 1 / Math.max(1e-3, now - lastFrameT);
    fpsEma = fpsEma ? 0.9 * fpsEma + 0.1 * inst : inst;
    document.body.dataset.fps = fpsEma.toFixed(0);
  }
  lastFrameT = now;
  if (!document.hidden) {
    // a throttled background tab has a low fps that says nothing about the setup
    const { result } = out;
    renderSetup(
      setup.update(
        {
          reason: result.kind === "unreliable" ? result.reason : null,
          state: out.transition.state,
          fps: fpsEma,
          metricsPresent: result.kind === "scored",
        },
        now,
      ),
    );
  }
  showReminderNote(policy.isCapped(now));
  if (!document.hidden) render(out, landmarks);
}

function visibleLoop() {
  // The demo has no detection to pace (its poses are scripted), so it renders at display rate; a real camera stays at VISIBLE_FPS.
  if (!document.hidden && (demo || performance.now() - lastProcessed >= 1000 / VISIBLE_FPS - 2)) {
    lastProcessed = performance.now();
    step();
  }
  requestAnimationFrame(visibleLoop);
}

// Background tabs pause requestAnimationFrame and throttle page timers; a
// worker's timer keeps ticking, so tracking (and reminders) continue at ~5 fps.
// The state machine is frame-rate independent, so the slower rate is fine.
function startHiddenTicker() {
  const src = `setInterval(() => postMessage(0), ${HIDDEN_TICK_MS});`;
  const worker = new Worker(URL.createObjectURL(new Blob([src], { type: "text/javascript" })));
  worker.onmessage = () => {
    if (document.hidden) step();
  };
}

// MediaPipe's web runtime uploads every camera frame through WebGL, even with
// the CPU delegate, so without WebGL it cannot run at all (it fails later, on
// the first frame, with "Cannot read properties of undefined (reading
// 'activeTexture')"). Probe first, on a throwaway canvas, and hand MediaPipe
// a fresh canvas of the kind that worked: OffscreenCanvas first (what it uses
// by default), then a regular canvas. null means no WebGL at all.
function webglCanvas() {
  const works = (canvas) => {
    try {
      return Boolean(canvas.getContext("webgl2") || canvas.getContext("webgl"));
    } catch {
      return false;
    }
  };
  if (typeof OffscreenCanvas !== "undefined" && works(new OffscreenCanvas(1, 1))) {
    return new OffscreenCanvas(1, 1);
  }
  if (works(document.createElement("canvas"))) return document.createElement("canvas");
  return null;
}

// "Not working?" is shown only while something calls for it: a slow or failed start
// ("start") or the user asking for it in Help ("asked"). The setup check keeps its
// status and fixes current, so they are right whenever the panel opens (#92).
const setupWhy = new Set();
function setSetupReason(why, on) {
  if (on) setupWhy.add(why);
  else setupWhy.delete(why);
  $("setup-panel").hidden = setupWhy.size === 0;
}
function showSetupPanel() {
  setSetupReason("start", true);
}

function showWebGLHelp(show) {
  $("webgl-help").hidden = !show;
  $("webgl-help").classList.toggle("show", show);
}

const CAMERA_LABEL = $("start-camera").textContent;

/** Without WebGL the camera still works, through the slower MoveNet fallback; say so up front. */
function applyWebGLState() {
  const ok = Boolean(webglCanvas());
  if (!ok) showSetupPanel();
  $("start-camera").textContent = CAMERA_LABEL;
  showWebGLHelp(!ok);
  return ok;
}

/**
 * Pose detector for browsers without WebGL: MoveNet Lightning on the TF.js
 * WASM backend. The sync wrapper lives in movenet-adapter.js.
 */
let moveNetDetector = null; // TF.js detector, built once and reused across retries

async function loadMoveNet(onProgress = () => {}, onDownloaded = () => {}, gate = Promise.resolve()) {
  if (moveNetDetector) return wrapMoveNet(moveNetDetector);
  // Warm the HTTP cache with byte-level progress; TF.js then reads from it.
  const files = [
    "movenet_bundle.mjs",
    "tfjs-backend-wasm-simd.wasm",
    "model.json",
    "group1-shard1of2.bin",
    "group1-shard2of2.bin",
  ];
  await downloadAll(
    files.map((f) => new URL(`${MOVENET_DIR}/${f}`, location.href).href),
    onProgress,
  );
  onDownloaded();
  await gate; // building the detector blocks the page; let the live video paint first
  const tfm = await import(`${MOVENET_DIR}/movenet_bundle.mjs`);
  tfm.setWasmPaths({
    "tfjs-backend-wasm.wasm": `${MOVENET_DIR}/tfjs-backend-wasm.wasm`,
    "tfjs-backend-wasm-simd.wasm": `${MOVENET_DIR}/tfjs-backend-wasm-simd.wasm`,
    "tfjs-backend-wasm-threaded-simd.wasm": `${MOVENET_DIR}/tfjs-backend-wasm-simd.wasm`,
  });
  if (!(await tfm.tf.setBackend("wasm"))) throw new Error("The WASM backend did not start");
  await tfm.tf.ready();
  const detector = await tfm.createDetector(tfm.SupportedModels.MoveNet, {
    modelType: tfm.movenet.modelType.SINGLEPOSE_LIGHTNING,
    modelUrl: `${MOVENET_DIR}/model.json`,
    enableSmoothing: true,
  });
  moveNetDetector = detector;
  return wrapMoveNet(detector);
}

/** One wrapper per camera session. A detector that failed is dropped, so the next try builds a fresh one. */
function wrapMoveNet(detector) {
  return createMoveNetDetector(detector, { onError: () => discardMoveNet(detector) });
}

function discardMoveNet(detector) {
  if (moveNetDetector !== detector) return;
  moveNetDetector = null;
  try {
    detector.dispose?.();
  } catch {
    // already broken; nothing more to release
  }
}

async function loadModel(canvas, onProgress = () => {}, onDownloaded = () => {}, gate = Promise.resolve()) {
  const { FilesetResolver, PoseLandmarker } = await import(`${TASKS_VISION}/vision_bundle.mjs`);
  const fileset = await FilesetResolver.forVisionTasks(`${TASKS_VISION}/wasm`);
  // Download the model and the runtime with progress, then hand both over as
  // buffers so nothing is fetched twice. The bundle passes a pre-existing
  // self.Module to the Emscripten loader, which uses Module.wasmBinary
  // instead of fetching the .wasm file.
  const [model, wasm] = await downloadAll([MODEL_URL, new URL(fileset.wasmBinaryPath, location.href).href], onProgress);
  onDownloaded();
  await gate; // creating the model blocks the page; let the live video paint first
  const options = (delegate) => ({
    baseOptions: { modelAssetBuffer: new Uint8Array(model), delegate },
    runningMode: "VIDEO",
    numPoses: 1,
    canvas,
  });
  const create = async (delegate) => {
    self.Module = { wasmBinary: new Uint8Array(wasm) };
    try {
      return await PoseLandmarker.createFromOptions(fileset, options(delegate));
    } finally {
      self.Module = undefined; // the bundle clears it on success; make sure on failure too
    }
  };
  try {
    return await create("GPU");
  } catch {
    return await create("CPU");
  }
}

let lowAccuracy = false; // tracking runs on the MoveNet fallback

/** Run one detection before showing the dashboard: the first call compiles
 *  the model for the GPU (or CPU) and can block for seconds. Doing it here,
 *  under "Start the model", means the page never sits frozen on a blank
 *  dashboard. Throws if the detector fails on a real frame. */
async function warmUp(detector, isMoveNet) {
  await nextPaint();
  if (video.readyState < 2) return;
  if (isMoveNet) {
    const tf = moveNetDetector;
    if (!tf) return;
    try {
      await tf.estimatePoses(video, { flipHorizontal: false });
    } catch (err) {
      discardMoveNet(tf); // do not reuse a detector that failed on a real frame
      throw err;
    }
    return;
  }
  const ts = Math.max(performance.now(), lastVideoMs + 1);
  lastVideoMs = ts;
  detector.detectForVideo(video, ts);
}

const loadSteps = new LoadSteps();

function showError(message) {
  $("start-error").textContent = message;
  if (message) {
    // The message and the Try again button must be in view, also on a small screen
    // where the page may be scrolled to the start button.
    $("start-overlay").scrollTop = 0;
    $("start-overlay").scrollIntoView({ block: "nearest" });
    logProblem(message);
  }
}

// -- bug reports ----------------------------------------------------------------
// "Report a bug" opens GitHub's new-issue form pre-filled with what a
// maintainer needs: browser, WebGL, app state, settings and recent errors.
// No login or backend; the user reviews and submits it on GitHub. Never
// includes video, images, landmarks or stored sounds.

const ISSUE_URL = "https://github.com/arii/backup/issues/new";
// Issue title from a "HH:MM:SS message" entry: drop the time, cut at a word boundary.
function bugTitle(entry, max = 71) {
  const msg = entry.slice(9);
  if (msg.length <= max) return msg;
  const cut = msg.slice(0, max);
  return `${cut.slice(0, Math.max(cut.lastIndexOf(" "), 1)).trimEnd()}...`;
}

const problems = []; // last few errors, newest last

function logProblem(message) {
  problems.push(`${new Date().toISOString().slice(11, 19)} ${sanitizeError(message)}`);
  if (problems.length > 8) problems.shift();
}
window.addEventListener("error", (e) => logProblem(e.message || e.error || "error"));
window.addEventListener("unhandledrejection", (e) => logProblem(`unhandled: ${e.reason?.message || e.reason}`));

function webglInfo() {
  try {
    const c = document.createElement("canvas");
    const gl = c.getContext("webgl2") || c.getContext("webgl");
    if (!gl) return typeof OffscreenCanvas !== "undefined" && webglCanvas() ? "offscreen only" : "unavailable";
    return webglSummary(gl); // no GPU renderer string: it adds to a browser fingerprint
  } catch (err) {
    return `probe failed: ${sanitizeError(err.message || err)}`;
  }
}

function bugReportUrl() {
  const cal = engine.calibration.status;
  const v = chosenVoice();
  const diag = [
    `- Page: ${location.origin}${location.pathname}`,
    `- Browser: ${navigator.userAgent}`,
    `- Language: ${navigator.language} | Screen: ${screen.width}x${screen.height} @${devicePixelRatio}x`,
    `- WebGL: ${webglInfo()}`,
    `- Source: ${document.body.dataset.source || "-"} | State: ${$("state-badge").textContent} | fps: ${document.body.dataset.fps || "-"} | paused: ${isPaused()}`,
    `- Mode: ${settings.mode} | specific cues: ${settings.specificCues} | spoken praise: ${settings.spokenPraise}`,
    `- Voice: ${selectedVoice() || "none"}${v ? ` | system fallback ${v.name} (${v.lang})` : ""} | ${voices.length} system voices | clips ${clipManifest ? "loaded" : "missing"}`,
    `- Calibration: good ${cal.good ? "recorded" : "no"}, slouch ${cal.bad ? "recorded" : "no"}, needle ${cal.calibrated ? "calibrated" : "default"}`,
    `- Custom sounds: bad ${custom.bad ? "yes" : "no"}, good ${custom.good ? "yes" : "no"}`,
    `- Reminders this session: ${reminderCount}`,
    `- Help opened this session: ${helpOpens}`,
    `- Load marks (ms): ${JSON.stringify(loadMarks)}`,
  ].join("\n");
  const body = buildBugReportBody({ diag, problems });
  const title = problems.length ? `Bug: ${bugTitle(problems[problems.length - 1])}` : "Bug: ";
  const params = new URLSearchParams({ title, body: body.slice(0, 6000), labels: "bug" });
  return `${ISSUE_URL}?${params}`;
}

// Fill the link at click time so it carries the current state.
for (const link of [$("report-bug"), ...document.querySelectorAll(".report-bug-link")]) {
  link.title = BUG_LINK_TIP;
  link.addEventListener("click", () => {
    link.href = bugReportUrl();
  });
}

/** Tracking failed after it started: release the camera, bring the start
 *  screen back with the reason, and leave the demo available. */
function stopCamera(err) {
  setPaused(false);
  guide.cancel();
  breath.cancel?.();
  activity.stop(ACTIVITY.BREATH_CHECK, clockS(), "cancelled");
  if (card) endCard("done"); // an eye break or demo card ends its activity with it
  if (practice) stopPractice();
  setup.reset();
  $("pause").disabled = true;
  setTrackingControls(false);
  poseLandmarker = null;
  const stream = video.srcObject;
  if (stream) for (const t of stream.getTracks()) t.stop();
  video.srcObject = null;
  lowAccuracy = false;
  $("lowpower-note").style.display = "none";
  document.body.classList.add("pre-start");
  showSetupPanel();
  showWebGLHelp(!webglCanvas());
  showError(`Tracking stopped: ${err.message || err}. Reload to try again, or try the demo.`);
  $("start-status").textContent = "";
  loadSteps.finish();
  $("start-overlay").classList.remove("compact");
  $("start-overlay").style.display = "";
  $("start-camera").disabled = $("start-demo").disabled = false;
  applyWebGLState();
}

let loopsStarted = false;

function begin(source) {
  document.body.dataset.source = source; // the bug report reads it; not shown (#98 v3)
  document.body.classList.remove("pre-start"); // status, Pause and Settings appear now
  setSetupReason("start", false); // started: a slow or failed earlier attempt no longer applies
  $("start-overlay").style.display = "none";
  showWebGLHelp(false);
  $("lowpower-note").style.display = lowAccuracy ? "block" : "none";
  $("pause").disabled = false;
  setTrackingControls(true);
  $("state-badge").focus(); // the Start button is gone: move focus to the status badge
  if (loopsStarted) return; // a retry after stopCamera reuses the running loops
  loopsStarted = true;
  if (source !== "demo" && !store.calibration) undoStep(store, "calibrate"); // saved only in the demo: ask again
  recordSession(store, Date.now());
  save();
  resetPromptSession();
  requestAnimationFrame(visibleLoop);
  startHiddenTicker();
}

/** The model load started by the last click, kept so a retry after a camera
 *  problem does not download it again. Dropped if it failed. */
let preloaded = null;

function preloadModel(canvas, gate) {
  if (preloaded && !preloaded.failed) return preloaded;
  const entry = { kind: canvas ? "mediapipe" : "movenet", downloaded: false, failed: false, promise: null };
  const onDownloaded = () => {
    entry.downloaded = true;
    markLoad("downloadDone");
    loadSteps.done("download");
  };
  const progress = (got, total) => loadSteps.progress(got, total);
  entry.promise = canvas ? loadModel(canvas, progress, onDownloaded, gate) : loadMoveNet(progress, onDownloaded, gate);
  entry.promise.catch(() => {
    entry.failed = true; // handled where it is awaited; this only stops an unhandled rejection
  });
  preloaded = entry;
  return entry;
}

/** Frames are playing: show the video and shrink the card to a strip under it. */
function showLivePreview() {
  $("start-overlay").classList.add("compact");
  // On a narrow screen the page was scrolled to the Start button; the card shrinks, so bring the video back.
  video.scrollIntoView({ block: "nearest" });
  $("start-status").textContent = "Camera is on. Loading the pose model, no video leaves your computer.";
}

/** Open the camera. A remembered device is tried first; if it is gone or cannot be used,
 *  forget it and ask again for the default camera. The id of the one that works is remembered. */
async function openCamera() {
  const ask = (id) => navigator.mediaDevices.getUserMedia({ video: cameraConstraints(id), audio: false });
  let stream;
  const remembered = typeof store.cameraDeviceId === "string" ? store.cameraDeviceId : "";
  try {
    stream = await ask(remembered);
  } catch (err) {
    if (!remembered || !isDeviceGoneError(err)) throw err;
    delete store.cameraDeviceId;
    save();
    stream = await ask("");
  }
  const id = stream.getVideoTracks?.()[0]?.getSettings?.().deviceId;
  if (typeof id === "string" && id && id !== store.cameraDeviceId) {
    store.cameraDeviceId = id;
    save();
  }
  return stream;
}

$("start-camera").addEventListener("click", async () => {
  ensureAudio(); // inside the click: browsers only allow audio after a gesture
  for (const k of Object.keys(loadMarks)) delete loadMarks[k];
  loadMarks.click = Math.round(performance.now());
  engine.setDriftBaselineEnabled(true); // back from the demo: real tracking adapts again
  $("start-camera").disabled = $("start-demo").disabled = true;
  showError("");
  showWebGLHelp(false);
  $("start-status").textContent = "";
  $("start-overlay").classList.remove("compact");
  loadSteps.open();
  // The model waits for this before its first long, blocking step (see loadModel).
  // It opens once the live video has painted, or when the start fails.
  let openGate = () => {};
  const gate = new Promise((r) => {
    openGate = r;
  });
  try {
    // MediaPipe needs WebGL; without it (or if it fails to start) use the MoveNet fallback.
    const canvas = webglCanvas();
    lowAccuracy = false;
    // Start the download first and let it run while the browser asks for the camera.
    // getUserMedia is still only called here, after the click.
    loadSteps.start("camera");
    const entry = preloadModel(canvas, gate);
    if (!canvas) loadSteps.label("download", "WebGL is off: downloading the simpler pose model");
    if (entry.downloaded) loadSteps.done("download");
    else loadSteps.start("download");
    const stream = await openCamera();
    video.srcObject = stream;
    await video.play();
    markLoad("streamPlaying");
    loadSteps.done("camera");
    showLivePreview();
    await nextPaint();
    openGate();
    if (canvas) {
      try {
        poseLandmarker = await entry.promise;
        loadSteps.start("start", { elapsed: true });
        await warmUp(poseLandmarker, false);
      } catch (err) {
        poseLandmarker = null;
        logProblem(`MediaPipe did not start, using the fallback: ${err.message || err}`);
      }
    }
    if (!poseLandmarker) {
      let detector;
      if (canvas) {
        loadSteps.label("download", "The main model didn't start: downloading the simpler one");
        loadSteps.start("download");
        preloaded = null; // the failed MediaPipe entry must not be reused
        detector = await preloadModel(null).promise;
      } else {
        detector = await entry.promise;
      }
      loadSteps.start("start", { elapsed: true });
      poseLandmarker = detector;
      await warmUp(detector, true);
      lowAccuracy = true;
    }
    markLoad("warmUpDone");
    loadSteps.done("start");
    loadSteps.finish();
    $("start-overlay").classList.remove("compact");
    begin("webcam");
  } catch (err) {
    openGate();
    loadSteps.fail();
    $("start-overlay").classList.remove("compact");
    $("start-status").textContent = "";
    const stream = video.srcObject;
    if (stream) for (const t of stream.getTracks()) t.stop();
    video.srcObject = null;
    poseLandmarker = null;
    showSetupPanel();
    showError(cameraErrorMessage(err));
    $("start-camera").disabled = $("start-demo").disabled = false;
    applyWebGLState();
    $("start-camera").textContent = "Try again"; // a retry reuses the model that is already loading
    // #102: a camera problem is not a model problem: no model steps, and the demo is the way on.
    const camera = cameraProblem(err) !== null;
    $("load-steps").hidden = camera;
    $("start-demo").classList.toggle("text-link", !camera);
    $("start-camera").classList.toggle("secondary", camera);
  }
});

// Probe as soon as the page loads, before any click.
applyWebGLState();

$("start-demo").addEventListener("click", () => {
  ensureAudio();
  loadSteps.finish();
  demo = true;
  engine.setDriftBaselineEnabled(false); // synthetic poses: no desk baseline to adapt (#67)
  video.style.display = "none";
  $("demo-note").style.display = "flex";
  begin("demo");
  // Nothing to find in a scripted demo: show the gauge now, not an empty "Finding you..." for 10 s.
  reveal.settleNow();
  applyReveal();
});

/** Leave the demo: back to the start screen with the camera and demo both on offer again. */
function stopDemo() {
  if (!demo) return;
  setPaused(false);
  guide.cancel();
  breath.cancel?.();
  activity.stop(ACTIVITY.BREATH_CHECK, clockS(), "cancelled");
  if (card) endCard("done");
  if (practice) stopPractice();
  setup.reset();
  if (firstRun.open) {
    // The setup card belongs to a running session. Hide it without marking the guide done,
    // so a real camera start still gets its first-run guide.
    firstRun.open = false;
    $("first-run").hidden = true;
    setGettingStarted(false);
    activity.stop(ACTIVITY.SETUP, clockS(), "cancelled");
  }
  demo = false;
  engine.setDriftBaselineEnabled(true);
  engine.restart();
  clearCoaching();
  owed.clear();
  reveal.reset();
  applyReveal();
  $("demo-note").style.display = "none";
  video.style.display = "";
  $("pause").disabled = true;
  setTrackingControls(false);
  document.body.classList.add("pre-start");
  delete document.body.dataset.source;
  setSetupReason("start", false); // the start screen is clean: no "Not working?" card until something fails
  updateDemoAvatar(null); // demo is already false, so this hides the avatar
  drawSkeleton(null, "");
  const badge = $("state-badge");
  badge.className = "badge post-start"; // renderBadge drops post-start, which would show a stale state up top
  badge.textContent = "waiting";
  syncTabTitle();
  $("start-status").textContent = "";
  $("start-overlay").classList.remove("compact");
  $("start-overlay").style.display = "";
  $("start-camera").disabled = $("start-demo").disabled = false;
  applyWebGLState();
  $("start-demo").focus();
}
$("stop-demo").addEventListener("click", stopDemo);

// -- first session (storyboard v3 frames 02 and 03, #98) -------------------------------------
// Two cards in #first-run, one line and one button each, x to skip: "Head and shoulders in
// frame" while tracking settles, then "Sit comfortably" with Save my posture, which closes the
// card when the posture is saved. The header icons show once the card has gone
// (body.getting-started). The sequence waits while a posture reminder, a pause, practice or
// a breath check is on. "Show the intro again" (Not working?) reopens it by hand.

const firstRun = { step: 0, open: false, manual: false };
let firstRunTried = false; // Save was pressed on this card (an old engine error must not show)

function renderFirstRun() {
  const step = SETUP_STEPS[firstRun.step];
  $("first-run-count").textContent = `${firstRun.step + 1} of ${SETUP_STEPS.length}`;
  $("first-run-heading").textContent = step.title;
  $("first-run-next").textContent = step.action;
  $("first-run-next").disabled = false;
  setFirstRunNote(step.key === "calibrate" ? calibrateNote(engine.calibration.status, firstRunTried) : "");
}

function setFirstRunNote(text) {
  const el = $("first-run-note");
  if (el.textContent !== text) el.textContent = text;
  el.hidden = text === "";
}

function setGettingStarted(on) {
  document.body.classList.toggle("getting-started", on);
}

function openFirstRun(key = SETUP_STEPS[0].key, manual = false) {
  firstRun.open = true;
  firstRun.manual = manual;
  firstRunTried = false;
  firstRun.step = SETUP_STEPS.findIndex((s) => s.key === key);
  viewSetupStep(store, key);
  save();
  $("first-run").hidden = false;
  setGettingStarted(!manual);
  if (!manual) activity.start(ACTIVITY.SETUP, clockS()); // no reminders, praise or prompts until set up
  renderFirstRun();
  // Getting started hides the badge that Start focused (frame 02): rather than leave focus
  // nowhere, put it on the card. Focus anywhere else that is still on screen stays put.
  const active = document.activeElement;
  if (!manual && (!active || active === document.body || !active.getClientRects().length)) focusFirstRunHeading();
}

function closeFirstRun() {
  firstRun.open = false;
  $("first-run").hidden = true;
  setGettingStarted(false);
  activity.stop(ACTIVITY.SETUP, clockS(), "completed");
  if (!store.firstRunDone) {
    store.firstRunDone = true;
    save();
  }
}

/** After a card is done or skipped: the next one, or close. Returns true when a card is showing. */
function advanceSetup() {
  // Straight on to the next card, even before tracking has settled: closing in between showed
  // the header icons and started coaching, then hid them again when "Sit comfortably" came.
  const key = firstRun.manual
    ? SETUP_STEPS.slice(firstRun.step + 1).find((s) => !setupProgress(store).done.includes(s.key))?.key
    : nextSetupStep(store, STAGES.settled);
  if (!key) {
    closeFirstRun();
    return false;
  }
  openFirstRun(key, firstRun.manual);
  return true;
}

function focusFirstRunHeading() {
  $("first-run-heading").tabIndex = -1;
  $("first-run-heading").focus();
}

/** True while something else owns the screen (a pause, the silent test, practice, a break):
 *  the sequence hides and waits. A slouch reading no longer hides it: during getting started
 *  (ACTIVITY.SETUP) there are no reminders to make room for. */
function setupBlocked() {
  // #103: the Learn the cues panel stays up for its breath-check offer after practice ends;
  // one card at a time, so the setup card waits until it closes.
  return !rules().setupCards || document.body.classList.contains("practice-done");
}

/** Runs every tracked frame: opens the next due card on its own (no focus move), hides it while blocked. */
function setupStep() {
  const blocked = setupBlocked();
  if (firstRun.open) {
    $("first-run").hidden = blocked;
    // A pause ends ACTIVITY.SETUP; when the card shows again, so does its quiet.
    if (!blocked && !firstRun.manual && !activity.isActive(ACTIVITY.SETUP)) activity.start(ACTIVITY.SETUP, clockS());
    return;
  }
  if (blocked || !loopsStarted) return;
  const key = nextSetupStep(store, currentStage());
  if (key) openFirstRun(key);
}

/** Step 2: Save my posture records, then the card closes by itself once the posture is saved. */
let firstRunSaving = false;
function updateFirstRunLive() {
  if (!firstRun.open || SETUP_STEPS[firstRun.step].key !== "calibrate") return;
  const c = engine.calibration.status;
  $("first-run-next").disabled = !!c.active;
  setFirstRunNote(calibrateNote(c, firstRunTried));
  if (firstRunSaving && !c.active) {
    firstRunSaving = false;
    if (c.good && !c.error) finishFirstRunStep();
  }
}

function finishFirstRunStep() {
  completeStep(store, SETUP_STEPS[firstRun.step].key);
  save();
  if (advanceSetup()) focusFirstRunHeading();
  else $("state-badge").focus();
}

$("first-run-next").addEventListener("click", () => {
  if (SETUP_STEPS[firstRun.step].key === "calibrate") {
    firstRunSaving = true;
    firstRunTried = true;
    engine.calibration.start("good");
    return;
  }
  finishFirstRunStep();
});
$("first-run-skip").addEventListener("click", () => {
  skipStep(store, SETUP_STEPS[firstRun.step].key);
  save();
  if (advanceSetup()) focusFirstRunHeading();
  else $("state-badge").focus();
});
$("intro-again").addEventListener("click", () => {
  if (!loopsStarted) {
    $("practice-msg").textContent = "Start the webcam or the demo first, then the intro can show your live framing.";
    return;
  }
  openFirstRun(SETUP_STEPS[0].key, true);
  $("first-run").scrollIntoView({ block: "nearest" });
  focusFirstRunHeading();
});

// -- onboarding, safety notice and help -----------------------------------------
// First run: one welcome card in the start overlay, with the full safety notice on
// demand. The Start buttons stay hidden until the notice is acknowledged
// (store.ackVersion). Help shows the welcome again any time. Text lives in
// onboarding.js; see .kiro/specs/onboarding-and-setup.

const LAST_STEP = ONBOARDING_STEPS.length - 1;
const ob = { step: needsAckOnly(store) ? LAST_STEP : 0, ackOnly: needsAckOnly(store) };

function fillList(ul, points) {
  ul.replaceChildren(
    ...points.map((text) => {
      const li = document.createElement("li");
      li.textContent = text;
      return li;
    }),
  );
}

function renderOnboardingStep() {
  const step = ONBOARDING_STEPS[ob.step];
  const last = ob.step === LAST_STEP;
  $("ob-count").textContent = ob.ackOnly ? "Safety notice" : "Welcome to BackUp";
  $("ob-title").textContent = ob.ackOnly ? "Before you start" : step.title;
  fillList($("ob-body"), ob.ackOnly ? SAFETY_POINTS : step.points);
  fillList($("ob-safety"), SAFETY_POINTS);
  $("ob-back").hidden = ob.step === 0 || ob.ackOnly;
  $("ob-next").hidden = last;
  $("ob-ack").hidden = !last;
  // #108: the full notice is open on the step that asks for the acknowledgement, so it can't be accepted unseen.
  $("onboarding").querySelector("details").open = last;
}

let introAgain = false; // the footer's Intro link (#106): show the welcome again before starting
function showStartControls() {
  const need = introAgain || shouldShowOnboarding(store);
  $("onboarding").hidden = !need;
  $("start-title").hidden = need; // make room for the card
  $("start-actions").hidden = need;
  if (need) renderOnboardingStep();
}

$("ob-next").addEventListener("click", () => {
  ob.step = Math.min(LAST_STEP, ob.step + 1);
  renderOnboardingStep();
  $("ob-title").focus();
});
$("ob-back").addEventListener("click", () => {
  ob.step = Math.max(0, ob.step - 1);
  renderOnboardingStep();
  $("ob-title").focus();
});
$("ob-ack").addEventListener("click", () => {
  introAgain = false;
  markOnboarded(store);
  save();
  showStartControls();
  ($("start-camera").disabled ? $("start-demo") : $("start-camera")).focus();
});
showStartControls();
// First visit: start keyboard focus on the card, so Tab reaches Next before the header controls.
if (!$("onboarding").hidden) $("ob-title").focus({ preventScroll: true });

// Help opens this session, for the bug report only (memory, never stored or sent) (#95).
let helpOpens = 0;
function openHelp() {
  helpOpens += 1;
  const body = $("help-body");
  body.replaceChildren();
  for (const step of ONBOARDING_STEPS) {
    const h = document.createElement("h3");
    h.textContent = step.title;
    const ul = document.createElement("ul");
    fillList(ul, step.points);
    body.append(h, ul);
  }
  $("help-msg").textContent = "";
  $("help-dialog").showModal();
}
$("help-close").addEventListener("click", () => $("help-dialog").close());
$("help-notworking").addEventListener("click", () => {
  $("help-dialog").close();
  setSettingsOpen(false);
  setSetupReason("asked", true);
  $("setup-panel").scrollIntoView({ block: "nearest" });
  $("setup-title").focus();
});
// "Hide" puts away what the user opened by hand; a live fault or start problem keeps it shown.
$("setup-close").addEventListener("click", () => {
  setSetupReason("asked", false);
  $("practice-msg").textContent = "";
  if (!$("setup-panel").hidden) return;
  $("settings-btn").focus();
});
$("help-practice").addEventListener("click", () => {
  const msg = startPractice();
  $("help-msg").textContent = msg;
  if (!msg) {
    $("help-dialog").close();
    setSettingsOpen(false);
    $("practice-next").focus();
  }
});

fillList($("safety-list"), SAFETY_POINTS);
$("safety-link").addEventListener("click", (e) => {
  e.preventDefault();
  $("safety-dialog").showModal();
});
$("safety-close").addEventListener("click", () => $("safety-dialog").close());
// #106: the welcome and safety notice are always reachable, before starting too.
$("intro-link").addEventListener("click", (e) => {
  e.preventDefault();
  if (loopsStarted) {
    openHelp();
    return;
  }
  introAgain = true;
  ob.step = 0;
  ob.ackOnly = false;
  showStartControls();
  $("ob-title").focus();
});
$("about-link").addEventListener("click", (e) => {
  e.preventDefault();
  $("about-dialog").showModal();
});
$("about-close").addEventListener("click", () => $("about-dialog").close());
// A slow camera, download or start also brings up the "Not working?" help.
new MutationObserver(() => {
  // Only while starting: a hint left over after tracking began is not a reason to show it.
  if ($("load-hint").textContent && !loopsStarted) showSetupPanel();
}).observe($("load-hint"), { childList: true, characterData: true, subtree: true });

// -- setup check ("Not working?") ----------------------------------------------------

const setup = new SetupCheck();
let setupShown = null;

function renderSetup({ diagnosis, fixes }) {
  if (diagnosis === setupShown) return;
  setupShown = diagnosis;
  const status = $("setup-status");
  status.textContent = DIAGNOSIS_TEXT[diagnosis];
  status.className = diagnosis === Diagnosis.OK ? "ok" : diagnosis === Diagnosis.CHECKING ? "" : "fault";
  fillList($("setup-fixes"), fixes);
}

// -- learn the cues (practice) ---------------------------------------------------------
// Steps through each cue and lights the matching tile. Silent like the retention
// test: no sounds, no reminders, and nothing is added to the daily totals.

let practice = null; // { i } while practising
let practiceLive = "";

function renderPracticeStep() {
  const step = PRACTICE_STEPS[practice.i];
  $("practice-count").textContent = `Step ${practice.i + 1} of ${PRACTICE_STEPS.length}`;
  $("practice-text").textContent = step.text;
  $("practice-next").textContent = practice.i === PRACTICE_STEPS.length - 1 ? "Finish" : "Next";
  practiceLive = "";
}

function startPractice() {
  if (!loopsStarted) return "Start the webcam or the demo first, then you can practise the cues.";
  if (isPaused()) return "Resume tracking first.";
  if (test) return "Stop the silent test first.";
  if (practice) return "";
  if (!activity.start(ACTIVITY.PRACTICE, clockS())) return "Finish the current activity first.";
  $("practice-breath").hidden = true;
  document.body.classList.remove("practice-done");
  $("practice-stop").textContent = "Stop practice";
  breath.cancel();
  stopSpeech();
  $("banner").classList.remove("show");
  cancelPrompts();
  practice = { i: 0 };
  document.body.classList.add("practicing");
  applyReveal();
  renderPracticeStep();
  return "";
}

function stopPractice() {
  practice = null;
  activity.stop(ACTIVITY.PRACTICE, clockS());
  document.body.classList.remove("practicing");
  applyReveal(); // the gauge goes back to its stage
  for (const metric of ["head_forward", "head_tilt", "shoulder_level", "trunk_lean"])
    $(`t-${metric}`).classList.remove("focus");
  engine.restart(); // practice postures must not carry into a reminder
  bracing.reset();
  relaxLastT = null;
  session.lastT = null;
  setup.reset();
}

function renderPractice(result) {
  const step = PRACTICE_STEPS[practice.i];
  const cfg = DEFAULTS.scoring;
  let live;
  if (!step.metric) {
    live = "Hold this for a moment. The tiles show your angles.";
  } else {
    const v =
      result.kind === "scored" && (step.metric !== "trunk_lean" || result.trunk_lean_available)
        ? result[`${step.metric}_deg`]
        : null;
    live =
      v !== null && Math.abs(v) > cfg[`${step.metric}_deg`]
        ? "That is what this cue looks like. Sit tall again, then press Next."
        : "Try it. The highlighted tile lights up when it is past your threshold.";
  }
  if (live !== practiceLive) {
    practiceLive = live;
    $("practice-live").textContent = live;
  }
  for (const metric of ["head_forward", "head_tilt", "shoulder_level", "trunk_lean"]) {
    $(`t-${metric}`).classList.toggle("focus", metric === step.metric);
  }
}

$("practice-start").addEventListener("click", () => {
  const msg = startPractice();
  $("practice-msg").textContent = msg;
  if (!msg) $("practice-next").focus();
});
$("practice-next").addEventListener("click", () => {
  if (!practice) return;
  if (practice.i >= PRACTICE_STEPS.length - 1) {
    stopPractice();
    // The panel stays up for the breath-check offer until the user picks it or closes it.
    document.body.classList.add("practice-done");
    $("practice-count").textContent = "";
    $("practice-text").textContent = "Practice finished. Try a breath check next?";
    $("practice-live").textContent = "";
    $("practice-breath").hidden = false;
    $("practice-breath").focus(); // Next is about to hide; keep focus in the panel
    $("practice-next").hidden = true;
    $("practice-stop").textContent = "Close";
    return;
  }
  practice.i += 1;
  renderPracticeStep();
});
function closePracticeDone() {
  document.body.classList.remove("practice-done");
  $("practice-breath").hidden = true;
  $("practice-next").hidden = false;
  $("practice-stop").textContent = "Stop practice";
}
$("practice-stop").addEventListener("click", () => {
  if (practice) stopPractice();
  closePracticeDone();
});
$("practice-breath").addEventListener("click", () => {
  closePracticeDone();
  startBreathCheck();
});

// -- your progress -----------------------------------------------------------------------
// A personal review of coached time per day and the silent-test trend. Aggregation
// is in progress.js. No streaks, no ranking; see .kiro/specs/progress-review.

let lastProgressRender = 0;

// My week (frame 12): seven bars, today last, from progress.js. Nothing else.
function renderProgress() {
  lastProgressRender = Date.now();
  const s = summarize(store, Date.now());
  $("progress-days").replaceChildren(
    ...s.days.map((d) => {
      const col = document.createElement("div");
      col.className = "week-bar";
      col.setAttribute("role", "listitem");
      const track = document.createElement("div");
      track.className = "track";
      const fill = document.createElement("div");
      fill.style.height = `${d.pct ?? 0}%`;
      track.append(fill);
      if (d.pct !== null) {
        const bad = document.createElement("div");
        bad.className = "bad";
        bad.style.height = `${100 - d.pct}%`;
        track.append(bad);
      }
      const label = document.createElement("span");
      const date = new Date(`${d.date}T00:00:00Z`);
      label.textContent = date.toLocaleDateString(undefined, { weekday: "narrow", timeZone: "UTC" });
      col.setAttribute(
        "aria-label",
        `${date.toLocaleDateString(undefined, { weekday: "long", timeZone: "UTC" })}: ${d.pct === null ? "no data" : `${d.pct}% good, ${100 - d.pct}% slouching`}`,
      );
      const pctText = document.createElement("small");
      pctText.textContent = d.pct === null ? "-" : `${d.pct}%`;
      col.append(track, label, pctText);
      return col;
    }),
  );
  renderDayList(s);
  renderWeekStats(s);
  renderBestDays(s);
}

/** Day by day, in words and numbers, so good vs bad never depends on colour alone. A day with
 *  too little tracked time is "not enough tracked", never counted as bad. */
function renderDayList(s) {
  const rows = [...s.days].reverse().map((d) => {
    const li = document.createElement("li");
    const date = new Date(`${d.date}T00:00:00Z`);
    const name = date.toLocaleDateString(undefined, {
      weekday: "short",
      month: "short",
      day: "numeric",
      timeZone: "UTC",
    });
    const b = document.createElement("b");
    b.textContent = name;
    if (d.tracked_s <= 0) {
      li.append(b, " no data (camera was not tracking)");
    } else if (d.pct === null) {
      li.append(b, ` not enough tracked yet (${minutesText(d.tracked_s)})`);
    } else {
      const bad = d.tracked_s - d.good_s;
      li.append(
        b,
        ` ${d.pct}% good: ${minutesText(d.good_s)} good, ${minutesText(bad)} slouching, ${minutesText(d.tracked_s)} tracked`,
      );
    }
    return li;
  });
  $soft("day-list").replaceChildren(...rows);
}

const minutesText = (sec) =>
  sec >= 3600 ? `${Math.floor(sec / 3600)} h ${Math.round((sec % 3600) / 60)} min` : `${Math.round(sec / 60)} min`;

/** Three numbers under the bars (frame 12 plus the totals people missed): good time so far,
 *  the latest silent test, and water this week. A missing number shows a dash. */
/** Counts exist only for days recorded since episodes were counted. When some tracked days of the
 *  week lack them, say so, so a partial total never reads as the whole week. */
function coverage(s) {
  const tracked = s.days.filter((d) => d.tracked_s > 0).length;
  return s.count_days_known < tracked ? `, ${s.count_days_known} of ${tracked} days counted` : "";
}
function renderWeekStats(s) {
  const n = s.tests.length;
  const latest = !n ? "-" : n === 1 ? `${s.tests[0].pct}%` : `${s.tests[0].pct}\u2192${s.tests[n - 1].pct}%`;
  const cells = [
    [s.empty ? "-" : minutesText(s.cumulative_good_s), "good so far"],
    [latest, n > 1 ? "silent tests" : "silent test"],
    [String(s.water_7d || 0), "drinks this week"],
    [s.empty ? "-" : minutesText(s.bad_s_7d), "slouching this week"],
    // Episodes exist only for days recorded since they were counted: older days stay unknown, not 0.
    [s.slouch_events_7d === null ? "-" : String(s.slouch_events_7d), `slouch episodes${coverage(s)}`],
    [s.good_runs_7d === null ? "-" : String(s.good_runs_7d), `good runs (${PROGRESS.good_run_s}s+)${coverage(s)}`],
  ];
  $("week-stats").replaceChildren(
    ...cells.map(([value, label]) => {
      const d = document.createElement("div");
      const b = document.createElement("b");
      b.textContent = value;
      const span = document.createElement("span");
      span.textContent = label;
      d.append(b, span);
      return d;
    }),
  );
  const suggest = s.try_light && settings.coachingLevel !== "light";
  $("progress-suggest").textContent = suggest
    ? "Your latest silent test is as good as your coached time. You could try light coaching."
    : "";
  $("progress-light").hidden = !suggest;
}
/** Personal best days on the main dashboard (this device only, no sharing). */
function renderBestDays(s) {
  const box = $soft("best-days");
  const { top, ranked_days: n, today_rank: rank } = s.board;
  box.hidden = top.length === 0;
  if (!top.length) return;
  const items = top.map((r, i) => {
    const li = document.createElement("li");
    const date = new Date(`${r.date}T00:00:00Z`);
    const name = r.today
      ? "Today"
      : date.toLocaleDateString(undefined, { weekday: "short", month: "short", day: "numeric", timeZone: "UTC" });
    const b = document.createElement("b");
    b.textContent = `${r.pct}%`;
    li.append(`${i + 1}. ${name} `, b, ` good, ${minutesText(r.tracked_s)} tracked`);
    if (r.today) li.className = "me";
    return li;
  });
  $soft("best-days-list").replaceChildren(...items);
  $soft("best-days-note").textContent =
    rank === null
      ? `Your best days (20+ min tracked). Today ranks once it has 20 minutes.`
      : `Today ranks ${rank} of ${n} tracked days.`;
}
$("progress-light").addEventListener("click", () => {
  setCoachingLevel("light");
  renderProgress();
  $("progress-msg").textContent = "Switched to light coaching. Change it under Help & more.";
});

// -- import ------------------------------------------------------------------------------
let pendingImport = null;
$("progress-import").addEventListener("click", () => $("progress-import-file").click());
$("progress-import-file").addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  e.target.value = "";
  if (!file) return;
  if (file.size > IMPORT_MAX_BYTES) {
    $("progress-msg").textContent = "Not imported: the file is larger than 1 MB.";
    return;
  }
  let parsed;
  try {
    parsed = parseImport(await file.text());
  } catch {
    parsed = { ok: false, error: "The file could not be read." };
  }
  if (!parsed.ok) {
    $("progress-msg").textContent = `Not imported: ${parsed.error}`;
    return;
  }
  pendingImport = parsed;
  const p = previewImport(parsed);
  $("import-preview").textContent =
    p.days || p.tests
      ? `${p.days} day${p.days === 1 ? "" : "s"}${p.first ? ` (${p.first} to ${p.last})` : ""}, ${minutesText(p.good_s)} of good posture, ${p.tests} silent test${p.tests === 1 ? "" : "s"}.`
      : "The file has no days or tests.";
  $("import-dialog").showModal();
});
function finishImport(mode) {
  if (pendingImport) {
    applyImport(store, pendingImport, mode);
    save();
    $("progress-msg").textContent =
      mode === "replace" ? "Replaced your progress with the file." : "Merged the file into your progress.";
    renderProgress();
  }
  pendingImport = null;
  $("import-dialog").close();
}
$("import-merge").addEventListener("click", () => finishImport("merge"));
$("import-replace").addEventListener("click", () => finishImport("replace"));
$("import-cancel").addEventListener("click", () => {
  pendingImport = null;
  $("import-dialog").close();
});

// -- share my week (a link with one record, no server, opt-in) ------------------------------
// Nothing is sent. The record is packed into a link the person copies and sends themselves;
// a friend's link opens in the same dialog. See consent.js and share.js.
const appBaseUrl = () => {
  const u = new URL(location.href);
  u.search = "";
  u.hash = "";
  return u.href;
};
const thisWeek = () => summarizeWeeks(store, Date.now()).at(-1);
function currentPeerRecord(week, nickname) {
  if (!week) return null;
  return peerRecord({
    nickname,
    week: week.week,
    tracked_s: week.tracked_s,
    score: consistencyScore(week),
    breaks: week.breaks,
  });
}
function renderPeer() {
  if (document.activeElement !== $("peer-nickname")) $("peer-nickname").value = store.shareNickname || "";
  const rec = currentPeerRecord(thisWeek(), $("peer-nickname").value);
  $("peer-preview").textContent = rec ? JSON.stringify(rec, null, 2) : "Choose a nickname to see the record.";
}
function openPeer() {
  $("peer-link").hidden = $("peer-copy").hidden = true;
  $("peer-msg").textContent = "";
  renderPeer();
  $("peer-dialog").showModal();
}
$("share-week").addEventListener("click", () => {
  if (hasShareConsent(store)) {
    openPeer();
    return;
  }
  $("share-consent-nick").value = store.shareNickname || "";
  $("share-consent-check").checked = false;
  $("share-consent-agree").disabled = true;
  $("share-consent-msg").textContent = "";
  renderShareConsentPreview();
  $("share-consent-dialog").showModal();
});
$("peer-nickname").addEventListener("input", renderPeer);
$("peer-create").addEventListener("click", () => {
  const nick = cleanNickname($("peer-nickname").value);
  const rec = currentPeerRecord(thisWeek(), nick);
  if (!rec) {
    $("peer-msg").textContent = "Pick a nickname of 1 to 20 letters or digits. No @, no email.";
    return;
  }
  store.shareNickname = nick;
  save();
  $("peer-link").value = peerLink(rec, appBaseUrl());
  $("peer-link").hidden = $("peer-copy").hidden = false;
  $("peer-msg").textContent = "Link created. It holds only the record above. Copy it and send it yourself.";
});
$("peer-copy").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText($("peer-link").value);
    $("peer-msg").textContent = "Copied.";
  } catch {
    $("peer-link").select();
    $("peer-msg").textContent = "Copy was blocked. The link is selected: press Ctrl+C (or Cmd+C).";
  }
});
$("peer-view").addEventListener("click", () => {
  const rec = readPeerLink($("peer-paste").value);
  if (!rec) {
    $("peer-msg").textContent = "That is not a BackUp share link.";
    return;
  }
  $("peer-msg").textContent = "";
  showPeer(rec);
});
$("peer-close").addEventListener("click", () => $("peer-dialog").close());
$("peer-stop").addEventListener("click", () => {
  withdrawShareConsent(store);
  save();
  $("peer-dialog").close();
  $("progress-msg").textContent = "Sharing is off. Links you already sent still work for whoever has them.";
});
function renderShareConsentPreview() {
  const rec = currentPeerRecord(thisWeek(), $("share-consent-nick").value);
  $("share-consent-preview").textContent = rec ? JSON.stringify(rec, null, 2) : "Choose a nickname to see the record.";
}
$("share-consent-title").textContent = SHARE_CONSENT_TITLE;
$("share-consent-label").append(SHARE_CONSENT_CHECKBOX);
$("share-consent-list").replaceChildren(
  ...SHARE_CONSENT_POINTS.map((t) => {
    const li = document.createElement("li");
    li.textContent = t;
    return li;
  }),
);
$("share-consent-nick").addEventListener("input", renderShareConsentPreview);
$("share-consent-check").addEventListener("change", (e) => {
  $("share-consent-agree").disabled = !e.target.checked;
});
$("share-consent-agree").addEventListener("click", () => {
  const nick = cleanNickname($("share-consent-nick").value);
  if (!nick) {
    $("share-consent-msg").textContent = "Pick a nickname of 1 to 20 letters or digits. No @, no email.";
    return;
  }
  recordShareConsent(store);
  store.shareNickname = nick;
  save();
  $("share-consent-dialog").close();
  openPeer();
});
$("share-consent-cancel").addEventListener("click", () => $("share-consent-dialog").close());

function showPeer(rec) {
  $("peer-view-body").textContent =
    `${rec.nickname}, ${rec.week}: ${rec.tracked_minutes} tracked minutes, consistency score ${rec.consistency_score} out of 100, ${rec.breaks_taken} breaks taken.`;
  $("peer-view-dialog").showModal();
}
$("peer-view-close").addEventListener("click", () => $("peer-view-dialog").close());

$("progress-export").addEventListener("click", () => {
  const url = URL.createObjectURL(new Blob([exportJson(store)], { type: "application/json" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = "backup-progress.json";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

$("progress-delete").addEventListener("click", () => {
  if (
    !window.confirm(
      "Delete all BackUp data stored in this browser (settings, calibration, daily totals, tests, uploaded sounds and the offline copy)? This cannot be undone.",
    )
  )
    return;
  deleted = true; // stop any pending save from writing it back
  clearStoredData(localStorage);
  removeOfflineCopy().finally(() => location.reload());
});

// ---- Offline mode (opt-in; sw.js caches the files listed in precache.json) ----
const offlineSupported = "serviceWorker" in navigator && "caches" in window;
const offline = {
  downloading: false,
  error: false,
  percent: 0,
  totalBytes: 0,
  ready: false,
  registered: false,
  stale: false,
};

function renderOffline() {
  const state = offlineState({ supported: offlineSupported, ...offline });
  $("offline-status").textContent = offlineMessage(state, offline);
  $("offline-enable").hidden = !offlineSupported || state === "ready" || state === "downloading";
  // A button you press, never automatic: the copy is about 32 MB (#98 v3 removed it; restored on request).
  $("offline-enable").querySelector("span").textContent =
    state === "update-available" ? "Update offline copy" : state === "error" ? "Try again" : "Make available offline";
  $("offline-remove").hidden = !offlineSupported || !(offline.registered || offline.ready);
  $("offline-bar").hidden = state !== "downloading";
  $("offline-fill").style.width = `${offline.percent}%`;
}

async function removeOfflineCopy() {
  if (!offlineSupported) return;
  try {
    for (const reg of await navigator.serviceWorker.getRegistrations()) await reg.unregister();
    for (const k of await caches.keys()) if (k.startsWith("posture-guard-")) await caches.delete(k);
  } catch {
    /* nothing more to remove */
  }
}

async function refreshOffline() {
  if (!offlineSupported) return renderOffline();
  try {
    offline.registered = !!(await navigator.serviceWorker.getRegistration());
    // precache.json always comes from the network; skip it when offline.
    let pre = null;
    if (navigator.onLine) pre = await (await fetch("./precache.json", { cache: "no-cache" })).json();
    // Prefer the cache for the current release, else any complete older one (then stale).
    let cache = null,
      meta = null;
    const names = (await caches.keys()).filter((k) => k.startsWith(CACHE_PREFIX)); // caches.open would create an empty one
    if (pre) names.sort((a, b) => (b === cacheNameFor(pre.version)) - (a === cacheNameFor(pre.version)));
    for (const k of names) {
      const c = await caches.open(k);
      const res = await c.match(META_URL);
      const m = res ? await res.json().catch(() => null) : null;
      if (m) {
        cache = c;
        meta = m;
        break;
      }
    }
    if (!pre && meta)
      pre = {
        version: meta.version,
        totalBytes: offline.totalBytes,
        files: Object.keys(meta.hashes).map((url) => ({ url })),
      };
    if (!pre) throw new Error("offline and nothing cached");
    offline.totalBytes = pre.totalBytes;
    const base = new URL("./", location.href).href;
    const cached = cache
      ? (await cache.keys()).map((r) => (r.url.startsWith(base) ? r.url.slice(base.length) : r.url))
      : [];
    offline.ready = isOfflineReady(pre, cached, meta?.version);
    offline.stale = offline.registered && !offline.ready;
  } catch {
    // Offline and not cached, or precache.json unreadable: keep whatever we knew.
  }
  renderOffline();
}

if (offlineSupported) {
  navigator.serviceWorker.addEventListener("message", (e) => {
    const m = e.data || {};
    if (m.type === "offline-progress") {
      offline.downloading = true;
      offline.error = false;
      offline.totalBytes = m.totalBytes || offline.totalBytes;
      offline.percent = m.percent;
      renderOffline();
    } else if (m.type === "offline-ready") {
      offline.downloading = false;
      offline.percent = 100;
      refreshOffline();
    } else if (m.type === "offline-error") {
      offline.downloading = false;
      offline.error = true;
      renderOffline();
    }
  });
}

$("offline-enable").addEventListener("click", async () => {
  offline.downloading = true;
  offline.error = false;
  offline.percent = progressPercent(0, 1);
  renderOffline();
  try {
    const reg = await navigator.serviceWorker.register("./sw.js");
    await reg.update(); // picks up a newer sw.js; no-op when already current
    const sw = reg.installing || reg.waiting;
    if (sw) {
      await new Promise((resolve, reject) => {
        const check = () => {
          if (sw.state === "activated") resolve();
          else if (sw.state === "redundant") reject(new Error("install failed"));
          else return false;
          return true;
        };
        if (!check()) sw.addEventListener("statechange", check); // it may already be done
      });
    }
    offline.downloading = false;
  } catch {
    offline.downloading = false;
    offline.error = true;
  }
  await refreshOffline();
});

$("offline-remove").addEventListener("click", async () => {
  await removeOfflineCopy();
  offline.ready = false;
  offline.registered = false;
  offline.stale = false;
  offline.error = false;
  offline.downloading = false;
  await refreshOffline();
  $("offline-status").textContent = "Offline copy removed. The app needs a network connection again.";
});

renderOffline();
refreshOffline();

renderCalibration();
renderStats();
renderProgress();
if (location.hash.startsWith("#peer=")) {
  // A friend's share link: show it once, keep nothing.
  const rec = readPeerLink(location.hash);
  history.replaceState(null, "", location.pathname + location.search);
  if (rec) showPeer(rec);
}
window.addEventListener("pagehide", save);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) save();
});

// ?demo (the landing page "Try the demo" link): start the demo as if #start-demo
// were clicked, then drop the param so a reload does not restart it.
if (new URLSearchParams(location.search).has("demo")) {
  history.replaceState(null, "", location.pathname + location.hash);
  if (!$("start-demo").disabled) $("start-demo").click();
}
