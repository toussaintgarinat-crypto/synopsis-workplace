"""Front : section « Modèles par tâche », clés i18n, champs envoyés aux routes."""
import re

import main
from lib.i18n import TEXTS


# ---------- front ----------

NOUVELLES_CLES = ["mt_titre", "mt_intro", "mt_defaut", "mt_identique", "mt_actualiser", "mt_erreur",
                  "mt_resume", "mt_resume_aide", "mt_fusion", "mt_fusion_aide", "mt_qa", "mt_qa_aide",
                  "mt_transcription_video", "mt_transcription_video_aide",
                  "mt_transcription_audio", "mt_transcription_audio_aide", "mt_image", "mt_image_aide",
                  "transcription_llm", "transcription_llm_light", "modeles_utilises", "mt_l_resume", "mt_l_fusion",
                  "transcription_source"]


def test_i18n_new_keys_both_languages():
    for k in NOUVELLES_CLES:
        assert TEXTS["fr"].get(k) and TEXTS["en"].get(k), k


def test_front_section_and_fallback_keys():
    html = main.index()
    assert "Modèles par tâche" in html
    for t in ("resume", "fusion", "qa", "transcription_video", "transcription_audio", "image"):
        assert f'id="mt-{t}"' in html
    assert 'data-cap="transcription_video"' in html and 'id="transcription-llm"' in html
    script = re.search(r"<script>(.*?)</script>", html, re.S).group(1)
    secours = re.search(r"var FR_SECOURS = \{(.*?)\};", script, re.S).group(1)
    for k in NOUVELLES_CLES:
        assert re.search(r"[\s,{]" + k + r":", secours), k
    for champ in ("modeles: getModelesTaches()", "transcription_llm: getTranscriptionLlm()"):
        assert champ in script
    assert "fd.append('modeles'" in script
    assert "synopsis_modeles_" in script


def test_front_script_has_no_backslash_in_new_code():
    script = re.search(r"<script>(.*?)</script>", main.index(), re.S).group(1)
    debut = script.index("// ---- Modèles par tâche ----")
    fin = script.index("// ---- fin Modèles par tâche ----")
    assert "\\" not in script[debut:fin]


# ---------- M6 : une liste en échec ne masque pas les autres ----------

import json
import shutil
import subprocess

import pytest


def _section():
    script = re.search(r"<script>(.*?)</script>", main.index(), re.S).group(1)
    return script[script.index("// ---- Modèles par tâche ----"):script.index("// ---- fin Modèles par tâche ----")]


def test_refresh_uses_allsettled_and_new_keys():
    sec = _section()
    assert "Promise.allSettled" in sec and "Promise.all(" not in sec
    for k in ("mt_listes_chargees", "mt_echec"):
        assert TEXTS["fr"].get(k) and TEXTS["en"].get(k), k
        script = re.search(r"<script>(.*?)</script>", main.index(), re.S).group(1)
        assert re.search(r"[\s,{]" + k + r":", re.search(r"var FR_SECOURS = \{(.*?)\};", script, re.S).group(1)), k
    assert "innerHTML" not in sec[sec.index("async function rafraichirModelesTaches"):]


HARNAIS = """
var els = {};
function el(id) {
  if (!els[id]) els[id] = {id: id, textContent: '', value: '', disabled: false, open: false, options: [],
    addEventListener: function() {}, setAttribute: function() {}, appendChild: function(o) { this.options.push(o); }};
  return els[id];
}
function $(s) { return el(s.replace('#', '')); }
var document = {createElement: function() { return {setAttribute: function() {}}; }};
var localStorage = {getItem: function() { return null; }, setItem: function() {}};
var FALLBACK_MODELS = {openrouter: ['a/a']};
function t(k) { return {mt_listes_chargees: 'listes chargées', mt_echec: 'échec', mt_erreur: 'réseau KO'}[k] || k; }
function getProvider() { return 'openrouter'; }
function getApiKey() { return 'sk-x'; }
var fetch = async function(url, opts) {
  var tache = JSON.parse(opts.body).tache;
  if (tache === 'image') return {ok: false, status: 429, json: async function() { return {detail: 'Trop de <b>requêtes</b>'}; }};
  if (tache === 'transcription_video') throw new TypeError('Failed to fetch');
  return {ok: true, status: 200, json: async function() { return {models: ['m/' + tache]}; }};
};
"""


@pytest.mark.skipif(shutil.which("node") is None, reason="node absent")
def test_refresh_partial_failure_runtime(tmp_path):
    f = tmp_path / "h.js"
    f.write_text(HARNAIS + _section() + """
rafraichirModelesTaches().then(function() {
  console.log(JSON.stringify({statut: $('#mt-statut').textContent, resume: $('#mt-resume').options.length,
                              btn: $('#btn-mt-refresh').disabled, charge: !!mtCharge.openrouter}));
});
""", encoding="utf-8")
    out = json.loads(subprocess.run(["node", str(f)], capture_output=True, text=True, check=True).stdout)
    assert out["statut"].startswith("4/6 listes chargées")
    assert "image : Trop de <b>requêtes</b>" in out["statut"]  # texte brut via textContent
    assert "transcription_video : réseau KO" in out["statut"] and "échec" in out["statut"]
    assert out["resume"] == 2 and out["btn"] is False and out["charge"] is False
