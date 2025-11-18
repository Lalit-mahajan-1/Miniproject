function toNumber(val) {
  if (val === undefined || val === null) return undefined;
  const n = Number(String(val).trim());
  return Number.isFinite(n) ? n : undefined;
}

function findKeyCI(obj, candidates) {
  if (!obj) return undefined;
  const map = new Map(Object.keys(obj).map(k => [k.toLowerCase(), k]));
  for (const c of candidates) {
    const hit = map.get(String(c).toLowerCase());
    if (hit) return hit;
  }
  return undefined;
}

function riskCategory(risk) {
  const r = Number(risk || 0);
  if (r > 70) return "High";
  if (r > 40) return "Medium";
  return "Low";
}

function looksLikeEmail(v) {
  const s = String(v || "").trim();
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(s);
}

function detectEmail(obj) {
  if (!obj) return undefined;
  const candidates = ["email", "student_email", "mail", "username", "user", "student"];
  const k = findKeyCI(obj, candidates);
  if (k && looksLikeEmail(obj[k])) return String(obj[k]).trim();

  for (const v of Object.values(obj)) {
    if (looksLikeEmail(v)) return String(v).trim();
  }
  return undefined;
}

function pickName(obj) {
  const k = findKeyCI(obj, ["name", "student_name", "full_name"]);
  return k ? String(obj[k]).trim() : undefined;
}

function pickStudentId(obj) {
  const k = findKeyCI(obj, ["student_id", "rollno", "roll", "id"]);
  return k ? String(obj[k]).trim() : undefined;
}

function pick(obj, keys) {
  const k = findKeyCI(obj, keys);
  return k ? obj[k] : undefined;
}

module.exports = {
  toNumber,
  riskCategory,
  detectEmail,
  pickName,
  pickStudentId,
  pick
};