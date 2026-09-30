import { ai } from "hatchable";

export const access = "public";
export const methods = ["POST"];

function scoreTransaction(input) {
  const reasons = [];
  let score = 8;
  const amount = Number(input.amount || 0);
  if (amount > 1000) { score += 30; reasons.push("High transaction amount"); }
  else if (amount > 500) { score += 16; reasons.push("Above-normal transaction amount"); }
  if ((input.device || "").toLowerCase().includes("new")) { score += 25; reasons.push("New device fingerprint"); }
  if ((input.country || "").toLowerCase() !== "us") { score += 15; reasons.push("Cross-border location"); }
  if ((input.channel || "").toLowerCase().includes("online")) { score += 8; reasons.push("Card-not-present channel"); }
  const text = (input.description || "").toLowerCase();
  if (/urgent|gift card|crypto|wire|password|verify account|otp/.test(text)) { score += 22; reasons.push("Suspicious NLP keywords"); }
  if (text.length > 120) { score += 5; reasons.push("Unusually detailed transaction narrative"); }
  score = Math.min(99.9, Math.max(0.1, score));
  const level = score >= 80 ? "Critical" : score >= 55 ? "High" : score >= 30 ? "Medium" : "Low";
  return { score, level, reasons };
}

export default async function(req, res) {
  const input = req.body || {};
  if (!input.merchant || !Number.isFinite(Number(input.amount))) {
    return res.status(400).json({ error: "merchant and numeric amount are required" });
  }
  const model = scoreTransaction(input);
  let explanation = "Behavioral and NLP feature scoring identified the signals shown below.";
  try {
    const r = await ai.generateText({
      model: "haiku",
      purpose: "fraud-explanation",
      system: "You are a fraud analyst. Explain transaction risk concisely. Do not invent facts. Return 2 short sentences.",
      prompt: JSON.stringify({ transaction: input, score: model.score, reasons: model.reasons })
    });
    if (r.finishReason !== "length" && r.text) explanation = r.text;
  } catch (e) {
    console.log("AI explanation unavailable", e?.message || e);
  }
  res.json({ ...model, explanation });
}