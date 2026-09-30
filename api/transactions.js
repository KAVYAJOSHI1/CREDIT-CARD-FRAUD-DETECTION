import { db } from "hatchable";

export const access = "public";
export const methods = ["GET"];

export default async function(req, res) {
  const limit = Math.min(Number(req.query?.limit || 50), 100);
  const { rows } = await db.query(
    "SELECT id, merchant, amount, currency, country, device, channel, description, risk_score, risk_level, status, reasons, created_at FROM transactions ORDER BY created_at DESC LIMIT $1",
    [limit]
  );
  res.json(rows);
}