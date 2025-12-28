import { neon } from "@neondatabase/serverless";

type LogEntry = {
  fileName: string;
  label: string;
  confidence: number;
  createdAt?: Date;
};

export async function logPrediction(entry: LogEntry): Promise<void> {
  const url = process.env.NEON_DATABASE_URL;
  if (!url) return;

  try {
    const sql = neon(url);
    const created = entry.createdAt ?? new Date();
    await sql`
      CREATE TABLE IF NOT EXISTS inference_logs (
        id serial PRIMARY KEY,
        file_name text,
        label text,
        confidence double precision,
        created_at timestamptz DEFAULT now()
      );
    `;
    await sql`
      INSERT INTO inference_logs (file_name, label, confidence, created_at)
      VALUES (${entry.fileName}, ${entry.label}, ${entry.confidence}, ${created.toISOString()});
    `;
  } catch (err) {
    console.error("Neon logging failed:", err);
  }
}
