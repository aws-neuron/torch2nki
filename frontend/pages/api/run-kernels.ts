import { NextApiRequest, NextApiResponse } from "next";

interface TestResult {
  timestamp: string;
  kernelId: string;
  latency: number;
  accuracy: number;
}

export default function handler(req: NextApiRequest, res: NextApiResponse<{ result?: TestResult; error?: string }>) {
  const { kernelId } = req.query;

  if (!kernelId || typeof kernelId !== "string") {
    return res.status(400).json({ error: "Kernel ID is required" });
  }

  const result: TestResult = {
    timestamp: new Date().toISOString(),
    kernelId,
    latency: Math.random() * 100,
    accuracy: Math.random(),
  };

  res.status(200).json({ result });
}