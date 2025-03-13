import { NextApiRequest, NextApiResponse } from "next";
import { exec } from "child_process";
import fs from "fs";

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const { operator } = req.body;
  if (!operator) {
    return res.status(400).json({ error: "Operator is required" });
  }

  console.log(`Executing benchmark for operator: ${operator}`);

  // Run the Python script
  exec(`python3 ../evaluation/benchmarking/run_benchmarking.py --operator ${operator}`, (error, stdout, stderr) => {
    if (error) {
      console.error(`Error executing script: ${stderr || error.message}`);
      return res.status(500).json({ error: stderr || error.message });
    }

    console.log(`Benchmark output: ${stdout}`);

    // Read results from JSON file
    fs.readFile("results.json", "utf8", (err, data) => {
      if (err) {
        return res.status(500).json({ error: "Failed to read results.json" });
      }
      
      // Parse and return JSON results
      try {
        const results = JSON.parse(data);
        return res.status(200).json(results);
      } catch (parseError) {
        return res.status(500).json({ error: "Error parsing results.json" });
      }
    });
  });
}