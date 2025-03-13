import { NextApiRequest, NextApiResponse } from "next";

interface Kernel {
  id: string;
  name: string;
}

export default function handler(req: NextApiRequest, res: NextApiResponse<{ kernels: Kernel[] }>) {
  const kernels: Kernel[] = [
    { id: "1", name: "NKI-Kernel-001" },
    { id: "2", name: "NKI-Kernel-002" },
    { id: "3", name: "NKI-Kernel-003" },
  ];

  res.status(200).json({ kernels });
}