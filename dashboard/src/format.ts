export type RiskBand = "unknown" | "low" | "mid" | "high";

export function riskBand(score: number | null | undefined, threshold: number): RiskBand {
  if (score == null || !Number.isFinite(score)) return "unknown";
  if (score < threshold) return "low";
  if (score < 0.75) return "mid";
  return "high";
}

export function formatSimTime(seconds: number): string {
  if (!Number.isFinite(seconds)) return "—";
  const whole = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(whole / 3600);
  const minutes = Math.floor((whole % 3600) / 60);
  const secs = whole % 60;
  if (hours > 0) return `${hours}h ${String(minutes).padStart(2, "0")}m`;
  return `${minutes}m ${String(secs).padStart(2, "0")}s`;
}

export function formatAmount(amount: number, currency: string): string {
  if (!Number.isFinite(amount)) return "—";
  if (currency === "BTC") {
    return `${amount.toLocaleString("en-US", { maximumFractionDigits: 6 })} BTC`;
  }
  if (currency === "USDT") {
    return `${amount.toLocaleString("en-US", { maximumFractionDigits: 2 })} USDT`;
  }
  try {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency,
      maximumFractionDigits: amount >= 100 ? 0 : 2,
    }).format(amount);
  } catch {
    return `${amount.toLocaleString("en-US", { maximumFractionDigits: 2 })} ${currency}`;
  }
}

export function formatPlainAmount(amount: number): string {
  if (!Number.isFinite(amount)) return "—";
  return amount.toLocaleString("en-US", { maximumFractionDigits: amount >= 100 ? 0 : 2 });
}

export function formatSpan(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds <= 0) return "0s";
  if (seconds < 90) return `${Math.round(seconds)}s`;
  return formatSimTime(seconds);
}

export function networkLabel(network: string): string {
  switch (network) {
    case "banking":
      return "Banking";
    case "crypto":
      return "Crypto";
    case "cross_border":
      return "Cross-border";
    default:
      return network || "Unknown";
  }
}

export function dispositionLabel(action: "open" | "approve" | "escalate"): string {
  switch (action) {
    case "approve":
      return "Approved";
    case "escalate":
      return "Escalated";
    default:
      return "Open";
  }
}
