/**
 * Analytics utilities for lottery probability analysis
 * - Bayesian estimation using Beta distribution
 * - Monte Carlo simulation
 * - Expected value and confidence intervals
 */

// Beta distribution - used for Bayesian estimation of win probability
function betaPdf(x: number, alpha: number, beta: number): number {
  if (x < 0 || x > 1) return 0;
  const B = (gammaLn(alpha) + gammaLn(beta) - gammaLn(alpha + beta));
  return Math.exp((alpha - 1) * Math.log(x) + (beta - 1) * Math.log(1 - x) - B);
}

function gammaLn(z: number): number {
  // Stirling's approximation for log-gamma
  const c = [76.18009172947146, -86.50532032941677, 24.01409824083091,
    -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
  let x = z;
  let y = z;
  let tmp = x + 5.5;
  tmp -= (x + 0.5) * Math.log(tmp);
  let ser = 1.000000000190015;
  for (let j = 0; j < 6; j++) {
    ser += c[j] / ++y;
  }
  return -tmp + Math.log(2.5066282746310005 * ser / x);
}

/**
 * Bayesian estimation of win probability for a store
 * Uses Beta(alpha, beta) as posterior with uniform prior Beta(1, 1)
 */
export function bayesianWinProbability(wins: number, losses: number): {
  mean: number;
  lower95: number;
  upper95: number;
} {
  // Prior: Beta(1, 1) = uniform
  const alpha = 1 + wins;
  const beta = 1 + losses;

  const mean = alpha / (alpha + beta);

  // Approximate 95% credible interval using normal approximation
  const variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1));
  const std = Math.sqrt(variance);
  const lower95 = Math.max(0, mean - 1.96 * std);
  const upper95 = Math.min(1, mean + 1.96 * std);

  return { mean, lower95, upper95 };
}

/**
 * Monte Carlo simulation for expected total wins
 * Given per-store win probabilities for current pending applications
 */
export function monteCarloSimulation(
  storeProbabilities: number[],
  numSimulations: number = 10000
): {
  expectedWins: number;
  percentile5: number;
  percentile25: number;
  median: number;
  percentile75: number;
  percentile95: number;
  distribution: Map<number, number>; // wins -> count
} {
  const results: number[] = [];

  for (let sim = 0; sim < numSimulations; sim++) {
    let wins = 0;
    for (const prob of storeProbabilities) {
      if (Math.random() < prob) wins++;
    }
    results.push(wins);
  }

  results.sort((a, b) => a - b);

  const distribution = new Map<number, number>();
  for (const r of results) {
    distribution.set(r, (distribution.get(r) ?? 0) + 1);
  }

  return {
    expectedWins: results.reduce((a, b) => a + b, 0) / numSimulations,
    percentile5: results[Math.floor(numSimulations * 0.05)],
    percentile25: results[Math.floor(numSimulations * 0.25)],
    median: results[Math.floor(numSimulations * 0.5)],
    percentile75: results[Math.floor(numSimulations * 0.75)],
    percentile95: results[Math.floor(numSimulations * 0.95)],
    distribution,
  };
}

/**
 * Calculate probability of reaching target wins given store probabilities
 */
export function targetAchievementProbability(
  storeProbabilities: number[],
  targetWins: number,
  numSimulations: number = 10000
): number {
  let achieved = 0;

  for (let sim = 0; sim < numSimulations; sim++) {
    let wins = 0;
    for (const prob of storeProbabilities) {
      if (Math.random() < prob) wins++;
    }
    if (wins >= targetWins) achieved++;
  }

  return achieved / numSimulations;
}

/**
 * Simulate how many additional stores needed to reach target
 * Returns array of { additionalStores, probability }
 */
export function additionalStoresNeeded(
  currentProbabilities: number[],
  averageWinRate: number,
  targetWins: number,
  maxAdditional: number = 20,
  numSimulations: number = 5000
): { additionalStores: number; probability: number }[] {
  const results: { additionalStores: number; probability: number }[] = [];

  for (let additional = 0; additional <= maxAdditional; additional++) {
    const allProbs = [
      ...currentProbabilities,
      ...Array(additional).fill(averageWinRate),
    ];
    const prob = targetAchievementProbability(allProbs, targetWins, numSimulations);
    results.push({ additionalStores: additional, probability: prob });

    // Stop if we've reached near certainty
    if (prob > 0.99) break;
  }

  return results;
}

/**
 * Calculate marginal value of one additional application
 */
export function marginalValue(
  currentProbabilities: number[],
  additionalWinRate: number,
  numSimulations: number = 10000
): {
  currentExpected: number;
  newExpected: number;
  marginalGain: number;
} {
  const current = monteCarloSimulation(currentProbabilities, numSimulations);
  const withAdditional = monteCarloSimulation(
    [...currentProbabilities, additionalWinRate],
    numSimulations
  );

  return {
    currentExpected: current.expectedWins,
    newExpected: withAdditional.expectedWins,
    marginalGain: withAdditional.expectedWins - current.expectedWins,
  };
}
