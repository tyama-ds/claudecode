import { useMemo } from 'react';
import {
  bayesianWinProbability,
  monteCarloSimulation,
  targetAchievementProbability,
  additionalStoresNeeded,
} from '../../../utils/analytics';
import { StoreWinRate } from '../../../types';

export function useSimulation(
  storeWinRates: StoreWinRate[],
  pendingStoreIds: string[],
  targetWins: number = 1
) {
  return useMemo(() => {
    if (storeWinRates.length === 0) {
      return {
        storeEstimates: [],
        simulation: null,
        targetProbability: 0,
        additionalNeeded: [],
      };
    }

    // Bayesian estimates per store
    const storeEstimates = storeWinRates.map((s) => ({
      ...s,
      bayesian: bayesianWinProbability(
        Number(s.total_won),
        Number(s.total_applied) - Number(s.total_won)
      ),
    }));

    // Get probabilities for pending stores
    const pendingProbabilities = pendingStoreIds
      .map((storeId) => {
        const estimate = storeEstimates.find((s) => s.store_id === storeId);
        return estimate?.bayesian.mean ?? 0.15; // default 15% if no data
      })
      .filter((p) => p > 0);

    // Monte Carlo simulation
    const simulation = pendingProbabilities.length > 0
      ? monteCarloSimulation(pendingProbabilities)
      : null;

    // Target achievement probability
    const targetProbability = pendingProbabilities.length > 0
      ? targetAchievementProbability(pendingProbabilities, targetWins)
      : 0;

    // Average win rate for additional store estimate
    const avgWinRate = storeEstimates.length > 0
      ? storeEstimates.reduce((sum, s) => sum + s.bayesian.mean, 0) / storeEstimates.length
      : 0.15;

    // Additional stores needed
    const additionalNeeded = additionalStoresNeeded(
      pendingProbabilities,
      avgWinRate,
      targetWins
    );

    return {
      storeEstimates,
      simulation,
      targetProbability,
      additionalNeeded,
    };
  }, [storeWinRates, pendingStoreIds, targetWins]);
}
