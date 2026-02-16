import React, { useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  ActivityIndicator,
} from 'react-native';
import { useGroup } from '../../providers/group-provider';
import { useStoreStats } from './hooks/use-store-stats';
import { useMemberStats } from './hooks/use-member-stats';
import { useSimulation } from './hooks/use-simulation';
import { Colors } from '../../constants/colors';

export function StatsScreen() {
  const { currentGroup } = useGroup();
  const { data: storeWinRates = [], isLoading: storesLoading } = useStoreStats(
    currentGroup?.id
  );
  const { data: memberStats = [], isLoading: membersLoading } = useMemberStats(
    currentGroup?.id
  );

  // For simulation, collect store IDs with pending applications
  // In a real implementation, this would come from pending application data
  const [targetWins] = useState(1);
  const pendingStoreIds = storeWinRates
    .filter((s) => Number(s.total_applied) > 0)
    .map((s) => s.store_id);

  const { storeEstimates, simulation, targetProbability, additionalNeeded } =
    useSimulation(storeWinRates, pendingStoreIds, targetWins);

  const isLoading = storesLoading || membersLoading;

  // Overview aggregates
  const totalApplied = memberStats.reduce(
    (sum, m) => sum + Number(m.total_applied),
    0
  );
  const totalWon = memberStats.reduce(
    (sum, m) => sum + Number(m.total_won),
    0
  );
  const overallWinRate =
    totalApplied > 0 ? ((totalWon / totalApplied) * 100).toFixed(1) : '0.0';

  // Sort stores by win rate descending
  const sortedStores = [...storeEstimates].sort(
    (a, b) => b.bayesian.mean - a.bayesian.mean
  );

  // Max applied for bar scaling
  const maxMemberApplied = Math.max(
    ...memberStats.map((m) => Number(m.total_applied)),
    1
  );
  const maxStoreApplied = Math.max(
    ...storeWinRates.map((s) => Number(s.total_applied)),
    1
  );

  if (isLoading) {
    return (
      <View style={styles.centered}>
        <ActivityIndicator size="large" color={Colors.primary} />
      </View>
    );
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* ===== Overview Cards ===== */}
      <Text style={styles.sectionTitle}>概要</Text>
      <View style={styles.cardRow}>
        <View style={styles.card}>
          <Text style={styles.cardValue}>{totalApplied}</Text>
          <Text style={styles.cardLabel}>総応募数</Text>
        </View>
        <View style={styles.card}>
          <Text style={[styles.cardValue, { color: Colors.success }]}>
            {totalWon}
          </Text>
          <Text style={styles.cardLabel}>総当選数</Text>
        </View>
        <View style={styles.card}>
          <Text style={[styles.cardValue, { color: Colors.primary }]}>
            {overallWinRate}%
          </Text>
          <Text style={styles.cardLabel}>当選率</Text>
        </View>
      </View>

      {/* ===== Member Comparison ===== */}
      <Text style={styles.sectionTitle}>メンバー比較</Text>
      <View style={styles.section}>
        {memberStats.map((member) => {
          const winRate = Number(member.win_rate) * 100;
          const appliedRatio =
            Number(member.total_applied) / maxMemberApplied;
          return (
            <View key={member.user_id} style={styles.barRow}>
              <Text style={styles.barLabel} numberOfLines={1}>
                {member.short_name}
              </Text>
              <View style={styles.barTrack}>
                <View
                  style={[
                    styles.barFillApplied,
                    { flex: appliedRatio },
                  ]}
                />
                <View style={{ flex: 1 - appliedRatio }} />
              </View>
              <View style={styles.barTrack}>
                <View
                  style={[
                    styles.barFillWon,
                    { flex: Math.min(winRate / 100, 1) },
                  ]}
                />
                <View
                  style={{ flex: 1 - Math.min(winRate / 100, 1) }}
                />
              </View>
              <Text style={styles.barValue}>
                {winRate.toFixed(1)}%
              </Text>
            </View>
          );
        })}
        {memberStats.length > 0 && (
          <View style={styles.legendRow}>
            <View style={styles.legendItem}>
              <View
                style={[styles.legendDot, { backgroundColor: Colors.primaryLight }]}
              />
              <Text style={styles.legendText}>応募数</Text>
            </View>
            <View style={styles.legendItem}>
              <View
                style={[styles.legendDot, { backgroundColor: Colors.success }]}
              />
              <Text style={styles.legendText}>当選率</Text>
            </View>
          </View>
        )}
        {memberStats.length === 0 && (
          <Text style={styles.emptyText}>データがありません</Text>
        )}
      </View>

      {/* ===== Store Ranking ===== */}
      <Text style={styles.sectionTitle}>店舗ランキング</Text>
      <View style={styles.section}>
        {sortedStores.map((store, index) => {
          const winPct = (store.bayesian.mean * 100).toFixed(1);
          const lower = (store.bayesian.lower95 * 100).toFixed(1);
          const upper = (store.bayesian.upper95 * 100).toFixed(1);
          return (
            <View key={store.store_id} style={styles.storeRow}>
              <Text style={styles.storeRank}>{index + 1}</Text>
              <View style={styles.storeInfo}>
                <Text style={styles.storeName} numberOfLines={1}>
                  {store.store_name}
                </Text>
                <View style={styles.storeBarTrack}>
                  <View
                    style={[
                      styles.storeBarFill,
                      {
                        width: `${Math.min(store.bayesian.mean * 100, 100)}%`,
                      },
                    ]}
                  />
                </View>
                <Text style={styles.storeDetail}>
                  推定 {winPct}% ({lower}% - {upper}%) / {Number(store.total_won)}/{Number(store.total_applied)}件
                </Text>
              </View>
            </View>
          );
        })}
        {sortedStores.length === 0 && (
          <Text style={styles.emptyText}>データがありません</Text>
        )}
      </View>

      {/* ===== Probability Calculator ===== */}
      <Text style={styles.sectionTitle}>当選確率シミュレーション</Text>
      <View style={styles.section}>
        {simulation ? (
          <>
            <View style={styles.simRow}>
              <Text style={styles.simLabel}>応募中の店舗数</Text>
              <Text style={styles.simValue}>{pendingStoreIds.length}</Text>
            </View>
            <View style={styles.simRow}>
              <Text style={styles.simLabel}>期待当選数</Text>
              <Text style={[styles.simValue, { color: Colors.primary }]}>
                {simulation.expectedWins.toFixed(2)}
              </Text>
            </View>
            <View style={styles.simRow}>
              <Text style={styles.simLabel}>中央値</Text>
              <Text style={styles.simValue}>{simulation.median}</Text>
            </View>
            <View style={styles.simRow}>
              <Text style={styles.simLabel}>90%信頼区間</Text>
              <Text style={styles.simValue}>
                {simulation.percentile5} - {simulation.percentile95}
              </Text>
            </View>
            <View style={styles.simRow}>
              <Text style={styles.simLabel}>
                {targetWins}件以上当選する確率
              </Text>
              <Text style={[styles.simValue, { color: Colors.success }]}>
                {(targetProbability * 100).toFixed(1)}%
              </Text>
            </View>

            {/* Distribution bar chart */}
            <Text style={styles.subTitle}>当選数の分布</Text>
            <View style={styles.distributionChart}>
              {Array.from(simulation.distribution.entries())
                .sort(([a], [b]) => a - b)
                .map(([wins, count]) => {
                  const pct = count / pendingStoreIds.length;
                  const maxCount = Math.max(
                    ...Array.from(simulation.distribution.values())
                  );
                  const barHeight = (count / maxCount) * 80;
                  return (
                    <View key={wins} style={styles.distBarCol}>
                      <Text style={styles.distBarPct}>
                        {((count / 10000) * 100).toFixed(0)}%
                      </Text>
                      <View
                        style={[
                          styles.distBar,
                          {
                            height: barHeight,
                            backgroundColor:
                              wins >= targetWins
                                ? Colors.success
                                : Colors.primaryLight,
                          },
                        ]}
                      />
                      <Text style={styles.distBarLabel}>{wins}</Text>
                    </View>
                  );
                })}
            </View>
          </>
        ) : (
          <Text style={styles.emptyText}>
            応募中のデータがありません
          </Text>
        )}
      </View>

      {/* ===== Strategy Section ===== */}
      <Text style={styles.sectionTitle}>戦略アドバイス</Text>
      <View style={styles.section}>
        {additionalNeeded.length > 0 ? (
          <>
            <Text style={styles.strategyDesc}>
              {targetWins}件当選するために必要な追加応募数の目安
            </Text>
            {additionalNeeded.map((item) => (
              <View key={item.additionalStores} style={styles.strategyRow}>
                <Text style={styles.strategyLabel}>
                  +{item.additionalStores}店舗
                </Text>
                <View style={styles.strategyBarTrack}>
                  <View
                    style={[
                      styles.strategyBarFill,
                      {
                        width: `${Math.min(item.probability * 100, 100)}%`,
                        backgroundColor:
                          item.probability >= 0.8
                            ? Colors.success
                            : item.probability >= 0.5
                            ? Colors.warning
                            : Colors.primaryLight,
                      },
                    ]}
                  />
                </View>
                <Text style={styles.strategyValue}>
                  {(item.probability * 100).toFixed(0)}%
                </Text>
              </View>
            ))}
          </>
        ) : (
          <Text style={styles.emptyText}>
            データが不足しています
          </Text>
        )}
      </View>

      <View style={styles.footer} />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  content: {
    padding: 16,
  },
  centered: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: Colors.background,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: Colors.text,
    marginTop: 20,
    marginBottom: 12,
  },
  subTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: Colors.textSecondary,
    marginTop: 16,
    marginBottom: 8,
  },
  section: {
    backgroundColor: Colors.surface,
    borderRadius: 12,
    padding: 16,
    marginBottom: 4,
  },
  emptyText: {
    color: Colors.textSecondary,
    fontSize: 14,
    textAlign: 'center',
    paddingVertical: 12,
  },

  /* Overview Cards */
  cardRow: {
    flexDirection: 'row',
    gap: 10,
    marginBottom: 4,
  },
  card: {
    flex: 1,
    backgroundColor: Colors.surface,
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
  },
  cardValue: {
    fontSize: 24,
    fontWeight: '700',
    color: Colors.text,
  },
  cardLabel: {
    fontSize: 12,
    color: Colors.textSecondary,
    marginTop: 4,
  },

  /* Member Bar Chart */
  barRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
    gap: 8,
  },
  barLabel: {
    width: 48,
    fontSize: 13,
    fontWeight: '600',
    color: Colors.text,
  },
  barTrack: {
    flex: 1,
    height: 12,
    backgroundColor: Colors.divider,
    borderRadius: 6,
    flexDirection: 'row',
    overflow: 'hidden',
  },
  barFillApplied: {
    backgroundColor: Colors.primaryLight,
    borderRadius: 6,
  },
  barFillWon: {
    backgroundColor: Colors.success,
    borderRadius: 6,
  },
  barValue: {
    width: 50,
    fontSize: 13,
    fontWeight: '600',
    color: Colors.text,
    textAlign: 'right',
  },
  legendRow: {
    flexDirection: 'row',
    gap: 16,
    marginTop: 8,
    justifyContent: 'center',
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  legendDot: {
    width: 10,
    height: 10,
    borderRadius: 5,
  },
  legendText: {
    fontSize: 12,
    color: Colors.textSecondary,
  },

  /* Store Ranking */
  storeRow: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 14,
    gap: 10,
  },
  storeRank: {
    width: 24,
    fontSize: 16,
    fontWeight: '700',
    color: Colors.primary,
    textAlign: 'center',
    marginTop: 2,
  },
  storeInfo: {
    flex: 1,
  },
  storeName: {
    fontSize: 14,
    fontWeight: '600',
    color: Colors.text,
    marginBottom: 4,
  },
  storeBarTrack: {
    height: 8,
    backgroundColor: Colors.divider,
    borderRadius: 4,
    overflow: 'hidden',
    marginBottom: 4,
  },
  storeBarFill: {
    height: '100%',
    backgroundColor: Colors.primary,
    borderRadius: 4,
  },
  storeDetail: {
    fontSize: 11,
    color: Colors.textSecondary,
  },

  /* Simulation */
  simRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
    borderBottomWidth: 1,
    borderBottomColor: Colors.divider,
  },
  simLabel: {
    fontSize: 14,
    color: Colors.text,
    flex: 1,
  },
  simValue: {
    fontSize: 16,
    fontWeight: '700',
    color: Colors.text,
  },

  /* Distribution Chart */
  distributionChart: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    justifyContent: 'center',
    gap: 6,
    height: 120,
    paddingTop: 16,
  },
  distBarCol: {
    alignItems: 'center',
    minWidth: 32,
  },
  distBarPct: {
    fontSize: 10,
    color: Colors.textSecondary,
    marginBottom: 2,
  },
  distBar: {
    width: 24,
    borderRadius: 4,
    minHeight: 4,
  },
  distBarLabel: {
    fontSize: 12,
    color: Colors.text,
    fontWeight: '600',
    marginTop: 4,
  },

  /* Strategy */
  strategyDesc: {
    fontSize: 13,
    color: Colors.textSecondary,
    marginBottom: 12,
  },
  strategyRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
    gap: 8,
  },
  strategyLabel: {
    width: 64,
    fontSize: 13,
    fontWeight: '600',
    color: Colors.text,
  },
  strategyBarTrack: {
    flex: 1,
    height: 14,
    backgroundColor: Colors.divider,
    borderRadius: 7,
    overflow: 'hidden',
  },
  strategyBarFill: {
    height: '100%',
    borderRadius: 7,
  },
  strategyValue: {
    width: 40,
    fontSize: 13,
    fontWeight: '600',
    color: Colors.text,
    textAlign: 'right',
  },

  footer: {
    height: 40,
  },
});
