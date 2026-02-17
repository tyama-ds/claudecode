import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { ProductSummary } from '../../../types';
import { Colors } from '../../../constants/colors';

interface Props {
  summary: ProductSummary[];
}

export function SummaryHeader({ summary }: Props) {
  if (summary.length === 0) return null;

  return (
    <View style={styles.container}>
      <View style={styles.row}>
        <Text style={styles.label}>確保:</Text>
        {summary.map((s) => (
          <View key={s.user_id} style={styles.chip}>
            <Text style={styles.chipName}>{s.short_name}</Text>
            <Text style={styles.chipValue}>{s.total_quantity_won}個</Text>
          </View>
        ))}
      </View>
      <View style={styles.row}>
        <Text style={styles.labelSmall}>応募中:</Text>
        {summary.map((s) => (
          <Text key={s.user_id} style={styles.pendingText}>
            {s.short_name}{s.total_pending}件
          </Text>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: Colors.surface,
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: Colors.divider,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 2,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    color: Colors.text,
    width: 40,
  },
  labelSmall: {
    fontSize: 12,
    color: Colors.textSecondary,
    width: 40,
  },
  chip: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#E8F5E9',
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 12,
    gap: 2,
  },
  chipName: {
    fontSize: 13,
    fontWeight: '600',
    color: Colors.success,
  },
  chipValue: {
    fontSize: 13,
    fontWeight: '700',
    color: Colors.success,
  },
  pendingText: {
    fontSize: 12,
    color: Colors.textSecondary,
  },
});
