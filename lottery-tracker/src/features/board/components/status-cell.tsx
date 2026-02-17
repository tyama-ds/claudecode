import React from 'react';
import { TouchableOpacity, Text, StyleSheet, View } from 'react-native';
import { ApplicationStatus, APPLICATION_STATUS } from '../../../constants/status';

interface Props {
  status: ApplicationStatus;
  quantityWon: number;
  onPress: () => void;
  onLongPress: () => void;
}

export function StatusCell({ status, quantityWon, onPress, onLongPress }: Props) {
  const config = APPLICATION_STATUS[status];

  return (
    <TouchableOpacity
      style={[styles.cell, { backgroundColor: config.bgColor }]}
      onPress={onPress}
      onLongPress={onLongPress}
      activeOpacity={0.6}
    >
      {status === 'not_applied' ? (
        <Text style={styles.empty}>-</Text>
      ) : status === 'won' ? (
        <View style={styles.wonContainer}>
          <Text style={styles.wonEmoji}>{config.emoji}</Text>
          {quantityWon > 0 && (
            <Text style={[styles.wonQty, { color: config.color }]}>
              {quantityWon}
            </Text>
          )}
        </View>
      ) : (
        <Text style={[styles.emoji, { color: config.color }]}>
          {config.emoji}
        </Text>
      )}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  cell: {
    width: 52,
    height: 44,
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 8,
    marginHorizontal: 2,
  },
  empty: {
    fontSize: 16,
    color: '#BDBDBD',
  },
  emoji: {
    fontSize: 18,
  },
  wonContainer: {
    alignItems: 'center',
  },
  wonEmoji: {
    fontSize: 16,
  },
  wonQty: {
    fontSize: 11,
    fontWeight: '700',
    marginTop: -2,
  },
});
