import React from 'react';
import { View, Text, TouchableOpacity, StyleSheet, Linking } from 'react-native';
import { StatusCell } from './status-cell';
import { BoardOffering } from '../../../types';
import { Profile } from '../../../types';
import { ApplicationStatus } from '../../../constants/status';
import { Colors } from '../../../constants/colors';
import { format, isPast } from 'date-fns';
import { ja } from 'date-fns/locale';

interface Props {
  offering: BoardOffering;
  members: Pick<Profile, 'id' | 'display_name' | 'short_name'>[];
  onStatusPress: (offeringId: string, userId: string, currentStatus: ApplicationStatus) => void;
  onStatusLongPress: (offeringId: string, userId: string, currentStatus: ApplicationStatus, quantityWon: number) => void;
}

export function StoreRow({ offering, members, onStatusPress, onStatusLongPress }: Props) {
  const store = offering.store;

  function getApplicationForMember(userId: string) {
    return offering.applications.find((a) => a.user_id === userId);
  }

  function handleStoreTap() {
    const url = offering.lottery_url || store.url;
    if (url) {
      Linking.openURL(url);
    }
  }

  const deadlineStr = offering.deadline_end
    ? format(new Date(offering.deadline_end), 'M/d', { locale: ja })
    : null;

  const isExpired = offering.deadline_end
    ? isPast(new Date(offering.deadline_end))
    : false;

  return (
    <View style={[styles.row, offering.is_ended && styles.rowEnded]}>
      <TouchableOpacity style={styles.storeInfo} onPress={handleStoreTap}>
        <Text style={[styles.storeName, offering.is_ended && styles.storeNameEnded]} numberOfLines={1}>
          {store.name}
        </Text>
        {offering.branch_info && (
          <Text style={styles.branchInfo}>{offering.branch_info}</Text>
        )}
        <View style={styles.storeMetaRow}>
          {deadlineStr && (
            <Text style={[styles.deadline, isExpired && styles.deadlineExpired]}>
              ~{deadlineStr}
            </Text>
          )}
          {offering.is_ended && (
            <View style={styles.endedBadge}>
              <Text style={styles.endedText}>終了</Text>
            </View>
          )}
          {(offering.lottery_url || store.url) && (
            <Text style={styles.linkIcon}>🔗</Text>
          )}
        </View>
      </TouchableOpacity>

      <View style={styles.cells}>
        {members.map((member) => {
          const app = getApplicationForMember(member.id);
          const status: ApplicationStatus = app?.status ?? 'not_applied';
          const qty = app?.quantity_won ?? 0;

          return (
            <StatusCell
              key={member.id}
              status={status}
              quantityWon={qty}
              onPress={() => onStatusPress(offering.id, member.id, status)}
              onLongPress={() => onStatusLongPress(offering.id, member.id, status, qty)}
            />
          );
        })}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 6,
    paddingHorizontal: 8,
    borderBottomWidth: 1,
    borderBottomColor: Colors.divider,
    backgroundColor: Colors.surface,
  },
  rowEnded: {
    opacity: 0.6,
  },
  storeInfo: {
    width: 100,
    paddingRight: 4,
  },
  storeName: {
    fontSize: 13,
    fontWeight: '600',
    color: Colors.text,
  },
  storeNameEnded: {
    color: Colors.textSecondary,
  },
  branchInfo: {
    fontSize: 10,
    color: Colors.textSecondary,
  },
  storeMetaRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    marginTop: 2,
  },
  deadline: {
    fontSize: 10,
    color: Colors.warning,
    fontWeight: '500',
  },
  deadlineExpired: {
    color: Colors.error,
  },
  endedBadge: {
    backgroundColor: Colors.textLight,
    borderRadius: 4,
    paddingHorizontal: 4,
    paddingVertical: 1,
  },
  endedText: {
    fontSize: 9,
    color: '#FFFFFF',
    fontWeight: '600',
  },
  linkIcon: {
    fontSize: 10,
  },
  cells: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'flex-start',
    gap: 2,
  },
});
