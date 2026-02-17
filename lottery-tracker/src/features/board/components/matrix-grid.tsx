import React from 'react';
import { FlatList, View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { StoreRow } from './store-row';
import { BoardOffering, Profile } from '../../../types';
import { ApplicationStatus } from '../../../constants/status';
import { Colors } from '../../../constants/colors';

interface Props {
  offerings: BoardOffering[];
  members: Pick<Profile, 'id' | 'display_name' | 'short_name'>[];
  onStatusPress: (offeringId: string, userId: string, currentStatus: ApplicationStatus) => void;
  onStatusLongPress: (offeringId: string, userId: string, currentStatus: ApplicationStatus, quantityWon: number) => void;
  onAddStore: () => void;
}

export function MatrixGrid({
  offerings,
  members,
  onStatusPress,
  onStatusLongPress,
  onAddStore,
}: Props) {
  return (
    <FlatList
      data={offerings}
      keyExtractor={(item) => item.id}
      ListHeaderComponent={
        <View style={styles.header}>
          <View style={styles.storeHeader}>
            <Text style={styles.headerLabel}>店舗</Text>
          </View>
          <View style={styles.memberHeaders}>
            {members.map((m) => (
              <View key={m.id} style={styles.memberHeader}>
                <Text style={styles.memberName}>{m.short_name}</Text>
              </View>
            ))}
          </View>
        </View>
      }
      renderItem={({ item }) => (
        <StoreRow
          offering={item}
          members={members}
          onStatusPress={onStatusPress}
          onStatusLongPress={onStatusLongPress}
        />
      )}
      ListFooterComponent={
        <TouchableOpacity style={styles.addButton} onPress={onAddStore}>
          <Text style={styles.addText}>+ ストアを追加</Text>
        </TouchableOpacity>
      }
      ListEmptyComponent={
        <View style={styles.empty}>
          <Text style={styles.emptyText}>まだ店舗が追加されていません</Text>
          <TouchableOpacity style={styles.addFirstButton} onPress={onAddStore}>
            <Text style={styles.addFirstText}>最初の店舗を追加</Text>
          </TouchableOpacity>
        </View>
      }
      stickyHeaderIndices={[0]}
      contentContainerStyle={styles.list}
    />
  );
}

const styles = StyleSheet.create({
  list: {
    flexGrow: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 8,
    backgroundColor: Colors.background,
    borderBottomWidth: 2,
    borderBottomColor: Colors.border,
  },
  storeHeader: {
    width: 100,
    paddingRight: 4,
  },
  headerLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: Colors.textSecondary,
  },
  memberHeaders: {
    flex: 1,
    flexDirection: 'row',
    gap: 2,
  },
  memberHeader: {
    width: 52,
    alignItems: 'center',
    marginHorizontal: 2,
  },
  memberName: {
    fontSize: 14,
    fontWeight: '700',
    color: Colors.primary,
  },
  addButton: {
    padding: 16,
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: Colors.divider,
  },
  addText: {
    fontSize: 14,
    color: Colors.primary,
    fontWeight: '600',
  },
  empty: {
    padding: 40,
    alignItems: 'center',
  },
  emptyText: {
    fontSize: 14,
    color: Colors.textSecondary,
    marginBottom: 16,
  },
  addFirstButton: {
    backgroundColor: Colors.primary,
    paddingHorizontal: 24,
    paddingVertical: 12,
    borderRadius: 12,
  },
  addFirstText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
  },
});
