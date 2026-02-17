import React, { useState } from 'react';
import {
  View,
  Text,
  FlatList,
  TouchableOpacity,
  StyleSheet,
  Alert,
  TextInput,
  Modal,
  ScrollView,
} from 'react-native';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { supabase } from '../../lib/supabase';
import { useGroup } from '../../providers/group-provider';
import { Store } from '../../types';
import { Colors } from '../../constants/colors';
import { COMMON_STORES, STORE_CATEGORIES, StorePreset } from '../../constants/stores';

export default function StoresTab() {
  const { currentGroup } = useGroup();
  const queryClient = useQueryClient();
  const [showAdd, setShowAdd] = useState(false);
  const [showPresets, setShowPresets] = useState(false);
  const [newName, setNewName] = useState('');
  const [newUrl, setNewUrl] = useState('');
  const [newCategory, setNewCategory] = useState<string>('other');

  const { data: stores = [] } = useQuery({
    queryKey: ['stores', currentGroup?.id],
    queryFn: async () => {
      const { data, error } = await supabase
        .from('stores')
        .select('*')
        .eq('group_id', currentGroup!.id)
        .eq('is_archived', false)
        .order('sort_order', { ascending: true });
      if (error) throw error;
      return data as Store[];
    },
    enabled: !!currentGroup?.id,
  });

  async function addStore(name: string, url?: string | null, category?: string) {
    if (!currentGroup) return;
    const { error } = await supabase.from('stores').insert({
      group_id: currentGroup.id,
      name: name.trim(),
      url: url || null,
      category: category || 'other',
    });
    if (error) {
      if (error.code === '23505') {
        Alert.alert('エラー', 'この店舗名は既に登録されています');
      } else {
        Alert.alert('エラー', error.message);
      }
      return;
    }
    queryClient.invalidateQueries({ queryKey: ['stores', currentGroup.id] });
    setShowAdd(false);
    setNewName('');
    setNewUrl('');
  }

  async function addFromPreset(preset: StorePreset) {
    await addStore(preset.name, preset.url, preset.category);
  }

  function renderStore({ item }: { item: Store }) {
    const catInfo = STORE_CATEGORIES[item.category as keyof typeof STORE_CATEGORIES] || STORE_CATEGORIES.other;
    return (
      <View style={styles.storeCard}>
        <View style={styles.storeMain}>
          <Text style={styles.storeName}>{item.name}</Text>
          {item.url && (
            <Text style={styles.storeUrl} numberOfLines={1}>
              {item.url}
            </Text>
          )}
        </View>
        <View style={[styles.categoryBadge, { backgroundColor: catInfo.color + '20' }]}>
          <Text style={[styles.categoryText, { color: catInfo.color }]}>
            {catInfo.label}
          </Text>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <FlatList
        data={stores}
        keyExtractor={(item) => item.id}
        renderItem={renderStore}
        contentContainerStyle={styles.list}
        ListEmptyComponent={
          <View style={styles.empty}>
            <Text style={styles.emptyText}>店舗が登録されていません</Text>
          </View>
        }
      />

      <View style={styles.bottomButtons}>
        <TouchableOpacity
          style={styles.addButton}
          onPress={() => setShowAdd(true)}
        >
          <Text style={styles.addButtonText}>+ 店舗を追加</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.presetButton}
          onPress={() => setShowPresets(true)}
        >
          <Text style={styles.presetButtonText}>プリセットから追加</Text>
        </TouchableOpacity>
      </View>

      {/* Add Store Modal */}
      <Modal visible={showAdd} transparent animationType="fade">
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>店舗を追加</Text>
            <TextInput
              style={styles.input}
              placeholder="店舗名"
              value={newName}
              onChangeText={setNewName}
              autoFocus
            />
            <TextInput
              style={styles.input}
              placeholder="URL（任意）"
              value={newUrl}
              onChangeText={setNewUrl}
              autoCapitalize="none"
              keyboardType="url"
            />
            <View style={styles.modalButtons}>
              <TouchableOpacity onPress={() => setShowAdd(false)}>
                <Text style={styles.cancelText}>キャンセル</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.submitButton}
                onPress={() => addStore(newName, newUrl, newCategory)}
              >
                <Text style={styles.submitText}>追加</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      {/* Presets Modal */}
      <Modal visible={showPresets} transparent animationType="slide">
        <View style={styles.modalOverlay}>
          <View style={[styles.modalContent, { maxHeight: '70%' }]}>
            <Text style={styles.modalTitle}>プリセットから選択</Text>
            <ScrollView>
              {COMMON_STORES.map((preset) => {
                const exists = stores.some((s) => s.name === preset.name);
                return (
                  <TouchableOpacity
                    key={preset.name}
                    style={[styles.presetItem, exists && styles.presetItemDisabled]}
                    onPress={() => !exists && addFromPreset(preset)}
                    disabled={exists}
                  >
                    <Text style={[styles.presetName, exists && styles.presetNameDisabled]}>
                      {preset.name}
                    </Text>
                    {exists && <Text style={styles.addedLabel}>追加済</Text>}
                  </TouchableOpacity>
                );
              })}
            </ScrollView>
            <TouchableOpacity
              style={styles.closeButton}
              onPress={() => setShowPresets(false)}
            >
              <Text style={styles.closeText}>閉じる</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: Colors.background },
  list: { padding: 16, paddingBottom: 100 },
  storeCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.surface,
    borderRadius: 12,
    padding: 16,
    marginBottom: 8,
    borderWidth: 1,
    borderColor: Colors.divider,
  },
  storeMain: { flex: 1 },
  storeName: { fontSize: 16, fontWeight: '600', color: Colors.text },
  storeUrl: { fontSize: 12, color: Colors.textSecondary, marginTop: 2 },
  categoryBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 8,
  },
  categoryText: { fontSize: 10, fontWeight: '600' },
  empty: { padding: 40, alignItems: 'center' },
  emptyText: { color: Colors.textSecondary, fontSize: 14 },
  bottomButtons: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    gap: 8,
    padding: 16,
    backgroundColor: Colors.surface,
    borderTopWidth: 1,
    borderTopColor: Colors.divider,
  },
  addButton: {
    flex: 1,
    backgroundColor: Colors.primary,
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: 'center',
  },
  addButtonText: { color: '#FFFFFF', fontWeight: '600', fontSize: 14 },
  presetButton: {
    flex: 1,
    backgroundColor: Colors.background,
    borderRadius: 12,
    paddingVertical: 14,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: Colors.primary,
  },
  presetButtonText: { color: Colors.primary, fontWeight: '600', fontSize: 14 },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    padding: 24,
  },
  modalContent: {
    backgroundColor: Colors.surface,
    borderRadius: 16,
    padding: 24,
  },
  modalTitle: { fontSize: 18, fontWeight: '700', color: Colors.text, marginBottom: 16 },
  input: {
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
    fontSize: 16,
    marginBottom: 12,
  },
  modalButtons: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    gap: 12,
    alignItems: 'center',
    marginTop: 8,
  },
  cancelText: { color: Colors.textSecondary, padding: 8 },
  submitButton: {
    backgroundColor: Colors.primary,
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 10,
  },
  submitText: { color: '#FFFFFF', fontWeight: '600' },
  presetItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: Colors.divider,
  },
  presetItemDisabled: { opacity: 0.4 },
  presetName: { fontSize: 15, color: Colors.text },
  presetNameDisabled: { color: Colors.textSecondary },
  addedLabel: { fontSize: 12, color: Colors.textSecondary },
  closeButton: {
    marginTop: 16,
    alignItems: 'center',
    paddingVertical: 12,
  },
  closeText: { color: Colors.primary, fontWeight: '600', fontSize: 16 },
});
