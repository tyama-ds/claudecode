import React, { useState, useCallback } from 'react';
import { View, Text, StyleSheet, Alert, TextInput, TouchableOpacity, Modal } from 'react-native';
import { ProductTabs } from './components/product-tabs';
import { SummaryHeader } from './components/summary-header';
import { MatrixGrid } from './components/matrix-grid';
import { useProducts, useBoardData, useProductSummary } from './hooks/use-board-data';
import { useStatusToggle } from './hooks/use-status-toggle';
import { useRealtimeSync } from './hooks/use-realtime-sync';
import { useGroup } from '../../providers/group-provider';
import { useAuth } from '../../providers/auth-provider';
import { createProduct, createLotteryOffering } from './api';
import { Product, Store } from '../../types';
import { ApplicationStatus } from '../../constants/status';
import { Colors } from '../../constants/colors';
import { useQueryClient } from '@tanstack/react-query';
import { supabase } from '../../lib/supabase';
import { exportBoardAsText } from '../../utils/text-export';
import * as Clipboard from 'expo-clipboard';

export function BoardScreen() {
  const { currentGroup, members } = useGroup();
  const { user } = useAuth();
  const queryClient = useQueryClient();

  const [selectedProductId, setSelectedProductId] = useState<string>();
  const [showAddProduct, setShowAddProduct] = useState(false);
  const [showAddStore, setShowAddStore] = useState(false);
  const [newProductName, setNewProductName] = useState('');
  const [newStoreName, setNewStoreName] = useState('');

  const { data: products = [] } = useProducts(currentGroup?.id);
  const { data: offerings = [] } = useBoardData(selectedProductId);
  const { data: summary = [] } = useProductSummary(selectedProductId);
  const statusToggle = useStatusToggle();

  // Auto-select first product
  React.useEffect(() => {
    if (products.length > 0 && !selectedProductId) {
      setSelectedProductId(products[0].id);
    }
  }, [products, selectedProductId]);

  useRealtimeSync(selectedProductId);

  const memberProfiles = members.map((m) => ({
    id: m.profile.id,
    display_name: m.profile.display_name,
    short_name: m.profile.short_name,
  }));

  const handleStatusPress = useCallback(
    (offeringId: string, userId: string, currentStatus: ApplicationStatus) => {
      if (!selectedProductId) return;
      statusToggle.mutate({
        offeringId,
        userId,
        currentStatus,
        productId: selectedProductId,
      });
    },
    [selectedProductId, statusToggle]
  );

  const handleStatusLongPress = useCallback(
    (offeringId: string, userId: string, currentStatus: ApplicationStatus, quantityWon: number) => {
      if (currentStatus !== 'won') return;
      Alert.prompt?.(
        '当選数量',
        '確保できた数を入力してください',
        [
          { text: 'キャンセル', style: 'cancel' },
          {
            text: '保存',
            onPress: (value) => {
              const qty = parseInt(value || '1', 10);
              if (qty > 0 && selectedProductId) {
                statusToggle.mutate({
                  offeringId,
                  userId,
                  currentStatus,
                  productId: selectedProductId,
                  quantityWon: qty,
                });
              }
            },
          },
        ],
        'plain-text',
        String(quantityWon || 1)
      );
    },
    [selectedProductId, statusToggle]
  );

  async function handleAddProduct() {
    if (!currentGroup || !newProductName.trim()) return;
    try {
      const product = await createProduct(currentGroup.id, newProductName.trim());
      setSelectedProductId(product.id);
      setNewProductName('');
      setShowAddProduct(false);
      queryClient.invalidateQueries({ queryKey: ['products', currentGroup.id] });
    } catch (e: any) {
      Alert.alert('エラー', e.message);
    }
  }

  async function handleAddStore() {
    if (!selectedProductId || !currentGroup || !newStoreName.trim()) return;
    try {
      // Find or create store
      let storeId: string;
      const { data: existingStore } = await supabase
        .from('stores')
        .select('id')
        .eq('group_id', currentGroup.id)
        .eq('name', newStoreName.trim())
        .single();

      if (existingStore) {
        storeId = existingStore.id;
      } else {
        const { data: newStore, error } = await supabase
          .from('stores')
          .insert({ group_id: currentGroup.id, name: newStoreName.trim() })
          .select('id')
          .single();
        if (error) throw error;
        storeId = newStore!.id;
      }

      await createLotteryOffering({
        productId: selectedProductId,
        storeId,
      });

      setNewStoreName('');
      setShowAddStore(false);
      queryClient.invalidateQueries({ queryKey: ['board', selectedProductId] });
    } catch (e: any) {
      Alert.alert('エラー', e.message);
    }
  }

  async function handleExport() {
    if (!selectedProductId) return;
    const product = products.find((p) => p.id === selectedProductId);
    if (!product) return;

    const text = exportBoardAsText(product, offerings, memberProfiles, summary);
    await Clipboard.setStringAsync(text);
    Alert.alert('コピーしました', 'LINEなどに貼り付けできます');
  }

  return (
    <View style={styles.container}>
      <View style={styles.headerBar}>
        <Text style={styles.groupName}>{currentGroup?.name ?? ''}</Text>
        <TouchableOpacity onPress={handleExport} style={styles.iconButton}>
          <Text style={styles.iconText}>📤</Text>
        </TouchableOpacity>
      </View>

      <ProductTabs
        products={products}
        selectedId={selectedProductId}
        onSelect={(p) => setSelectedProductId(p.id)}
        onAdd={() => setShowAddProduct(true)}
      />

      <SummaryHeader summary={summary} />

      <MatrixGrid
        offerings={offerings}
        members={memberProfiles}
        onStatusPress={handleStatusPress}
        onStatusLongPress={handleStatusLongPress}
        onAddStore={() => setShowAddStore(true)}
      />

      {/* Add Product Modal */}
      <Modal visible={showAddProduct} transparent animationType="fade">
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>商品を追加</Text>
            <TextInput
              style={styles.modalInput}
              placeholder="例: ムニキスゼロ"
              value={newProductName}
              onChangeText={setNewProductName}
              autoFocus
            />
            <View style={styles.modalButtons}>
              <TouchableOpacity onPress={() => setShowAddProduct(false)}>
                <Text style={styles.modalCancel}>キャンセル</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.modalSubmit} onPress={handleAddProduct}>
                <Text style={styles.modalSubmitText}>追加</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>

      {/* Add Store Modal */}
      <Modal visible={showAddStore} transparent animationType="fade">
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <Text style={styles.modalTitle}>ストアを追加</Text>
            <TextInput
              style={styles.modalInput}
              placeholder="例: ヨドバシカメラ"
              value={newStoreName}
              onChangeText={setNewStoreName}
              autoFocus
            />
            <View style={styles.modalButtons}>
              <TouchableOpacity onPress={() => setShowAddStore(false)}>
                <Text style={styles.modalCancel}>キャンセル</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.modalSubmit} onPress={handleAddStore}>
                <Text style={styles.modalSubmitText}>追加</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  headerBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 8,
    backgroundColor: Colors.surface,
    borderBottomWidth: 1,
    borderBottomColor: Colors.divider,
  },
  groupName: {
    fontSize: 16,
    fontWeight: '600',
    color: Colors.text,
  },
  iconButton: {
    padding: 4,
  },
  iconText: {
    fontSize: 20,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.5)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  modalContent: {
    backgroundColor: Colors.surface,
    borderRadius: 16,
    padding: 24,
    width: '100%',
    maxWidth: 400,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: Colors.text,
    marginBottom: 16,
  },
  modalInput: {
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 12,
    fontSize: 16,
    marginBottom: 16,
  },
  modalButtons: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    gap: 12,
    alignItems: 'center',
  },
  modalCancel: {
    fontSize: 14,
    color: Colors.textSecondary,
    padding: 8,
  },
  modalSubmit: {
    backgroundColor: Colors.primary,
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 10,
  },
  modalSubmitText: {
    color: '#FFFFFF',
    fontSize: 14,
    fontWeight: '600',
  },
});
