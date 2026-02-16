import React from 'react';
import {
  ScrollView,
  TouchableOpacity,
  Text,
  StyleSheet,
  View,
} from 'react-native';
import { Product } from '../../../types';
import { Colors } from '../../../constants/colors';

interface Props {
  products: Product[];
  selectedId: string | undefined;
  onSelect: (product: Product) => void;
  onAdd: () => void;
}

export function ProductTabs({ products, selectedId, onSelect, onAdd }: Props) {
  return (
    <View style={styles.container}>
      <ScrollView
        horizontal
        showsHorizontalScrollIndicator={false}
        contentContainerStyle={styles.scroll}
      >
        {products.map((product) => (
          <TouchableOpacity
            key={product.id}
            style={[
              styles.tab,
              product.id === selectedId && styles.tabActive,
            ]}
            onPress={() => onSelect(product)}
          >
            <Text
              style={[
                styles.tabText,
                product.id === selectedId && styles.tabTextActive,
              ]}
              numberOfLines={1}
            >
              {product.name}
            </Text>
          </TouchableOpacity>
        ))}
        <TouchableOpacity style={styles.addButton} onPress={onAdd}>
          <Text style={styles.addText}>+</Text>
        </TouchableOpacity>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: Colors.surface,
    borderBottomWidth: 1,
    borderBottomColor: Colors.divider,
  },
  scroll: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    gap: 8,
  },
  tab: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: Colors.background,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  tabActive: {
    backgroundColor: Colors.primary,
    borderColor: Colors.primary,
  },
  tabText: {
    fontSize: 14,
    fontWeight: '500',
    color: Colors.text,
  },
  tabTextActive: {
    color: '#FFFFFF',
  },
  addButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: Colors.background,
    borderWidth: 1,
    borderColor: Colors.border,
    justifyContent: 'center',
    alignItems: 'center',
  },
  addText: {
    fontSize: 20,
    color: Colors.primary,
    fontWeight: '600',
  },
});
