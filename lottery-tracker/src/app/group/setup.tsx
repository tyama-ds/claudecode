import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
} from 'react-native';
import { router } from 'expo-router';
import { useGroup } from '../../providers/group-provider';
import { Colors } from '../../constants/colors';

export default function GroupSetup() {
  const { createGroup, joinGroup } = useGroup();
  const [mode, setMode] = useState<'choose' | 'create' | 'join'>('choose');
  const [groupName, setGroupName] = useState('');
  const [inviteCode, setInviteCode] = useState('');
  const [loading, setLoading] = useState(false);

  async function handleCreate() {
    if (!groupName.trim()) {
      Alert.alert('エラー', 'グループ名を入力してください');
      return;
    }
    setLoading(true);
    const { error } = await createGroup(groupName.trim());
    setLoading(false);
    if (error) {
      Alert.alert('エラー', error.message);
    } else {
      router.replace('/(tabs)');
    }
  }

  async function handleJoin() {
    if (!inviteCode.trim()) {
      Alert.alert('エラー', '招待コードを入力してください');
      return;
    }
    setLoading(true);
    const { error } = await joinGroup(inviteCode.trim());
    setLoading(false);
    if (error) {
      Alert.alert('エラー', error.message);
    } else {
      router.replace('/(tabs)');
    }
  }

  if (mode === 'choose') {
    return (
      <View style={styles.container}>
        <Text style={styles.title}>グループ設定</Text>
        <Text style={styles.subtitle}>
          抽選を一緒に管理するグループを作成または参加してください
        </Text>

        <TouchableOpacity
          style={styles.optionCard}
          onPress={() => setMode('create')}
        >
          <Text style={styles.optionTitle}>新しいグループを作成</Text>
          <Text style={styles.optionDesc}>
            グループを作成して、招待コードで友人を招待します
          </Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.optionCard}
          onPress={() => setMode('join')}
        >
          <Text style={styles.optionTitle}>招待コードで参加</Text>
          <Text style={styles.optionDesc}>
            友人から受け取った招待コードで既存のグループに参加します
          </Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <TouchableOpacity onPress={() => setMode('choose')}>
        <Text style={styles.backLink}>← 戻る</Text>
      </TouchableOpacity>

      <Text style={styles.title}>
        {mode === 'create' ? 'グループ作成' : 'グループ参加'}
      </Text>

      {mode === 'create' ? (
        <>
          <Text style={styles.label}>グループ名</Text>
          <TextInput
            style={styles.input}
            placeholder="例: ポケカ抽選グループ"
            value={groupName}
            onChangeText={setGroupName}
          />
          <TouchableOpacity
            style={[styles.button, loading && styles.buttonDisabled]}
            onPress={handleCreate}
            disabled={loading}
          >
            <Text style={styles.buttonText}>
              {loading ? '作成中...' : 'グループを作成'}
            </Text>
          </TouchableOpacity>
        </>
      ) : (
        <>
          <Text style={styles.label}>招待コード</Text>
          <TextInput
            style={styles.input}
            placeholder="招待コードを入力"
            value={inviteCode}
            onChangeText={setInviteCode}
            autoCapitalize="none"
          />
          <TouchableOpacity
            style={[styles.button, loading && styles.buttonDisabled]}
            onPress={handleJoin}
            disabled={loading}
          >
            <Text style={styles.buttonText}>
              {loading ? '参加中...' : 'グループに参加'}
            </Text>
          </TouchableOpacity>
        </>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
    paddingHorizontal: 24,
    justifyContent: 'center',
  },
  backLink: {
    color: Colors.primary,
    fontSize: 16,
    marginBottom: 20,
  },
  title: {
    fontSize: 24,
    fontWeight: '700',
    color: Colors.text,
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 14,
    color: Colors.textSecondary,
    marginBottom: 32,
  },
  optionCard: {
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: 16,
    padding: 20,
    marginBottom: 16,
  },
  optionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: Colors.text,
    marginBottom: 4,
  },
  optionDesc: {
    fontSize: 14,
    color: Colors.textSecondary,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    color: Colors.text,
    marginBottom: 8,
    marginTop: 16,
  },
  input: {
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 14,
    fontSize: 16,
    marginBottom: 16,
  },
  button: {
    backgroundColor: Colors.primary,
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
    marginTop: 8,
  },
  buttonDisabled: { opacity: 0.6 },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
});
