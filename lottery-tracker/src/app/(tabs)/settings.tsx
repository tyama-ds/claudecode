import React from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  TouchableOpacity,
  Alert,
} from 'react-native';
import * as Clipboard from 'expo-clipboard';
import { useAuth } from '../../providers/auth-provider';
import { useGroup } from '../../providers/group-provider';
import { Colors } from '../../constants/colors';

export default function SettingsScreen() {
  const { profile, signOut } = useAuth();
  const { currentGroup, members } = useGroup();

  async function handleCopyInviteCode() {
    if (!currentGroup?.invite_code) return;
    await Clipboard.setStringAsync(currentGroup.invite_code);
    Alert.alert('コピーしました', '招待コードをクリップボードにコピーしました');
  }

  function handleSignOut() {
    Alert.alert('ログアウト', 'ログアウトしますか？', [
      { text: 'キャンセル', style: 'cancel' },
      {
        text: 'ログアウト',
        style: 'destructive',
        onPress: () => signOut(),
      },
    ]);
  }

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.content}>
      {/* ===== Profile Section ===== */}
      <Text style={styles.sectionTitle}>プロフィール</Text>
      <View style={styles.section}>
        <View style={styles.row}>
          <Text style={styles.rowLabel}>表示名</Text>
          <Text style={styles.rowValue}>{profile?.display_name ?? '-'}</Text>
        </View>
        <View style={styles.divider} />
        <View style={styles.row}>
          <Text style={styles.rowLabel}>短縮名</Text>
          <Text style={styles.rowValue}>{profile?.short_name ?? '-'}</Text>
        </View>
      </View>

      {/* ===== Group Section ===== */}
      <Text style={styles.sectionTitle}>グループ</Text>
      <View style={styles.section}>
        <View style={styles.row}>
          <Text style={styles.rowLabel}>グループ名</Text>
          <Text style={styles.rowValue}>{currentGroup?.name ?? '-'}</Text>
        </View>
        <View style={styles.divider} />
        <View style={styles.row}>
          <Text style={styles.rowLabel}>招待コード</Text>
          <TouchableOpacity
            style={styles.codeButton}
            onPress={handleCopyInviteCode}
          >
            <Text style={styles.codeText}>
              {currentGroup?.invite_code ?? '-'}
            </Text>
            <Text style={styles.copyIcon}>📋</Text>
          </TouchableOpacity>
        </View>
      </View>

      {/* ===== Members Section ===== */}
      <Text style={styles.sectionTitle}>メンバー</Text>
      <View style={styles.section}>
        {members.map((member, index) => (
          <React.Fragment key={member.id}>
            {index > 0 && <View style={styles.divider} />}
            <View style={styles.memberRow}>
              <View style={styles.avatar}>
                <Text style={styles.avatarText}>
                  {member.profile.short_name.charAt(0)}
                </Text>
              </View>
              <View style={styles.memberInfo}>
                <Text style={styles.memberName}>
                  {member.profile.display_name}
                </Text>
                <Text style={styles.memberMeta}>
                  {member.role === 'owner'
                    ? 'オーナー'
                    : member.role === 'admin'
                    ? '管理者'
                    : 'メンバー'}
                </Text>
              </View>
            </View>
          </React.Fragment>
        ))}
        {members.length === 0 && (
          <Text style={styles.emptyText}>メンバーがいません</Text>
        )}
      </View>

      {/* ===== Sign Out ===== */}
      <TouchableOpacity style={styles.signOutButton} onPress={handleSignOut}>
        <Text style={styles.signOutText}>ログアウト</Text>
      </TouchableOpacity>

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
  sectionTitle: {
    fontSize: 18,
    fontWeight: '700',
    color: Colors.text,
    marginTop: 20,
    marginBottom: 12,
  },
  section: {
    backgroundColor: Colors.surface,
    borderRadius: 12,
    padding: 16,
    marginBottom: 4,
  },
  row: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingVertical: 8,
  },
  rowLabel: {
    fontSize: 14,
    color: Colors.textSecondary,
  },
  rowValue: {
    fontSize: 14,
    fontWeight: '600',
    color: Colors.text,
  },
  divider: {
    height: 1,
    backgroundColor: Colors.divider,
  },
  codeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: Colors.background,
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    gap: 6,
  },
  codeText: {
    fontSize: 14,
    fontWeight: '600',
    color: Colors.primary,
    fontFamily: 'monospace',
  },
  copyIcon: {
    fontSize: 14,
  },
  memberRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 10,
    gap: 12,
  },
  avatar: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: Colors.primaryLight,
    justifyContent: 'center',
    alignItems: 'center',
  },
  avatarText: {
    fontSize: 16,
    fontWeight: '700',
    color: Colors.primary,
  },
  memberInfo: {
    flex: 1,
  },
  memberName: {
    fontSize: 14,
    fontWeight: '600',
    color: Colors.text,
  },
  memberMeta: {
    fontSize: 12,
    color: Colors.textSecondary,
    marginTop: 2,
  },
  emptyText: {
    color: Colors.textSecondary,
    fontSize: 14,
    textAlign: 'center',
    paddingVertical: 12,
  },
  signOutButton: {
    marginTop: 32,
    backgroundColor: Colors.surface,
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: Colors.error,
  },
  signOutText: {
    fontSize: 16,
    fontWeight: '600',
    color: Colors.error,
  },
  footer: {
    height: 40,
  },
});
