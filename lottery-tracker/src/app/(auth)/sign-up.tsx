import React, { useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  Alert,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
} from 'react-native';
import { Link, router } from 'expo-router';
import { useAuth } from '../../providers/auth-provider';
import { Colors } from '../../constants/colors';

export default function SignUp() {
  const { signUp } = useAuth();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [shortName, setShortName] = useState('');
  const [loading, setLoading] = useState(false);

  async function handleSignUp() {
    if (!email || !password || !displayName || !shortName) {
      Alert.alert('エラー', '全ての項目を入力してください');
      return;
    }
    if (shortName.length > 2) {
      Alert.alert('エラー', '略称は2文字以内で入力してください');
      return;
    }
    setLoading(true);
    const { error } = await signUp(email, password, displayName, shortName);
    setLoading(false);
    if (error) {
      Alert.alert('登録エラー', error.message);
    } else {
      router.replace('/');
    }
  }

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
    >
      <ScrollView contentContainerStyle={styles.inner} keyboardShouldPersistTaps="handled">
        <Text style={styles.title}>新規登録</Text>
        <Text style={styles.subtitle}>アカウントを作成してください</Text>

        <Text style={styles.label}>表示名</Text>
        <TextInput
          style={styles.input}
          placeholder="例: たかし"
          value={displayName}
          onChangeText={setDisplayName}
        />

        <Text style={styles.label}>略称（1-2文字）</Text>
        <Text style={styles.hint}>ボード上で表示される短い名前です</Text>
        <TextInput
          style={styles.input}
          placeholder="例: た"
          value={shortName}
          onChangeText={setShortName}
          maxLength={2}
        />

        <Text style={styles.label}>メールアドレス</Text>
        <TextInput
          style={styles.input}
          placeholder="example@mail.com"
          value={email}
          onChangeText={setEmail}
          autoCapitalize="none"
          keyboardType="email-address"
          textContentType="emailAddress"
        />

        <Text style={styles.label}>パスワード</Text>
        <TextInput
          style={styles.input}
          placeholder="6文字以上"
          value={password}
          onChangeText={setPassword}
          secureTextEntry
          textContentType="newPassword"
        />

        <TouchableOpacity
          style={[styles.button, loading && styles.buttonDisabled]}
          onPress={handleSignUp}
          disabled={loading}
        >
          <Text style={styles.buttonText}>
            {loading ? '登録中...' : '登録する'}
          </Text>
        </TouchableOpacity>

        <Link href="/(auth)/sign-in" asChild>
          <TouchableOpacity style={styles.linkButton}>
            <Text style={styles.linkText}>
              アカウントをお持ちの方は<Text style={styles.linkBold}>ログイン</Text>
            </Text>
          </TouchableOpacity>
        </Link>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: Colors.background,
  },
  inner: {
    flexGrow: 1,
    justifyContent: 'center',
    paddingHorizontal: 24,
    paddingVertical: 40,
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    color: Colors.text,
    textAlign: 'center',
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    color: Colors.textSecondary,
    textAlign: 'center',
    marginBottom: 32,
  },
  label: {
    fontSize: 14,
    fontWeight: '600',
    color: Colors.text,
    marginBottom: 4,
    marginTop: 8,
  },
  hint: {
    fontSize: 12,
    color: Colors.textSecondary,
    marginBottom: 4,
  },
  input: {
    backgroundColor: Colors.surface,
    borderWidth: 1,
    borderColor: Colors.border,
    borderRadius: 12,
    paddingHorizontal: 16,
    paddingVertical: 14,
    fontSize: 16,
    marginBottom: 8,
  },
  button: {
    backgroundColor: Colors.primary,
    borderRadius: 12,
    paddingVertical: 16,
    alignItems: 'center',
    marginTop: 16,
  },
  buttonDisabled: {
    opacity: 0.6,
  },
  buttonText: {
    color: '#FFFFFF',
    fontSize: 16,
    fontWeight: '600',
  },
  linkButton: {
    marginTop: 20,
    alignItems: 'center',
  },
  linkText: {
    color: Colors.textSecondary,
    fontSize: 14,
  },
  linkBold: {
    color: Colors.primary,
    fontWeight: '600',
  },
});
