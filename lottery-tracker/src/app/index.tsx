import { Redirect } from 'expo-router';
import { ActivityIndicator, View, StyleSheet } from 'react-native';
import { useAuth } from '../providers/auth-provider';
import { useGroup } from '../providers/group-provider';

export default function Index() {
  const { session, loading: authLoading } = useAuth();
  const { currentGroup, loading: groupLoading } = useGroup();

  if (authLoading || groupLoading) {
    return (
      <View style={styles.center}>
        <ActivityIndicator size="large" />
      </View>
    );
  }

  if (!session) {
    return <Redirect href="/(auth)/sign-in" />;
  }

  if (!currentGroup) {
    return <Redirect href="/group/setup" />;
  }

  return <Redirect href="/(tabs)" />;
}

const styles = StyleSheet.create({
  center: { flex: 1, justifyContent: 'center', alignItems: 'center' },
});
