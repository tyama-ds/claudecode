import { Slot } from 'expo-router';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { StyleSheet } from 'react-native';
import { AuthProvider } from '../providers/auth-provider';
import { GroupProvider } from '../providers/group-provider';
import { QueryProvider } from '../providers/query-provider';

export default function RootLayout() {
  return (
    <GestureHandlerRootView style={styles.root}>
      <QueryProvider>
        <AuthProvider>
          <GroupProvider>
            <Slot />
          </GroupProvider>
        </AuthProvider>
      </QueryProvider>
    </GestureHandlerRootView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1 },
});
