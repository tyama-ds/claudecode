import { useQuery } from '@tanstack/react-query';
import { fetchStoreWinRates } from '../api';

export function useStoreStats(groupId: string | undefined) {
  return useQuery({
    queryKey: ['store-win-rates', groupId],
    queryFn: () => fetchStoreWinRates(groupId!),
    enabled: !!groupId,
  });
}
