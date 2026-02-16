import { useQuery } from '@tanstack/react-query';
import { fetchMemberStats } from '../api';

export function useMemberStats(groupId: string | undefined) {
  return useQuery({
    queryKey: ['member-stats', groupId],
    queryFn: () => fetchMemberStats(groupId!),
    enabled: !!groupId,
  });
}
