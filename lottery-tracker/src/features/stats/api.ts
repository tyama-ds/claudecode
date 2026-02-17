import { supabase } from '../../lib/supabase';
import { StoreWinRate, MemberStats, ApplicationCorrelation } from '../../types';

export async function fetchStoreWinRates(groupId: string): Promise<StoreWinRate[]> {
  const { data, error } = await supabase.rpc('get_store_win_rates', {
    p_group_id: groupId,
  });
  if (error) throw error;
  return (data ?? []) as StoreWinRate[];
}

export async function fetchMemberStats(groupId: string): Promise<MemberStats[]> {
  const { data, error } = await supabase.rpc('get_member_stats', {
    p_group_id: groupId,
  });
  if (error) throw error;
  return (data ?? []) as MemberStats[];
}

export async function fetchApplicationCorrelation(groupId: string): Promise<ApplicationCorrelation[]> {
  const { data, error } = await supabase.rpc('get_application_correlation', {
    p_group_id: groupId,
  });
  if (error) throw error;
  return (data ?? []) as ApplicationCorrelation[];
}
