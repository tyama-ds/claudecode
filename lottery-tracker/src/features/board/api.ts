import { supabase } from '../../lib/supabase';
import { ApplicationStatus } from '../../constants/status';
import {
  Product,
  Store,
  LotteryOffering,
  Application,
  BoardOffering,
  ProductSummary,
} from '../../types';

export async function fetchProducts(groupId: string) {
  const { data, error } = await supabase
    .from('products')
    .select('*')
    .eq('group_id', groupId)
    .eq('is_active', true)
    .order('sort_order', { ascending: true });

  if (error) throw error;
  return data as Product[];
}

export async function fetchBoardData(productId: string): Promise<BoardOffering[]> {
  const { data, error } = await supabase
    .from('lottery_offerings')
    .select(`
      *,
      store:stores(*),
      applications(
        *,
        profile:profiles(id, display_name, short_name)
      )
    `)
    .eq('product_id', productId)
    .order('created_at', { ascending: true });

  if (error) throw error;
  return (data ?? []) as unknown as BoardOffering[];
}

export async function fetchProductSummary(productId: string): Promise<ProductSummary[]> {
  const { data, error } = await supabase.rpc('get_product_summary', {
    p_product_id: productId,
  });

  if (error) throw error;
  return (data ?? []) as ProductSummary[];
}

export async function upsertApplication(params: {
  offeringId: string;
  userId: string;
  status: ApplicationStatus;
  quantityWon?: number;
  notes?: string;
}) {
  const { data, error } = await supabase
    .from('applications')
    .upsert(
      {
        offering_id: params.offeringId,
        user_id: params.userId,
        status: params.status,
        quantity_won: params.quantityWon ?? 0,
        notes: params.notes ?? null,
        updated_at: new Date().toISOString(),
      },
      { onConflict: 'offering_id,user_id' }
    )
    .select()
    .single();

  if (error) throw error;
  return data as Application;
}

export async function createProduct(groupId: string, name: string, releaseDate?: string) {
  const { data, error } = await supabase
    .from('products')
    .insert({
      group_id: groupId,
      name,
      release_date: releaseDate ?? null,
    })
    .select()
    .single();

  if (error) throw error;
  return data as Product;
}

export async function createLotteryOffering(params: {
  productId: string;
  storeId: string;
  deadlineEnd?: string;
  lotteryUrl?: string;
  branchInfo?: string;
  method?: 'online' | 'in_store' | 'first_come';
}) {
  const { data, error } = await supabase
    .from('lottery_offerings')
    .insert({
      product_id: params.productId,
      store_id: params.storeId,
      deadline_end: params.deadlineEnd ?? null,
      lottery_url: params.lotteryUrl ?? null,
      branch_info: params.branchInfo ?? null,
      method: params.method ?? null,
    })
    .select(`
      *,
      store:stores(*),
      applications(
        *,
        profile:profiles(id, display_name, short_name)
      )
    `)
    .single();

  if (error) throw error;
  return data as unknown as BoardOffering;
}

export async function toggleOfferingEnded(offeringId: string, isEnded: boolean) {
  const { error } = await supabase
    .from('lottery_offerings')
    .update({ is_ended: isEnded })
    .eq('id', offeringId);

  if (error) throw error;
}
