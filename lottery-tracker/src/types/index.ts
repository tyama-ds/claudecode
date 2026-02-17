import { ApplicationStatus } from '../constants/status';

export interface Profile {
  id: string;
  display_name: string;
  short_name: string;
  avatar_url: string | null;
  created_at: string;
  updated_at: string;
}

export interface Group {
  id: string;
  name: string;
  invite_code: string;
  created_by: string;
  created_at: string;
}

export interface GroupMember {
  id: string;
  group_id: string;
  user_id: string;
  role: 'owner' | 'admin' | 'member';
  joined_at: string;
}

export interface Store {
  id: string;
  group_id: string;
  name: string;
  url: string | null;
  category: 'ec' | 'retail' | 'card_shop' | 'other';
  notes: string | null;
  sort_order: number;
  is_archived: boolean;
  created_at: string;
}

export interface Product {
  id: string;
  group_id: string;
  name: string;
  release_date: string | null;
  is_active: boolean;
  sort_order: number;
  created_at: string;
}

export interface LotteryOffering {
  id: string;
  product_id: string;
  store_id: string;
  deadline_start: string | null;
  deadline_end: string | null;
  lottery_url: string | null;
  branch_info: string | null;
  method: 'online' | 'in_store' | 'first_come' | null;
  is_ended: boolean;
  notes: string | null;
  created_at: string;
}

export interface Application {
  id: string;
  offering_id: string;
  user_id: string;
  status: ApplicationStatus;
  quantity_won: number;
  notes: string | null;
  updated_at: string;
  created_at: string;
}

// Joined types for board display
export interface BoardOffering extends LotteryOffering {
  store: Store;
  applications: ApplicationWithMember[];
}

export interface ApplicationWithMember extends Application {
  profile: Pick<Profile, 'id' | 'display_name' | 'short_name'>;
}

export interface ProductSummary {
  user_id: string;
  short_name: string;
  total_applied: number;
  total_won: number;
  total_quantity_won: number;
  total_lost: number;
  total_pending: number;
}

// Stats types
export interface StoreWinRate {
  store_id: string;
  store_name: string;
  category: string;
  total_applied: number;
  total_won: number;
  win_rate: number;
}

export interface MemberStats {
  user_id: string;
  short_name: string;
  display_name: string;
  total_applied: number;
  total_won: number;
  total_quantity_won: number;
  win_rate: number;
}

export interface ApplicationCorrelation {
  product_name: string;
  user_short_name: string;
  num_applied: number;
  num_won: number;
  qty_won: number;
}
