import React, { createContext, useContext, useEffect, useState } from 'react';
import { supabase } from '../lib/supabase';
import { useAuth } from './auth-provider';
import { Group, GroupMember, Profile } from '../types';

interface GroupContextType {
  currentGroup: Group | null;
  members: (GroupMember & { profile: Profile })[];
  loading: boolean;
  setCurrentGroup: (group: Group | null) => void;
  createGroup: (name: string) => Promise<{ group: Group | null; error: Error | null }>;
  joinGroup: (inviteCode: string) => Promise<{ error: Error | null }>;
  refreshMembers: () => Promise<void>;
}

const GroupContext = createContext<GroupContextType | undefined>(undefined);

export function GroupProvider({ children }: { children: React.ReactNode }) {
  const { user } = useAuth();
  const [currentGroup, setCurrentGroup] = useState<Group | null>(null);
  const [members, setMembers] = useState<(GroupMember & { profile: Profile })[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (user) {
      loadUserGroup();
    } else {
      setCurrentGroup(null);
      setMembers([]);
      setLoading(false);
    }
  }, [user]);

  useEffect(() => {
    if (currentGroup) {
      fetchMembers();
    }
  }, [currentGroup?.id]);

  async function loadUserGroup() {
    setLoading(true);
    const { data } = await supabase
      .from('group_members')
      .select('group_id, groups(*)')
      .eq('user_id', user!.id)
      .limit(1)
      .single();

    if (data?.groups) {
      setCurrentGroup(data.groups as unknown as Group);
    }
    setLoading(false);
  }

  async function fetchMembers() {
    if (!currentGroup) return;
    const { data } = await supabase
      .from('group_members')
      .select('*, profile:profiles(*)')
      .eq('group_id', currentGroup.id)
      .order('joined_at', { ascending: true });

    if (data) {
      setMembers(data as unknown as (GroupMember & { profile: Profile })[]);
    }
  }

  async function createGroup(name: string) {
    if (!user) return { group: null, error: new Error('Not authenticated') };

    const { data, error } = await supabase
      .from('groups')
      .insert({ name, created_by: user.id })
      .select()
      .single();

    if (error) return { group: null, error: new Error(error.message) };

    const group = data as Group;

    // Add creator as owner
    await supabase.from('group_members').insert({
      group_id: group.id,
      user_id: user.id,
      role: 'owner',
    });

    setCurrentGroup(group);
    return { group, error: null };
  }

  async function joinGroup(inviteCode: string) {
    if (!user) return { error: new Error('Not authenticated') };

    const { data: group, error: findError } = await supabase
      .from('groups')
      .select('*')
      .eq('invite_code', inviteCode.trim())
      .single();

    if (findError || !group) return { error: new Error('招待コードが見つかりません') };

    const { error: joinError } = await supabase.from('group_members').insert({
      group_id: group.id,
      user_id: user.id,
      role: 'member',
    });

    if (joinError) {
      if (joinError.code === '23505') {
        return { error: new Error('既にこのグループに参加しています') };
      }
      return { error: new Error(joinError.message) };
    }

    setCurrentGroup(group as Group);
    return { error: null };
  }

  return (
    <GroupContext.Provider
      value={{
        currentGroup,
        members,
        loading,
        setCurrentGroup,
        createGroup,
        joinGroup,
        refreshMembers: fetchMembers,
      }}
    >
      {children}
    </GroupContext.Provider>
  );
}

export function useGroup() {
  const context = useContext(GroupContext);
  if (!context) {
    throw new Error('useGroup must be used within GroupProvider');
  }
  return context;
}
