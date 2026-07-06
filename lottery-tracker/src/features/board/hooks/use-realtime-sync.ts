import { useEffect } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { supabase } from '../../../lib/supabase';

export function useRealtimeSync(productId: string | undefined) {
  const queryClient = useQueryClient();

  useEffect(() => {
    if (!productId) return;

    const channel = supabase
      .channel(`product-${productId}`)
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: 'applications',
        },
        () => {
          // Invalidate and refetch board data when any application changes
          queryClient.invalidateQueries({ queryKey: ['board', productId] });
          queryClient.invalidateQueries({ queryKey: ['product-summary', productId] });
        }
      )
      .subscribe();

    return () => {
      supabase.removeChannel(channel);
    };
  }, [productId, queryClient]);
}
