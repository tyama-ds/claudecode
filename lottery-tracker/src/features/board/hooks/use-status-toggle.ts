import { useMutation, useQueryClient } from '@tanstack/react-query';
import * as Haptics from 'expo-haptics';
import { upsertApplication } from '../api';
import { ApplicationStatus, getNextStatus } from '../../../constants/status';
import { BoardOffering } from '../../../types';

interface ToggleParams {
  offeringId: string;
  userId: string;
  currentStatus: ApplicationStatus;
  productId: string;
  quantityWon?: number;
}

export function useStatusToggle() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: async (params: ToggleParams) => {
      const newStatus = params.quantityWon !== undefined
        ? params.currentStatus // When setting quantity, keep current status
        : getNextStatus(params.currentStatus);

      return upsertApplication({
        offeringId: params.offeringId,
        userId: params.userId,
        status: newStatus,
        quantityWon: newStatus === 'won' ? (params.quantityWon ?? 1) : 0,
      });
    },

    onMutate: async (params) => {
      const queryKey = ['board', params.productId];
      await queryClient.cancelQueries({ queryKey });

      const previous = queryClient.getQueryData<BoardOffering[]>(queryKey);

      // Optimistic update
      queryClient.setQueryData<BoardOffering[]>(queryKey, (old) => {
        if (!old) return old;
        return old.map((offering) => {
          if (offering.id !== params.offeringId) return offering;

          const newStatus = params.quantityWon !== undefined
            ? params.currentStatus
            : getNextStatus(params.currentStatus);

          const existingApp = offering.applications.find(
            (a) => a.user_id === params.userId
          );

          if (existingApp) {
            return {
              ...offering,
              applications: offering.applications.map((a) =>
                a.user_id === params.userId
                  ? {
                      ...a,
                      status: newStatus,
                      quantity_won: newStatus === 'won' ? (params.quantityWon ?? 1) : 0,
                    }
                  : a
              ),
            };
          }

          // New application
          return {
            ...offering,
            applications: [
              ...offering.applications,
              {
                id: `temp-${Date.now()}`,
                offering_id: params.offeringId,
                user_id: params.userId,
                status: newStatus,
                quantity_won: newStatus === 'won' ? (params.quantityWon ?? 1) : 0,
                notes: null,
                updated_at: new Date().toISOString(),
                created_at: new Date().toISOString(),
                profile: { id: params.userId, display_name: '', short_name: '' },
              },
            ],
          };
        });
      });

      Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);

      return { previous };
    },

    onError: (_err, params, context) => {
      if (context?.previous) {
        queryClient.setQueryData(['board', params.productId], context.previous);
      }
    },

    onSettled: (_data, _err, params) => {
      queryClient.invalidateQueries({ queryKey: ['board', params.productId] });
      queryClient.invalidateQueries({ queryKey: ['product-summary', params.productId] });
    },
  });
}
