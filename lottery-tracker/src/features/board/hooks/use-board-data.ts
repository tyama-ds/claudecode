import { useQuery } from '@tanstack/react-query';
import { fetchProducts, fetchBoardData, fetchProductSummary } from '../api';

export function useProducts(groupId: string | undefined) {
  return useQuery({
    queryKey: ['products', groupId],
    queryFn: () => fetchProducts(groupId!),
    enabled: !!groupId,
  });
}

export function useBoardData(productId: string | undefined) {
  return useQuery({
    queryKey: ['board', productId],
    queryFn: () => fetchBoardData(productId!),
    enabled: !!productId,
  });
}

export function useProductSummary(productId: string | undefined) {
  return useQuery({
    queryKey: ['product-summary', productId],
    queryFn: () => fetchProductSummary(productId!),
    enabled: !!productId,
  });
}
