import { format } from 'date-fns';
import { Product, BoardOffering, ProductSummary, Profile } from '../types';
import { APPLICATION_STATUS, ApplicationStatus } from '../constants/status';

export function exportBoardAsText(
  product: Product,
  offerings: BoardOffering[],
  members: Pick<Profile, 'id' | 'display_name' | 'short_name'>[],
  summary: ProductSummary[]
): string {
  // Header: 確保状況
  let text = '確保状況：';
  for (const s of summary) {
    if (s.total_quantity_won > 0) {
      text += `${s.short_name}${s.total_quantity_won}`;
    }
  }
  text += `\n＜${product.name}＞\n`;

  // Each store line
  for (const offering of offerings) {
    let line = `・${offering.store.name}`;

    if (offering.branch_info) {
      line += offering.branch_info;
    }

    line += '：';

    // Deadline
    if (offering.deadline_end) {
      const d = new Date(offering.deadline_end);
      line += `~${format(d, 'M/d')}：`;
    }

    // Per-member status
    const memberStatuses: string[] = [];
    for (const member of members) {
      const app = offering.applications.find((a) => a.user_id === member.id);
      if (!app || app.status === 'not_applied') continue;

      const name = member.short_name;
      switch (app.status as ApplicationStatus) {
        case 'applied':
          memberStatuses.push(`${name}☑️`);
          break;
        case 'won':
          memberStatuses.push(`${name}🎊${app.quantity_won > 1 ? app.quantity_won : ''}`);
          break;
        case 'lost':
          memberStatuses.push(`${name}❎`);
          break;
        case 'cancelled':
          memberStatuses.push(`${name}ー`);
          break;
      }
    }

    if (memberStatuses.length === 0) {
      line += '未応募';
    } else {
      line += memberStatuses.join('');
    }

    if (offering.is_ended) {
      line += ' 終了';
    }

    text += line + '\n';
  }

  return text;
}
