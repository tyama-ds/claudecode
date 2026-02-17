export type ApplicationStatus = 'not_applied' | 'applied' | 'won' | 'lost' | 'cancelled';

export const APPLICATION_STATUS: Record<
  ApplicationStatus,
  {
    key: ApplicationStatus;
    label: string;
    emoji: string;
    color: string;
    bgColor: string;
  }
> = {
  not_applied: {
    key: 'not_applied',
    label: '未応募',
    emoji: '',
    color: '#9E9E9E',
    bgColor: '#F5F5F5',
  },
  applied: {
    key: 'applied',
    label: '応募済',
    emoji: '☑️',
    color: '#1976D2',
    bgColor: '#E3F2FD',
  },
  won: {
    key: 'won',
    label: '当選',
    emoji: '🎊',
    color: '#388E3C',
    bgColor: '#E8F5E9',
  },
  lost: {
    key: 'lost',
    label: '落選',
    emoji: '❎',
    color: '#D32F2F',
    bgColor: '#FFEBEE',
  },
  cancelled: {
    key: 'cancelled',
    label: 'キャンセル',
    emoji: 'ー',
    color: '#757575',
    bgColor: '#F5F5F5',
  },
};

export const STATUS_CYCLE: ApplicationStatus[] = [
  'not_applied',
  'applied',
  'won',
  'lost',
];

export function getNextStatus(current: ApplicationStatus): ApplicationStatus {
  const idx = STATUS_CYCLE.indexOf(current);
  return STATUS_CYCLE[(idx + 1) % STATUS_CYCLE.length];
}
