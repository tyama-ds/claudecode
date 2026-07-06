export const Colors = {
  primary: '#1976D2',
  primaryLight: '#BBDEFB',
  primaryDark: '#0D47A1',
  secondary: '#FF6F00',
  background: '#FAFAFA',
  surface: '#FFFFFF',
  text: '#212121',
  textSecondary: '#757575',
  textLight: '#BDBDBD',
  border: '#E0E0E0',
  divider: '#EEEEEE',
  error: '#D32F2F',
  success: '#388E3C',
  warning: '#F57C00',

  status: {
    notApplied: { bg: '#F5F5F5', text: '#9E9E9E' },
    applied: { bg: '#E3F2FD', text: '#1976D2' },
    won: { bg: '#E8F5E9', text: '#388E3C' },
    lost: { bg: '#FFEBEE', text: '#D32F2F' },
    cancelled: { bg: '#F5F5F5', text: '#757575' },
  },
} as const;
