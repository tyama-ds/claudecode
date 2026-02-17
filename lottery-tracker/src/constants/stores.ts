export interface StorePreset {
  name: string;
  url: string | null;
  category: 'ec' | 'retail' | 'card_shop' | 'other';
}

export const COMMON_STORES: StorePreset[] = [
  { name: 'Amazon', url: 'https://www.amazon.co.jp', category: 'ec' },
  { name: 'ポケモンセンターオンライン', url: 'https://www.pokemoncenter-online.com', category: 'ec' },
  { name: 'ヨドバシカメラ', url: 'https://www.yodobashi.com', category: 'retail' },
  { name: 'ビックカメラ', url: 'https://www.biccamera.com', category: 'retail' },
  { name: 'EDION', url: 'https://www.edion.com', category: 'retail' },
  { name: 'ヤマダ電機', url: 'https://www.yamada-denkiweb.com', category: 'retail' },
  { name: 'GEO', url: 'https://geo-online.co.jp', category: 'retail' },
  { name: '7net / omni7', url: 'https://7net.omni7.jp', category: 'ec' },
  { name: '古本市場', url: 'https://www.furu1online.net', category: 'retail' },
  { name: 'あみあみ', url: 'https://www.amiami.jp', category: 'card_shop' },
  { name: 'イエローサブマリン', url: 'https://www.yellowsubmarine.co.jp', category: 'card_shop' },
  { name: 'ホビーステーション', url: 'https://www.hbst.net', category: 'card_shop' },
  { name: 'ジョーシン', url: 'https://joshinweb.jp', category: 'retail' },
  { name: 'トイザらス', url: 'https://www.toysrus.co.jp', category: 'retail' },
  { name: 'TSUTAYA', url: null, category: 'retail' },
  { name: 'カードボックス', url: null, category: 'card_shop' },
  { name: 'ドラスタ', url: 'https://dorasuta.membercard.jp', category: 'card_shop' },
  { name: 'キディランド', url: null, category: 'retail' },
  { name: 'PAO', url: null, category: 'card_shop' },
  { name: 'BM（バトロコ）', url: null, category: 'card_shop' },
  { name: 'フルコンプ', url: null, category: 'card_shop' },
  { name: 'HMV', url: null, category: 'retail' },
  { name: 'イオンスタイルオンライン', url: null, category: 'ec' },
  { name: 'キッズリパブリック', url: null, category: 'ec' },
  { name: 'ファミマオンライン', url: null, category: 'ec' },
  { name: 'イトーヨーカ堂ネット通販', url: null, category: 'ec' },
  { name: 'カードウィングス', url: null, category: 'card_shop' },
  { name: '駿河屋', url: 'https://www.suruga-ya.jp', category: 'card_shop' },
  { name: 'WonderGOO', url: null, category: 'retail' },
  { name: 'Bee本舗', url: null, category: 'card_shop' },
];

export const STORE_CATEGORIES = {
  ec: { label: 'ECサイト', color: '#1976D2' },
  retail: { label: '量販店・小売', color: '#388E3C' },
  card_shop: { label: 'カード専門店', color: '#F57C00' },
  other: { label: 'その他', color: '#757575' },
} as const;
