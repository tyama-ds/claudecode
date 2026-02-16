CREATE TABLE stores (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  group_id UUID NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  url TEXT,
  category TEXT NOT NULL DEFAULT 'other' CHECK (category IN ('ec', 'retail', 'card_shop', 'other')),
  notes TEXT,
  sort_order INT NOT NULL DEFAULT 0,
  is_archived BOOLEAN NOT NULL DEFAULT false,
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE (group_id, name)
);

ALTER TABLE stores ENABLE ROW LEVEL SECURITY;
CREATE POLICY "stores_select" ON stores FOR SELECT USING (is_group_member(group_id));
CREATE POLICY "stores_insert" ON stores FOR INSERT WITH CHECK (is_group_member(group_id));
CREATE POLICY "stores_update" ON stores FOR UPDATE USING (is_group_member(group_id));
CREATE POLICY "stores_delete" ON stores FOR DELETE USING (is_group_member(group_id));

CREATE INDEX idx_stores_group ON stores(group_id);
