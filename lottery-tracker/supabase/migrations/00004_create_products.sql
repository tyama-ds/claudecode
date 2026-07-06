CREATE TABLE products (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  group_id UUID NOT NULL REFERENCES groups(id) ON DELETE CASCADE,
  name TEXT NOT NULL,
  release_date DATE,
  is_active BOOLEAN NOT NULL DEFAULT true,
  sort_order INT NOT NULL DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT now()
);

ALTER TABLE products ENABLE ROW LEVEL SECURITY;
CREATE POLICY "products_select" ON products FOR SELECT USING (is_group_member(group_id));
CREATE POLICY "products_insert" ON products FOR INSERT WITH CHECK (is_group_member(group_id));
CREATE POLICY "products_update" ON products FOR UPDATE USING (is_group_member(group_id));
CREATE POLICY "products_delete" ON products FOR DELETE USING (is_group_member(group_id));

CREATE INDEX idx_products_group ON products(group_id);
CREATE INDEX idx_products_active ON products(group_id, is_active);
