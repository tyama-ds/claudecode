CREATE TABLE lottery_offerings (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  product_id UUID NOT NULL REFERENCES products(id) ON DELETE CASCADE,
  store_id UUID NOT NULL REFERENCES stores(id) ON DELETE CASCADE,
  deadline_start TIMESTAMPTZ,
  deadline_end TIMESTAMPTZ,
  lottery_url TEXT,
  branch_info TEXT,
  method TEXT CHECK (method IN ('online', 'in_store', 'first_come')),
  is_ended BOOLEAN NOT NULL DEFAULT false,
  notes TEXT,
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE (product_id, store_id)
);

ALTER TABLE lottery_offerings ENABLE ROW LEVEL SECURITY;
CREATE POLICY "offerings_select" ON lottery_offerings FOR SELECT
  USING (EXISTS (
    SELECT 1 FROM products p WHERE p.id = product_id AND is_group_member(p.group_id)
  ));
CREATE POLICY "offerings_insert" ON lottery_offerings FOR INSERT
  WITH CHECK (EXISTS (
    SELECT 1 FROM products p WHERE p.id = product_id AND is_group_member(p.group_id)
  ));
CREATE POLICY "offerings_update" ON lottery_offerings FOR UPDATE
  USING (EXISTS (
    SELECT 1 FROM products p WHERE p.id = product_id AND is_group_member(p.group_id)
  ));
CREATE POLICY "offerings_delete" ON lottery_offerings FOR DELETE
  USING (EXISTS (
    SELECT 1 FROM products p WHERE p.id = product_id AND is_group_member(p.group_id)
  ));

CREATE INDEX idx_offerings_product ON lottery_offerings(product_id);
CREATE INDEX idx_offerings_store ON lottery_offerings(store_id);
