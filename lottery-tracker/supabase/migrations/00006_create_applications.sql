CREATE TABLE applications (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  offering_id UUID NOT NULL REFERENCES lottery_offerings(id) ON DELETE CASCADE,
  user_id UUID NOT NULL REFERENCES profiles(id) ON DELETE CASCADE,
  status TEXT NOT NULL DEFAULT 'not_applied'
    CHECK (status IN ('not_applied', 'applied', 'won', 'lost', 'cancelled')),
  quantity_won INT NOT NULL DEFAULT 0,
  notes TEXT,
  updated_at TIMESTAMPTZ DEFAULT now(),
  created_at TIMESTAMPTZ DEFAULT now(),
  UNIQUE (offering_id, user_id)
);

ALTER TABLE applications ENABLE ROW LEVEL SECURITY;
CREATE POLICY "apps_select" ON applications FOR SELECT
  USING (EXISTS (
    SELECT 1 FROM lottery_offerings lo
    JOIN products p ON p.id = lo.product_id
    WHERE lo.id = offering_id AND is_group_member(p.group_id)
  ));
CREATE POLICY "apps_insert" ON applications FOR INSERT
  WITH CHECK (EXISTS (
    SELECT 1 FROM lottery_offerings lo
    JOIN products p ON p.id = lo.product_id
    WHERE lo.id = offering_id AND is_group_member(p.group_id)
  ));
CREATE POLICY "apps_update" ON applications FOR UPDATE
  USING (EXISTS (
    SELECT 1 FROM lottery_offerings lo
    JOIN products p ON p.id = lo.product_id
    WHERE lo.id = offering_id AND is_group_member(p.group_id)
  ));
CREATE POLICY "apps_delete" ON applications FOR DELETE
  USING (EXISTS (
    SELECT 1 FROM lottery_offerings lo
    JOIN products p ON p.id = lo.product_id
    WHERE lo.id = offering_id AND is_group_member(p.group_id)
  ));

CREATE INDEX idx_applications_offering ON applications(offering_id);
CREATE INDEX idx_applications_user ON applications(user_id);
CREATE INDEX idx_applications_status ON applications(offering_id, status);
