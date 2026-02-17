-- Get per-member summary for a product
CREATE OR REPLACE FUNCTION get_product_summary(p_product_id UUID)
RETURNS TABLE (
  user_id UUID,
  short_name TEXT,
  total_applied BIGINT,
  total_won BIGINT,
  total_quantity_won BIGINT,
  total_lost BIGINT,
  total_pending BIGINT
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    a.user_id,
    p.short_name,
    COUNT(*) FILTER (WHERE a.status != 'not_applied') AS total_applied,
    COUNT(*) FILTER (WHERE a.status = 'won') AS total_won,
    COALESCE(SUM(a.quantity_won), 0) AS total_quantity_won,
    COUNT(*) FILTER (WHERE a.status = 'lost') AS total_lost,
    COUNT(*) FILTER (WHERE a.status = 'applied') AS total_pending
  FROM applications a
  JOIN profiles p ON p.id = a.user_id
  JOIN lottery_offerings lo ON lo.id = a.offering_id
  WHERE lo.product_id = p_product_id
  GROUP BY a.user_id, p.short_name;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Get win rates per store for a group
CREATE OR REPLACE FUNCTION get_store_win_rates(p_group_id UUID)
RETURNS TABLE (
  store_id UUID,
  store_name TEXT,
  category TEXT,
  total_applied BIGINT,
  total_won BIGINT,
  win_rate NUMERIC
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    s.id AS store_id,
    s.name AS store_name,
    s.category,
    COUNT(*) FILTER (WHERE a.status IN ('applied', 'won', 'lost')) AS total_applied,
    COUNT(*) FILTER (WHERE a.status = 'won') AS total_won,
    CASE
      WHEN COUNT(*) FILTER (WHERE a.status IN ('won', 'lost')) > 0
      THEN ROUND(
        COUNT(*) FILTER (WHERE a.status = 'won')::NUMERIC /
        COUNT(*) FILTER (WHERE a.status IN ('won', 'lost'))::NUMERIC * 100, 1
      )
      ELSE 0
    END AS win_rate
  FROM stores s
  JOIN lottery_offerings lo ON lo.store_id = s.id
  JOIN applications a ON a.offering_id = lo.id
  WHERE s.group_id = p_group_id
  GROUP BY s.id, s.name, s.category
  ORDER BY win_rate DESC;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Get member history across products for a group
CREATE OR REPLACE FUNCTION get_member_stats(p_group_id UUID)
RETURNS TABLE (
  user_id UUID,
  short_name TEXT,
  display_name TEXT,
  total_applied BIGINT,
  total_won BIGINT,
  total_quantity_won BIGINT,
  win_rate NUMERIC
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    a.user_id,
    pr.short_name,
    pr.display_name,
    COUNT(*) FILTER (WHERE a.status IN ('applied', 'won', 'lost')) AS total_applied,
    COUNT(*) FILTER (WHERE a.status = 'won') AS total_won,
    COALESCE(SUM(a.quantity_won), 0) AS total_quantity_won,
    CASE
      WHEN COUNT(*) FILTER (WHERE a.status IN ('won', 'lost')) > 0
      THEN ROUND(
        COUNT(*) FILTER (WHERE a.status = 'won')::NUMERIC /
        COUNT(*) FILTER (WHERE a.status IN ('won', 'lost'))::NUMERIC * 100, 1
      )
      ELSE 0
    END AS win_rate
  FROM applications a
  JOIN lottery_offerings lo ON lo.id = a.offering_id
  JOIN products p ON p.id = lo.product_id
  JOIN profiles pr ON pr.id = a.user_id
  WHERE p.group_id = p_group_id
  GROUP BY a.user_id, pr.short_name, pr.display_name;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Get application correlation data (applications vs wins per product per member)
CREATE OR REPLACE FUNCTION get_application_correlation(p_group_id UUID)
RETURNS TABLE (
  product_id UUID,
  product_name TEXT,
  user_id UUID,
  user_short_name TEXT,
  num_applied BIGINT,
  num_won BIGINT,
  qty_won BIGINT
) AS $$
BEGIN
  RETURN QUERY
  SELECT
    p.id AS product_id,
    p.name AS product_name,
    a.user_id,
    pr.short_name AS user_short_name,
    COUNT(*) FILTER (WHERE a.status IN ('applied', 'won', 'lost')) AS num_applied,
    COUNT(*) FILTER (WHERE a.status = 'won') AS num_won,
    COALESCE(SUM(a.quantity_won), 0) AS qty_won
  FROM applications a
  JOIN lottery_offerings lo ON lo.id = a.offering_id
  JOIN products p ON p.id = lo.product_id
  JOIN profiles pr ON pr.id = a.user_id
  WHERE p.group_id = p_group_id
  GROUP BY p.id, p.name, a.user_id, pr.short_name
  ORDER BY p.created_at DESC, pr.short_name;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Enable realtime for applications table
ALTER PUBLICATION supabase_realtime ADD TABLE applications;
