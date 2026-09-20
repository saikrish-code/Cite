-- Add Rate Limiting Table
CREATE TABLE user_rate_limits (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    request_count INT DEFAULT 0,
    last_reset TIMESTAMPTZ DEFAULT NOW()
);

-- Enable RLS (Service role will bypass this, but good practice)
ALTER TABLE user_rate_limits ENABLE ROW LEVEL SECURITY;

CREATE OR REPLACE FUNCTION check_and_increment_rate_limit(
    p_user_id UUID,
    p_max_requests INT,
    p_window_minutes INT
) RETURNS BOOLEAN
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    v_last_reset TIMESTAMPTZ;
    v_count INT;
BEGIN
    -- Get current limit state
    SELECT request_count, last_reset INTO v_count, v_last_reset
    FROM user_rate_limits
    WHERE user_id = p_user_id;

    IF NOT FOUND THEN
        -- First request ever
        INSERT INTO user_rate_limits (user_id, request_count, last_reset)
        VALUES (p_user_id, 1, NOW());
        RETURN TRUE;
    END IF;

    -- Check if window expired
    IF NOW() > v_last_reset + (p_window_minutes || ' minutes')::interval THEN
        UPDATE user_rate_limits
        SET request_count = 1, last_reset = NOW()
        WHERE user_id = p_user_id;
        RETURN TRUE;
    END IF;

    -- Check limit
    IF v_count >= p_max_requests THEN
        RETURN FALSE;
    END IF;

    -- Increment
    UPDATE user_rate_limits
    SET request_count = request_count + 1
    WHERE user_id = p_user_id;
    
    RETURN TRUE;
END;
$$;
