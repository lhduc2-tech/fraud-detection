-- =========================================================================
-- SCRIPT TẠO BẢNG FRAUD_ALERTS CHO HỆ THỐNG FRAUD DETECTION
-- Thực thi script này trên PostgreSQL database của Fintech
-- =========================================================================

CREATE TABLE IF NOT EXISTS fraud_alerts
(
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    correlation_id          VARCHAR(100) NOT NULL,
    reference_code          VARCHAR(100) NOT NULL,
    user_id                 UUID,
    user_email              VARCHAR(255),
    user_full_name          VARCHAR(255),
    transaction_type        VARCHAR(50) NOT NULL,
    amount                  NUMERIC(19, 2) NOT NULL,
    bank_code               VARCHAR(20),
    account_number          VARCHAR(50),
    recipient_full_name     VARCHAR(255),
    recipient_phone_number  VARCHAR(20),

    -- Điểm số & Đánh giá rủi ro
    final_risk_score        NUMERIC(5, 2) NOT NULL,
    risk_level              VARCHAR(20) NOT NULL, -- LOW, MEDIUM, HIGH
    system_action           TEXT NOT NULL,

    -- Chi tiết các chỉ số AI & Luật
    rule_score              NUMERIC(5, 2) DEFAULT 0,
    ai_score                NUMERIC(5, 2) DEFAULT 0,
    lgbm_prob_pct           NUMERIC(5, 2) DEFAULT 0,
    ae_reconstruction_error NUMERIC(10, 6) DEFAULT 0,
    ae_threshold            NUMERIC(10, 6) DEFAULT 0,

    -- Danh sách cảnh báo & Luật vi phạm (định dạng JSON)
    triggered_rules         JSONB,
    threat_warnings         JSONB,
    payload_snapshot        JSONB,

    -- Trạng thái xử lý
    status                  VARCHAR(30) DEFAULT 'NEW' NOT NULL, -- NEW, REVIEWED, RESOLVED, DISMISSED
    created_at              TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at              TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

-- Tạo Index để tối ưu truy vấn tra cứu theo mã giao dịch, người dùng và mức độ rủi ro
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_reference_code ON fraud_alerts (reference_code);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_correlation_id ON fraud_alerts (correlation_id);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_user_id ON fraud_alerts (user_id);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_risk_level ON fraud_alerts (risk_level);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_status ON fraud_alerts (status);
CREATE INDEX IF NOT EXISTS idx_fraud_alerts_created_at ON fraud_alerts (created_at DESC);

-- Chú thích bảng và các cột
COMMENT ON TABLE fraud_alerts IS 'Bảng lưu vết kết quả sàng lọc rủi ro gian lận giao dịch từ Fraud Detection Engine';
COMMENT ON COLUMN fraud_alerts.correlation_id IS 'Mã UUID theo vết request xuyên suốt các microservice';
COMMENT ON COLUMN fraud_alerts.reference_code IS 'Mã tham chiếu giao dịch ví (referenceCode)';
COMMENT ON COLUMN fraud_alerts.final_risk_score IS 'Điểm rủi ro tổng hợp từ 0.00 đến 100.00';
COMMENT ON COLUMN fraud_alerts.risk_level IS 'Phân cấp rủi ro: LOW, MEDIUM, HIGH';
COMMENT ON COLUMN fraud_alerts.triggered_rules IS 'Danh sách các mã quy tắc nghiệp vụ bị kích hoạt (JSON array)';
COMMENT ON COLUMN fraud_alerts.threat_warnings IS 'Danh sách các thông điệp cảnh báo nguy cơ gian lận (JSON array)';
